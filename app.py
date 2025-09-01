#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rankeador de CVs — LLM Scoring (rápido, 5 niveles de veredicto, sin exponer secrets)
- Subes hasta 10 PDFs
- El LLM evalúa y devuelve JSON por CV
- Tabla final + descarga Excel (fallback CSV)
- Paraleliza evaluaciones y reduce tokens (modo rápido)
"""
from __future__ import annotations

import io
import os
import re
import json
import base64
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
import pdfplumber
from dotenv import load_dotenv

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("Falta el paquete 'openai'. Instala: pip install openai")

# ==========================
# Configuración base
# ==========================
load_dotenv()
st.set_page_config(page_title="📊 Rankeador de CVs — LLM Fast", page_icon="📄", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL_DEFAULT = "gpt-5"            # puedes cambiar a gpt-5-mini / gpt-5-nano en modo admin
MAX_PDFS_FREE = 10

# Admin panel visible solo si lo habilitas en Secrets
SHOW_ADMIN = str(st.secrets.get("SHOW_ADMIN", "false")).lower() == "true"

# ==========================
# Estado global (UX)
# ==========================
if "busy" not in st.session_state:
    st.session_state["busy"] = False
if "evaluaciones" not in st.session_state:
    st.session_state["evaluaciones"] = None

def set_busy(flag: bool):
    st.session_state["busy"] = flag

# ==========================
# Helpers
# ==========================
def read_pdf_text(file) -> str:
    parts: List[str] = []
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            parts.append(p.extract_text() or "")
    return "\n\n".join(parts)

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or " ").strip()

def windows_around_skills(text: str, skills: List[str], radius: int = 220, max_windows: int = 8) -> str:
    """
    Recorta el CV a ventanas alrededor de cada skill (reduce tokens y acelera).
    """
    windows = []
    for s in skills:
        for m in re.finditer(rf"(.{{0,{radius}}}\b{s}\b.{{0,{radius}}})", text, flags=re.IGNORECASE):
            windows.append(m.group(1))
            if len(windows) >= max_windows:
                break
        if len(windows) >= max_windows:
            break
    return "\n\n".join(windows)

def safe_json_from_text(txt: str) -> Dict[str, Any]:
    if not txt:
        return {}
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"(\{.*\})", txt, flags=re.DOTALL)
    if m2:
        candidate = m2.group(1)
        # intenta recortar hasta que parsee
        for cut in range(len(candidate), max(len(candidate) - 4000, 100), -50):
            try:
                return json.loads(candidate[:cut])
            except Exception:
                continue
    return {}

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Resultados") -> Optional[bytes]:
    """
    Intenta exportar Excel con openpyxl; si no está instalado, devuelve None (usaremos CSV).
    """
    try:
        import openpyxl  # noqa: F401
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name=sheet_name)
        bio.seek(0)
        return bio.getvalue()
    except Exception:
        return None

def b64_download_link(data: bytes, filename: str, label: str, mime: str = "application/octet-stream") -> str:
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{label}</a>'
    return href

# ==========================
# Sidebar (oculto para usuarios)
# ==========================
st.sidebar.title("⚙️ Configuración")

# API Key: oculta siempre que exista en secrets/env
api_key_input = OPENAI_API_KEY
if not api_key_input:
    if SHOW_ADMIN:
        api_key_input = st.sidebar.text_input("OpenAI API Key", value="", type="password")
    else:
        st.sidebar.error("Configura OPENAI_API_KEY en Secrets.")
else:
    st.sidebar.success("🔑 API Key cargada.")

# Modelo visible solo en admin
if SHOW_ADMIN:
    CHAT_MODEL = st.sidebar.selectbox(
        "Modelo de Chat",
        options=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        index=0,
        help="Solo visible para admin.",
        disabled=st.session_state["busy"]
    )
else:
    CHAT_MODEL = CHAT_MODEL_DEFAULT

# Modo rápido (fijo ON para usuarios)
FAST_MODE = True
st.sidebar.caption(f"Modelo: {CHAT_MODEL} • Modo rápido: {'ON' if FAST_MODE else 'OFF'}")

# ==========================
# Main UI
# ==========================
st.title("📊 Rankeador de CVs — LLM scoring (rápido, 5 niveles)")
st.caption("El LLM evalúa y puntúa. Salida: nombre, resumen corto, skills, veredicto (5 niveles), veredicto_detallado, calificación 0–100.")

col_left, col_right = st.columns([1, 1])
with col_left:
    cargo = st.text_area(
        "Cargo / Descripción del puesto",
        placeholder="Científico/a de Datos Senior en Colombia. Python, SQL, Machine Learning, arquitectura de datos, MLOps…",
        height=130,
        disabled=st.session_state["busy"]
    )
with col_right:
    skills_raw = st.text_input(
        "Skills requeridos (coma separada)",
        value="Python, SQL, Machine Learning",
        disabled=st.session_state["busy"]
    )
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

files = st.file_uploader(
    "Sube CVs en PDF (hasta 10)",
    type=["pdf"], accept_multiple_files=True,
    disabled=st.session_state["busy"]
)
if files and len(files) > MAX_PDFS_FREE:
    st.warning(f"Solo se permiten {MAX_PDFS_FREE} PDFs por sesión. Tomaré los primeros {MAX_PDFS_FREE}.")
    files = files[:MAX_PDFS_FREE]

run = st.button("🧠 Evaluar y generar tabla", disabled=st.session_state["busy"])

# ==========================
# Prompt y evaluación
# ==========================
def build_prompt(name: str, cargo: str, skills: List[str], context: str) -> str:
    return f"""
Eres un reclutador técnico en Colombia. Evalúa el CV para el cargo descrito.

CARGO / DESCRIPCIÓN:
{cargo}

SKILLS REQUERIDAS (coma separada): {', '.join(skills) if skills else '(no especificadas)'}

TEXTO DEL CV (extracto):
{context}

Devuelve SOLO un JSON con esta forma EXACTA (sin texto adicional):
{{
  "nombre": "{name}",
  "resumen_corto": "2-3 frases o 3-5 viñetas concisas sobre la experiencia más relevante",
  "skills_detectadas": ["skill1", "skill2", "..."],
  "veredicto": "Excelente|Muy Bueno|Regular|Débil|Muy Débil",
  "veredicto_detallado": "frase clara: ¿sirve para el puesto? ¿qué riesgos/gaps hay?",
  "calificacion": 0-100
}}
Criterios de evaluación (usa estos rangos y etiquetas EXACTAS):
- Excelente (80–100): candidato ideal, cumple casi todo.
- Muy Bueno (61–80): buen ajuste, con algunos gaps.
- Regular (41–60): encaje parcial, riesgos importantes.
- Débil (21–40): pocos requisitos cumplidos.
- Muy Débil (0–20): muy lejos del cargo.
"""

def llm_evaluate_one(client: OpenAI, name: str, text: str, cargo: str, skills: List[str]) -> Dict[str, Any]:
    # Contexto rápido: base recortada + ventanas alrededor de skills
    base = text[:2500]
    skill_windows = windows_around_skills(text, skills, radius=220, max_windows=8)
    context = (base + "\n\n" + skill_windows).strip()[:7000] if FAST_MODE else text[:12000]

    prompt = build_prompt(name, cargo, skills, context)
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Eres un reclutador técnico conciso, objetivo y estricto con el formato JSON."},
                {"role": "user", "content": prompt},
            ]
            # sin temperature/top_p: la familia gpt-5 no soporta otros valores
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        content = f'{{"nombre":"{name}","resumen_corto":"Error al invocar el modelo","skills_detectadas":[],"veredicto":"Muy Débil","veredicto_detallado":"No se pudo evaluar","calificacion":0,"_error":"{e}"}}'

    data = safe_json_from_text(content)
    # Defaults / normalización
    data.setdefault("nombre", name)
    data.setdefault("resumen_corto", "")
    data.setdefault("skills_detectadas", [])
    data.setdefault("veredicto", "Muy Débil")
    data.setdefault("veredicto_detallado", "")
    try:
        score = int(data.get("calificacion", 0))
    except Exception:
        score = 0
    data["calificacion"] = max(0, min(100, score))
    return data

# ==========================
# Ejecución
# ==========================
if run:
    if not api_key_input:
        st.error("Falta OPENAI_API_KEY. Configura el secreto en Streamlit Cloud.")
        st.stop()
    if not cargo.strip():
        st.error("Describe el cargo a evaluar.")
        st.stop()
    if not files:
        st.error("Sube al menos 1 PDF.")
        st.stop()

    client = OpenAI(api_key=api_key_input)
    set_busy(True)

    with st.status("Procesando CVs…", expanded=True) as status:
        status.write("📥 Extrayendo texto…")
        names, texts = [], []
        for f in files:
            try:
                t = clean_text(read_pdf_text(f))
            except Exception as e:
                t = ""
                st.warning(f"No pude leer {f.name}: {e}")
            names.append(f.name)
            texts.append(t)

        status.write("🧠 Evaluando con LLM (paralelo)…")
        progress = st.progress(0, text="Enviando solicitudes…")

        resultados: List[Dict[str, Any]] = []
        done_count = 0

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(llm_evaluate_one, client, n, t, cargo, skills): n for n, t in zip(names, texts)}
            for fut in as_completed(futures):
                data = fut.result()
                resultados.append(data)
                done_count += 1
                progress.progress(int(done_count / len(names) * 100), text=f"Completados {done_count}/{len(names)}")

        progress.empty()
        status.update(label="✅ Evaluación completa", state="complete")

    # Tabla final
    df = pd.DataFrame(resultados, columns=[
        "nombre", "resumen_corto", "skills_detectadas", "veredicto", "veredicto_detallado", "calificacion"
    ]).sort_values("calificacion", ascending=False).reset_index(drop=True)
    st.session_state["evaluaciones"] = df
    set_busy(False)

# ==========================
# Resultados + Descarga
# ==========================
if st.session_state["evaluaciones"] is not None:
    df = st.session_state["evaluaciones"]
    st.subheader("🏁 Resultados (según LLM — 5 niveles)")
    st.dataframe(df, use_container_width=True)

    # Excel preferido
    xlsx = df_to_excel_bytes(df, sheet_name="Resultados")
    if xlsx is not None:
        st.markdown(b64_download_link(xlsx, "ranking_llm.xlsx", "⬇️ Descargar Excel"), unsafe_allow_html=True)
    else:
        # Fallback CSV
        st.warning("No se encontró openpyxl. Descargando CSV (instala openpyxl para Excel).")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.markdown(b64_download_link(csv_bytes, "ranking_llm.csv", "⬇️ Descargar CSV"), unsafe_allow_html=True)

# ==========================
# Footer visual (para LinkedIn)
# ==========================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 15px; line-height:1.4;">
        <p>📩 <b>Contacto:</b> Andrés Grisales Ardila</p>
        <p>✉️ <a href="mailto:agrisalesa@unal.edu.co">agrisalesa@unal.edu.co</a></p>
        <p>🔗 <a href="https://www.linkedin.com/in/andres-grisales-ardila/" target="_blank">linkedin.com/in/andres-grisales-ardila</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

