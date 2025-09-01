#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluador de CVs — Perfiles para un Cargo (con capa visual)
-----------------------------------------------------------
• Sube hasta 10 hojas de vida en PDF
• Escribe el cargo y las skills requeridas
• Obtén una tabla con: nombre, resumen corto, skills detectadas, veredicto (5 niveles), veredicto detallado y calificación 0–100
• Diagnóstico opcional de calidad del CV: estructura, redacción, calidad de experiencia y observaciones
• Descarga en Excel (fallback a CSV si no hay openpyxl)
• Capa visual: colores por veredicto + barra de progreso en calificación

*La configuración sensible (clave) se toma de Secrets/entorno. La interfaz no muestra controles técnicos al usuario final.*
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

# SDK OpenAI
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("Falta el paquete 'openai'. Instala: pip install openai")

# ==========================
# Configuración base (no visible para usuarios)
# ==========================
load_dotenv()
st.set_page_config(page_title="Evaluador de CVs — Perfiles para un Cargo", page_icon="📄", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL_DEFAULT = "gpt-5"  # puedes cambiarlo en secrets con SHOW_ADMIN=true
MAX_PDFS_FREE = 10
SHOW_ADMIN = str(st.secrets.get("SHOW_ADMIN", "false")).lower() == "true"

# ==========================
# Estado global
# ==========================
if "busy" not in st.session_state:
    st.session_state["busy"] = False
if "evaluaciones" not in st.session_state:
    st.session_state["evaluaciones"] = None

def set_busy(flag: bool):
    st.session_state["busy"] = flag

# ==========================
# Utilidades
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
        for cut in range(len(candidate), max(len(candidate) - 4000, 100), -50):
            try:
                return json.loads(candidate[:cut])
            except Exception:
                continue
    return {}

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Resultados") -> Optional[bytes]:
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

# ========= Estilos visuales para la tabla =========
def color_veredicto(val: str) -> str:
    palette = {
        "Excelente": "background-color:#2e7d32; color:white;",     # verde oscuro
        "Muy Bueno": "background-color:#43a047; color:white;",     # verde
        "Regular":   "background-color:#ffeb3b; color:#111;",      # amarillo
        "Débil":     "background-color:#fb8c00; color:white;",     # naranja
        "Muy Débil": "background-color:#e53935; color:white;",     # rojo
    }
    return palette.get(val, "")

def barra_calificacion(val: Any) -> str:
    try:
        v = int(val)
    except Exception:
        v = 0
    v = max(0, min(100, v))
    # Color por umbrales
    color = "#2e7d32" if v >= 80 else "#43a047" if v >= 61 else "#ffb300" if v >= 41 else "#fb8c00" if v >= 21 else "#e53935"
    return f"""
    <div style="width:110px; background:#e5e7eb; border-radius:6px; overflow:hidden;">
      <div style="width:{v}%; background:{color}; color:white; font-size:12px; text-align:center; padding:2px 0;">
        {v}
      </div>
    </div>
    """

def badges_skills(val: Any) -> str:
    if isinstance(val, list):
        skills = val
    elif isinstance(val, str):
        # por si llega como string de lista
        skills = [s.strip() for s in re.split(r"[,\|·]", val) if s.strip()]
    else:
        skills = []
    chips = "".join(
        f'<span style="display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:12px;">{s}</span>'
        for s in skills[:12]
    )
    return f'<div style="line-height:1.9">{chips}</div>'

def style_table(df: pd.DataFrame, columns_order: List[str]) -> str:
    sty = (
        df[columns_order]
        .style
        .applymap(color_veredicto, subset=["veredicto"])
        .format({"calificacion": barra_calificacion, "skills_detectadas": badges_skills}, escape="html")
    )
    # to_html con escape=False para inyectar barras/badges
    return sty.to_html(escape=False)

# ==========================
# Panel lateral (solo info)
# ==========================
st.sidebar.title("Configuración")
if OPENAI_API_KEY:
    st.sidebar.success("Clave configurada de forma segura.")
else:
    st.sidebar.error("Falta la clave segura para procesar CVs. (Configurar en Secrets).")

# Controles admin (opcionales)
if SHOW_ADMIN:
    CHAT_MODEL = st.sidebar.selectbox(
        "Modelo interno (solo administrador)",
        options=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        index=0,
        disabled=st.session_state["busy"]
    )
else:
    CHAT_MODEL = CHAT_MODEL_DEFAULT

# ==========================
# Interfaz principal
# ==========================
st.title("📄 Evaluador de CVs — Perfiles para un Cargo")
st.caption("Sube hojas de vida en PDF, define el cargo y recibe un ranking con análisis por perfil.")

col_left, col_right = st.columns([1, 1])
with col_left:
    cargo = st.text_area(
        "Cargo / Descripción del puesto",
        placeholder="Ejemplo: Cargo: Científico de Datos (Colombia).Buscamos un Científico de Datos con al menos 4 años de experiencia en el desarrollo..",
        height=130,
        disabled=st.session_state["busy"]
    )
with col_right:
    skills_raw = st.text_input(
        "Skills requeridas (separadas por comas)",
        value="Python, SQL, Machine Learning",
        disabled=st.session_state["busy"]
    )
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

files = st.file_uploader(
    "📄 Sube CVs en formato PDF (máximo 10 archivos)",
    type=["pdf"],
    accept_multiple_files=True,
    disabled=st.session_state["busy"]
)
if files and len(files) > MAX_PDFS_FREE:
    st.warning(f"Solo se permiten {MAX_PDFS_FREE} PDFs por sesión. Tomaré los primeros {MAX_PDFS_FREE}.")
    files = files[:MAX_PDFS_FREE]

run = st.button("Evaluar y generar tabla", disabled=st.session_state["busy"])

# ==========================
# Prompt + Evaluación
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
  "calificacion": 0-100,
  "estructura": "Bien organizada | Poco clara | Deficiente",
  "redaccion": "Clara y profesional | Regular | Confusa",
  "calidad_experiencia": "Alta | Media | Baja",
  "observaciones": "frases cortas con puntos de mejora"
}}
Criterios (rango → etiqueta):
- Excelente (80–100) • Muy Bueno (61–80) • Regular (41–60) • Débil (21–40) • Muy Débil (0–20)
"""

def llm_evaluate_one(client: OpenAI, name: str, text: str, cargo: str, skills: List[str]) -> Dict[str, Any]:
    base = text[:2500]
    skill_windows = windows_around_skills(text, skills, radius=220, max_windows=8)
    context = (base + "\n\n" + skill_windows).strip()[:7000]

    prompt = build_prompt(name, cargo, skills, context)
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Responde únicamente con JSON válido. Sé conciso y objetivo."},
                {"role": "user", "content": prompt},
            ]
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        content = f'{{"nombre":"{name}","resumen_corto":"No fue posible evaluar el CV","skills_detectadas":[],"veredicto":"Muy Débil","veredicto_detallado":"Error de evaluación","calificacion":0,"estructura":"","redaccion":"","calidad_experiencia":"","observaciones":"","_error":"{e}"}}'

    data = safe_json_from_text(content)
    # Defaults
    data.setdefault("nombre", name)
    data.setdefault("resumen_corto", "")
    data.setdefault("skills_detectadas", [])
    data.setdefault("veredicto", "Muy Débil")
    data.setdefault("veredicto_detallado", "")
    data.setdefault("estructura", "")
    data.setdefault("redaccion", "")
    data.setdefault("calidad_experiencia", "")
    data.setdefault("observaciones", "")
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
    if not OPENAI_API_KEY:
        st.error("No hay una clave segura configurada para procesar CVs.")
        st.stop()
    if not cargo.strip():
        st.error("Describe el cargo a evaluar.")
        st.stop()
    if not files:
        st.error("Sube al menos 1 PDF.")
        st.stop()

    client = OpenAI(api_key=OPENAI_API_KEY)
    set_busy(True)

    with st.status("Procesando CVs…", expanded=True) as status:
        status.write("Extrayendo texto…")
        names, texts = [], []
        for f in files:
            try:
                t = clean_text(read_pdf_text(f))
            except Exception as e:
                t = ""
                st.warning(f"No pude leer {f.name}: {e}")
            names.append(f.name)
            texts.append(t)

        status.write("Analizando perfiles…")
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
        status.update(label="Evaluación completa", state="complete")

    df = pd.DataFrame(resultados, columns=[
        "nombre", "resumen_corto", "skills_detectadas",
        "veredicto", "veredicto_detallado", "calificacion",
        "estructura", "redaccion", "calidad_experiencia", "observaciones"
    ]).sort_values("calificacion", ascending=False).reset_index(drop=True)
    st.session_state["evaluaciones"] = df
    set_busy(False)

# ==========================
# Resultados + Descarga (con capa visual)
# ==========================
if st.session_state["evaluaciones"] is not None:
    df = st.session_state["evaluaciones"]
    st.subheader("Resultados de la evaluación")

    mostrar_diag = st.toggle("Mostrar diagnóstico de calidad del CV", value=False)

    if mostrar_diag:
        cols = ["nombre", "resumen_corto", "skills_detectadas",
                "veredicto", "veredicto_detallado", "calificacion",
                "estructura", "redaccion", "calidad_experiencia", "observaciones"]
    else:
        cols = ["nombre", "resumen_corto", "skills_detectadas",
                "veredicto", "veredicto_detallado", "calificacion"]

    # Render con estilos (colores + barra + chips)
    html_table = style_table(df, cols)
    st.write(html_table, unsafe_allow_html=True)

    # Descarga (Excel preferido)
    xlsx = df_to_excel_bytes(df, sheet_name="Resultados")
    if xlsx is not None:
        st.markdown(b64_download_link(xlsx, "ranking_perfiles.xlsx", "⬇️ Descargar Excel"), unsafe_allow_html=True)
    else:
        st.info("Descargando CSV.")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.markdown(b64_download_link(csv_bytes, "ranking_perfiles.csv", "⬇️ Descargar CSV"), unsafe_allow_html=True)

# ==========================
# Pie de página (contacto)
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

