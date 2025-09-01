#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
App Rankeador de CVs — Streamlit (OpenAI) — LLM-Only Scoring
------------------------------------------------------------
• Usuarios suben hasta 10 PDFs (CVs).
• Definen cargo + skills requeridas.
• El LLM evalúa cada CV y devuelve JSON con:
  nombre, resumen_corto, skills_detectadas, veredicto (Fuerte/Medio/Débil), calificacion (0–100).
• Se muestra tabla final y se permite descargar Excel.

Cómo ejecutar localmente:
  1) pip install -U streamlit pdfplumber openai numpy pandas python-dotenv openpyxl
  2) export OPENAI_API_KEY="sk-..."   # o usar Secrets en Streamlit Cloud
  3) streamlit run app.py
"""
from __future__ import annotations

import io
import os
import re
import json
import base64
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber
from dotenv import load_dotenv

# OpenAI SDK (paquete oficial)
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("Falta el paquete 'openai'. Instala: pip install openai")

# ==========================
# Configuración / llaves
# ==========================
load_dotenv()
st.set_page_config(page_title="Rankeador de CVs — LLM Scoring", page_icon="📄", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL_DEFAULT = "gpt-5"  # puedes cambiar a "gpt-5-mini" o "gpt-5-nano" en el sidebar
MAX_PDFS_FREE = 10

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
def read_pdf_text(file: io.BytesIO) -> str:
    """Extrae texto de un PDF completo."""
    text_parts: List[str] = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n\n".join(text_parts)

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or " ")
    return s.strip()

def safe_json_from_text(txt: str) -> Dict[str, Any]:
    """
    Intenta parsear JSON robustamente:
    - si viene dentro de ```json ... ```
    - si hay texto antes/después
    - fallback a heurística sencilla
    """
    if not txt:
        return {}
    # 1) Bloque ```json
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 2) Primer objeto { ... } en el texto
    m2 = re.search(r"(\{.*\})", txt, flags=re.DOTALL)
    if m2:
        candidate = m2.group(1)
        # recorte balanceando llaves (simple)
        # intenta hasta encontrar un cierre válido
        for cut in range(len(candidate), max(len(candidate)-2000, 0), -1):
            try:
                return json.loads(candidate[:cut])
            except Exception:
                continue
    # 3) fallback vacío
    return {}

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Resultados") -> bytes:
    """Convierte un DataFrame en bytes de Excel (.xlsx)."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.getvalue()

def b64_download_link(data: bytes, filename: str, label: str) -> str:
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    return href

# ==========================
# UI — Sidebar
# ==========================
st.sidebar.title("⚙️ Configuración")

# API key desde secrets/env; solo pedirla si no existe
api_key_input = OPENAI_API_KEY
if not api_key_input:
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key",
        value="",
        type="password",
        help="No se guarda en el servidor."
    )
else:
    st.sidebar.success("🔑 API Key cargada desde configuración segura.")

CHAT_MODEL = st.sidebar.selectbox(
    "Modelo de Chat",
    options=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
    index=0 if CHAT_MODEL_DEFAULT == "gpt-5" else 1,
    help="gpt-5 = máxima calidad; gpt-5-mini = muy buen balance; gpt-5-nano = ultra barato.",
    disabled=st.session_state["busy"]
)
st.sidebar.markdown("---")

# ==========================
# UI — Main
# ==========================
st.title("📄 Rankeador de CVs — LLM scoring (sin similitud)")
st.caption("El LLM evalúa y puntúa. Salida: nombre, resumen corto, skills, veredicto, calificación 0–100, con descarga en Excel.")

col_left, col_right = st.columns([1, 1])
with col_left:
    cargo = st.text_area(
        "Cargo / Descripción del puesto",
        placeholder="Ej: Científico/a de Datos Senior, Python, SQL, ML, arquitectura de datos, MLOps…",
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
    type=["pdf"],
    accept_multiple_files=True,
    disabled=st.session_state["busy"]
)
if files and len(files) > MAX_PDFS_FREE:
    st.warning(f"Solo se permiten {MAX_PDFS_FREE} PDFs por sesión. Tomaré los primeros {MAX_PDFS_FREE}.")
    files = files[:MAX_PDFS_FREE]

st.sidebar.info(f"Usando {len(files) if files else 0}/{MAX_PDFS_FREE} CVs en esta sesión.")

# Botón principal
run = st.button("🧠 Evaluar y generar tabla", disabled=st.session_state["busy"])

if run:
    if not api_key_input:
        st.error("Falta OpenAI API Key. Configúrala en Secrets o ingrésala en el campo correspondiente.")
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
        status.write("📥 Extrayendo texto de PDFs…")
        docs_text: List[str] = []
        names: List[str] = []
        for f in files:
            try:
                text = clean_text(read_pdf_text(f))
            except Exception as e:
                text = ""
                st.warning(f"No pude leer {f.name}: {e}")
            docs_text.append(text)
            names.append(f.name)

        status.write("🧠 Solicitando evaluación al LLM…")
        progress = st.progress(0, text="Evaluando candidatos…")

        resultados: List[Dict[str, Any]] = []

        for i, (name, text) in enumerate(zip(names, docs_text), start=1):
            # Recorte defensivo para no enviar textos enormes
            # (puedes ajustar si tu cuenta lo permite)
            context = text[:7000]

            prompt = f"""
Eres un reclutador técnico en Colombia. Evalúa el siguiente CV para el cargo descrito.

CARGO / DESCRIPCIÓN:
{cargo}

SKILLS REQUERIDAS (coma separada): {', '.join(skills) if skills else '(no especificadas)'}

TEXTO DEL CV:
{context}

Devuelve SOLO un JSON con esta forma EXACTA (sin texto adicional, sin comentarios):
{{
  "nombre": "{name}",
  "resumen_corto": "2-3 frases o 3-5 viñetas concisas sobre la experiencia más relevante",
  "skills_detectadas": ["skill1", "skill2", "..."],  // solo skills relevantes encontradas
  "veredicto": "Fuerte|Medio|Débil",                  // elige solo una palabra
  "calificacion": 0-100                                // número entero, 0 a 100
}}
Criterios:
- Si cubre la mayoría de skills clave y experiencia relevante: veredicto "Fuerte" (80–100).
- Si cubre parcialmente o tiene gaps claros: "Medio" (60–79).
- Si hay poca correspondencia: "Débil" (<60).
"""

            try:
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "Eres un reclutador técnico conciso, objetivo y estricto con el formato JSON."},
                        {"role": "user", "content": prompt},
                    ]
                    # Sin temperature: la familia gpt-5 no soporta valores distintos de 1
                )
                content = resp.choices[0].message.content or ""
            except Exception as e:
                content = f'{{"nombre":"{name}","resumen_corto":"Error al invocar el modelo","skills_detectadas":[],"veredicto":"Débil","calificacion":0,"_error":"{e}"}}'

            data = safe_json_from_text(content)
            # Normalización y defaults
            data.setdefault("nombre", name)
            data.setdefault("resumen_corto", "")
            data.setdefault("skills_detectadas", [])
            data.setdefault("veredicto", "Débil")
            try:
                score = int(data.get("calificacion", 0))
            except Exception:
                score = 0
            data["calificacion"] = max(0, min(100, score))
            resultados.append(data)

            progress.progress(int(i / len(names) * 100), text=f"Evaluando candidatos… {i}/{len(names)}")

        progress.empty()
        status.update(label="✅ Evaluación completa", state="complete")

    # Construir DataFrame final (ordenado por calificación)
    df = pd.DataFrame(resultados, columns=["nombre", "resumen_corto", "skills_detectadas", "veredicto", "calificacion"])
    df = df.sort_values("calificacion", ascending=False).reset_index(drop=True)
    st.session_state["evaluaciones"] = df
    set_busy(False)

# Mostrar resultados si existen
if st.session_state["evaluaciones"] is not None:
    df = st.session_state["evaluaciones"]
    st.subheader("🏁 Resultados (según LLM)")
    st.dataframe(df, use_container_width=True)

    # Descargar Excel
    xlsx_bytes = df_to_excel_bytes(df, sheet_name="Resultados")
    st.markdown(b64_download_link(xlsx_bytes, "ranking_llm.xlsx", "⬇️ Descargar Excel"), unsafe_allow_html=True)


# ==========================
# Footer
# ==========================
st.caption("Contacto: Andrés Grisales Ardila. Correo: agrisalesa@unal.edu.co")

