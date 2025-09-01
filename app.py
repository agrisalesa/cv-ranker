#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
App Rankeador de CVs — Streamlit (OpenAI) — TOP PERFORMANCE
-----------------------------------------------------------
- Usuarios suben hasta 10 PDFs (CVs)
- Definen cargo + skills requeridos
- Ranking usando embeddings (similaridad por chunks, Top-K)
- Explicaciones con modelo de chat (GPT-5 family)
- UX con estado, status panel y barra de progreso

Cómo ejecutar localmente:
  1) pip install -U streamlit pdfplumber openai numpy pandas scikit-learn python-dotenv
  2) export OPENAI_API_KEY="sk-..."   # o usar Secrets en Streamlit Cloud
  3) streamlit run app.py
"""
from __future__ import annotations

import io
import os
import re
import base64
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("Falta el paquete 'openai'. Instala: pip install openai")

# ==========================
# Configuración / llaves
# ==========================
load_dotenv()
st.set_page_config(page_title="Rankeador de CVs — TOP", page_icon="📄", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Modelos por defecto (performance)
EMBEDDING_MODEL_DEFAULT = "text-embedding-3-large"
CHAT_MODEL_DEFAULT = "gpt-5"
MAX_PDFS_FREE = 10

# ==========================
# Estado global (UX)
# ==========================
if "busy" not in st.session_state:
    st.session_state["busy"] = False
if "last_df" not in st.session_state:
    st.session_state["last_df"] = None
if "docs_text" not in st.session_state:
    st.session_state["docs_text"] = None
if "names" not in st.session_state:
    st.session_state["names"] = None

def set_busy(flag: bool):
    st.session_state["busy"] = flag

# ==========================
# Helpers
# ==========================
def read_pdf_text(file: io.BytesIO) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n\n".join(text_parts)

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or " ")
    return s.strip()

def chunk_text(s: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    chunks, start = [], 0
    while start < len(s):
        end = min(len(s), start + max_chars)
        chunks.append(s[start:end])
        if end == len(s):
            break
        start = max(0, end - overlap)
    return chunks

def to_embeddings(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
    return np.vstack(vecs)

def calc_skill_coverage(text: str, skills: List[str]) -> float:
    if not skills:
        return 0.0
    t = text.lower()
    hits = sum(1 for s in skills if s.lower() in t)
    return hits / max(1, len(skills))

def score_candidate(cos_sim: float, coverage: float, w_sim: float = 0.7, w_cov: float = 0.3) -> float:
    return float(w_sim * cos_sim + w_cov * coverage)

def b64_download_link(data: bytes, filename: str, label: str) -> str:
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">{label}</a>'
    return href

def best_chunk_similarity(
    client: OpenAI,
    doc_text: str,
    query_emb: np.ndarray,
    model: str,
    chunk_chars: int = 1200,
    overlap: int = 200,
    topk_avg: int = 2,
) -> float:
    """
    Similaridad coseno entre la consulta y todos los chunks del documento.
    Devuelve el promedio del Top-K (robusto a ruido).
    """
    chunks = chunk_text(doc_text, max_chars=chunk_chars, overlap=overlap) or [doc_text[:chunk_chars]]
    emb_chunks = to_embeddings(client, chunks, model)          # (n_chunks, d)
    sims = cosine_similarity(query_emb, emb_chunks)[0]         # (n_chunks,)
    sims_sorted = np.sort(sims)[::-1]
    k = min(max(1, topk_avg), len(sims_sorted))
    return float(np.mean(sims_sorted[:k]))

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

# Selector de modelos (chat y embeddings)
CHAT_MODEL = st.sidebar.selectbox(
    "Modelo de Chat",
    options=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
    index=0,
    help="gpt-5 = máxima calidad; gpt-5-mini = muy buen balance; gpt-5-nano = ultra barato.",
    disabled=st.session_state["busy"]
)
EMBEDDING_MODEL = st.sidebar.selectbox(
    "Modelo de Embeddings",
    options=["text-embedding-3-large", "text-embedding-3-small"],
    index=0,  # top performance por defecto
    help="3-large = mejor recall/precisión; 3-small = más barato.",
    disabled=st.session_state["busy"]
)
st.sidebar.caption(f"Modelos activos → Chat={CHAT_MODEL} | Embeddings={EMBEDDING_MODEL}")
st.sidebar.markdown("---")

# ==========================
# UI — Main
# ==========================
st.title("📄 Rankeador de CVs — TOP performance")
st.caption("Sube hasta 10 PDFs, define cargo y skills, y genera un ranking + explicación.")

col_left, col_right = st.columns([1, 1])
with col_left:
    cargo = st.text_area(
        "Cargo / Descripción del puesto",
        placeholder="Ej: Científico/a de Datos Senior, Python, SQL, ML, arquitectura de datos, MLOps…",
        height=120,
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

# Paso A: Evaluar y rankear
run_eval = st.button("🔎 Evaluar y rankear", disabled=st.session_state["busy"])

if run_eval:
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

        status.write("🧮 Calculando embeddings de la consulta…")
        query_text = cargo.strip()
        if skills:
            query_text += "\n\nSkills requeridos: " + ", ".join(skills)
        try:
            q_emb = to_embeddings(client, [query_text], EMBEDDING_MODEL)[0].reshape(1, -1)
        except Exception as e:
            set_busy(False)
            st.error(f"Error generando embeddings de la consulta: {e}")
            st.stop()

        status.write("🔎 Calculando similaridad por documento (chunks + Top-K)…")
        sims = []
        coverages = []
        for t in docs_text:
            try:
                s = best_chunk_similarity(client, t, q_emb, EMBEDDING_MODEL,
                                          chunk_chars=1200, overlap=200, topk_avg=2)
            except Exception as e:
                s = 0.0
                st.warning(f"Error con embeddings del CV: {e}")
            sims.append(s)
            coverages.append(calc_skill_coverage(t, skills))

        final_scores = [score_candidate(float(s), float(c)) for s, c in zip(sims, coverages)]

        df = pd.DataFrame({
            "archivo": names,
            "similaridad_consulta_cos": sims,
            "%skills": [round(c * 100, 1) for c in coverages],
            "score": final_scores,
        }).sort_values("score", ascending=False).reset_index(drop=True)

        st.session_state["last_df"] = df
        st.session_state["docs_text"] = docs_text
        st.session_state["names"] = names

        status.update(label="✅ Evaluación completa", state="complete")
    set_busy(False)

# Mostrar ranking si ya existe
if st.session_state["last_df"] is not None:
    df = st.session_state["last_df"]
    docs_text = st.session_state["docs_text"]
    names = st.session_state["names"]

    st.subheader("🏆 Ranking de candidatos")
    st.caption(
        "• **similaridad_consulta_cos**: similitud coseno (0–1) entre la consulta (cargo+skills) y el CV, "
        "usando el mejor fragmento (chunks) con promedio Top-K.\n"
        "• **%skills**: porcentaje de skills requeridas encontradas literal en el CV.\n"
        "• **score**: 0.7×similaridad + 0.3×cobertura de skills."
    )
    st.dataframe(df, use_container_width=True)

    # Descargar CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.markdown(b64_download_link(csv_bytes, "ranking_candidatos.csv", "⬇️ Descargar CSV"), unsafe_allow_html=True)

    # Paso B: Explicaciones
    st.subheader("🧠 Explicaciones (LLM)")
    top_k = st.slider(
        "¿Cuántos candidatos explicar?",
        1, min(5, len(df)), min(3, len(df)),
        disabled=st.session_state["busy"]
    )
    gen = st.button("🧠 Generar explicaciones", disabled=st.session_state["busy"])

    if gen:
        if not OPENAI_API_KEY and not api_key_input:
            st.error("Falta OpenAI API Key.")
            st.stop()

        client = OpenAI(api_key=api_key_input)
        set_busy(True)
        progress = st.progress(0, text="Generando explicaciones…")

        for i in range(top_k):
            row = df.iloc[i]
            idx = int(df.index[i])
            name = row["archivo"]
            text = docs_text[idx]

            # Recorte defensivo + snippets por skill
            base_snip = text[:2500]
            skill_snips = []
            for s in skills:
                m = re.search(rf"(.{{0,200}}\b{s}\b.{{0,200}})", text, flags=re.IGNORECASE)
                if m:
                    skill_snips.append(m.group(1))
            context = (base_snip + "\n\n" + "\n\n".join(skill_snips)).strip()[:6000]

            prompt = f"""
Eres reclutador técnico. Dado el perfil del cargo y el texto de un CV, evalúa si el candidato encaja.

CARGO:
{cargo}

SKILLS REQUERIDAS: {', '.join(skills) if skills else '(no especificadas)'}

EXTRACTO DE CV:
{context}

Responde en español, con:
1) Resumen de experiencia relevante (3–5 viñetas)
2) Cobertura de skills: lista ✓/✗ para cada skill requerida
3) Riesgos o gaps
4) Veredicto (Fuerte / Medio / Débil)
"""
            try:
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "Eres un reclutador técnico conciso y objetivo. Respondes en español."},
                        {"role": "user", "content": prompt},
                    ]
                    # Sin temperature: la familia gpt-5 no soporta otro valor que 1
                )
                explanation = resp.choices[0].message.content
            except Exception as e:
                explanation = f"No pude generar explicación: {e}"

            with st.expander(f"Explicación — {name}"):
                st.markdown(explanation)

            progress.progress(int((i + 1) / top_k * 100), text=f"Generando explicaciones… {i+1}/{top_k}")

        progress.empty()
        set_busy(False)

# ==========================
# Footer
# ==========================
st.caption("Contacto: Andrés Grisales Ardila. Correo: agrisalesa@unal.edu.co")

