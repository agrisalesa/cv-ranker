#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
App Rankeador de CVs — Streamlit (OpenAI)
-----------------------------------------
Usuarios suben hasta 10 PDFs (CVs), escriben el cargo y skills requeridos,
y se genera un ranking (similitud + cobertura de skills) + explicación con la API de ChatGPT.

Cómo ejecutar localmente:
  1) pip install -U streamlit pdfplumber openai numpy pandas scikit-learn python-dotenv
  2) export OPENAI_API_KEY="sk-..."   # o ingrésala en el sidebar
  3) streamlit run app.py

Despliegue barato:
  - Streamlit Community Cloud (gratis) o Railway/Render (free tier).

Privacidad (MVP):
  - No guarda los CVs en disco; procesa en memoria.
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

# OpenAI SDK (paquete oficial)
try:
    from openai import OpenAI
except Exception:
    raise RuntimeError("Falta el paquete 'openai'. Instala: pip install openai")

# ==========================
# Configuración / llaves
# ==========================
load_dotenv()
st.set_page_config(page_title="Rankeador de CVs — MVP", page_icon="📄", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_EMB_DEFAULT = "text-embedding-3-small"       # embeddings MUY baratos
MODEL_CHAT_DEFAULT = "gpt-5"                       # calidad primero (puedes bajar a 5-mini o 5-nano)

# Límite gratis (por sesión del navegador)
MAX_PDFS_FREE = 10

# ==========================
# Precios por 1M tokens (USD) — sep/2025
# ==========================
EMB_PRICE_PER_MTOK = 0.02  # text-embedding-3-small
CHAT_PRICE = {
    "gpt-5":      {"in": 1.25, "out": 10.00},
    "gpt-5-mini": {"in": 0.25, "out": 2.00},
    "gpt-5-nano": {"in": 0.05, "out": 0.40},
}

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

def approx_tokens(text: str) -> int:
    # Aproximación simple: ~4 caracteres por token
    return max(1, int(len(text) / 4))

# ==========================
# UI — Sidebar
# ==========================
st.sidebar.title("⚙️ Configuración")

# Usar clave de secrets o variable de entorno si existe
api_key_input = OPENAI_API_KEY

if not api_key_input:
    # Solo mostrar input si no hay key en secrets/env
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key",
        value="",
        type="password",
        help="No se guarda en el servidor."
    )
else:
    st.sidebar.success("🔑 API Key cargada desde configuración segura.")


MODEL_CHAT = st.sidebar.selectbox(
    "Modelo de Chat",
    options=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
    index=0,
    help="gpt-5 = máxima calidad; gpt-5-mini = buen balance; gpt-5-nano = ultra barato."
)
MODEL_EMB = MODEL_EMB_DEFAULT
st.sidebar.caption(f"Modelos: Chat={MODEL_CHAT} | Embeddings={MODEL_EMB}")
st.sidebar.markdown("---")
free_info = st.sidebar.empty()

st.title("📄 Rankeador de CVs — MVP")
st.caption("Sube hasta 10 PDFs, define cargo y skills, y genera un ranking + explicación.")

# ==========================
# Inputs principales
# ==========================
col_left, col_right = st.columns([1, 1])
with col_left:
    cargo = st.text_area(
        "Cargo / Descripción del puesto",
        placeholder="Ej: Data Scientist con Python, SQL, Machine Learning, MLOps...",
        height=120
    )
with col_right:
    skills_raw = st.text_input("Skills requeridos (coma separada)", value="Python, SQL, Machine Learning")
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

files = st.file_uploader("Sube CVs en PDF (hasta 10)", type=["pdf"], accept_multiple_files=True)
if files and len(files) > MAX_PDFS_FREE:
    st.warning(f"Solo se permiten {MAX_PDFS_FREE} PDFs gratuitos por sesión. Tomaré los primeros {MAX_PDFS_FREE}.")
    files = files[:MAX_PDFS_FREE]
free_info.info(f"Usando {len(files) if files else 0}/{MAX_PDFS_FREE} CVs en esta sesión gratuita.")

run = st.button("🔎 Evaluar y rankear")

# ==========================
# Lógica principal
# ==========================
if run:
    if not api_key_input:
        st.error("Falta OpenAI API Key. Ingresa tu clave en el panel izquierdo.")
        st.stop()
    if not cargo.strip():
        st.error("Describe el cargo a evaluar.")
        st.stop()
    if not files:
        st.error("Sube al menos 1 PDF.")
        st.stop()

    client = OpenAI(api_key=api_key_input)

    # 1) Ingesta
    with st.spinner("Extrayendo texto de PDFs..."):
        docs_text: List[str] = []
        names: List[str] = []
        for f in files:
            try:
                text = read_pdf_text(f)
                text = clean_text(text)
            except Exception as e:
                text = ""
                st.warning(f"No pude leer {f.name}: {e}")
            docs_text.append(text)
            names.append(f.name)

    # 2) Consulta base (cargo + skills)
    query_text = cargo.strip()
    if skills:
        query_text += "\n\nSkills requeridos: " + ", ".join(skills)

    # Estimación de tokens de embeddings (query + CVs)
    emb_tokens = approx_tokens(query_text) + sum(approx_tokens(t) for t in docs_text)

    # 3) Embeddings y similitud
    with st.spinner("Calculando embeddings y similitud..."):
        try:
            q_emb = to_embeddings(client, [query_text], MODEL_EMB)[0].reshape(1, -1)
            d_embs = to_embeddings(client, docs_text, MODEL_EMB)
        except Exception as e:
            st.error(f"Error generando embeddings: {e}")
            st.stop()

        sims = cosine_similarity(q_emb, d_embs)[0]  # (n_docs,)
        coverages = [calc_skill_coverage(t, skills) for t in docs_text]
        final_scores = [score_candidate(float(s), float(c)) for s, c in zip(sims, coverages)]

    # 4) Tabla de resultados
    df = pd.DataFrame({
        "archivo": names,
        "sim_cargo": sims,
        "%skills": [round(c * 100, 1) for c in coverages],
        "score": final_scores,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    st.subheader("🏆 Ranking de candidatos")
    st.dataframe(df, use_container_width=True)

    # Exportar CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.markdown(b64_download_link(csv_bytes, "ranking_candidatos.csv", "⬇️ Descargar CSV"), unsafe_allow_html=True)

    # 5) Explicaciones con LLM
    st.subheader("🧠 Explicaciones (LLM)")
    top_k = st.slider("¿Cuántos candidatos explicar?", 1, min(5, len(df)), min(3, len(df)))

    chat_prompt_tokens = 0
    chat_completion_tokens = 0

    for i in range(top_k):
        row = df.iloc[i]
        idx = int(df.index[i])
        name = row["archivo"]
        text = docs_text[idx]

        # Recorte defensivo: base + snippets por skill
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
                model=MODEL_CHAT,
                messages=[
                    {"role": "system", "content": "Eres un reclutador técnico conciso y objetivo. Respondes en español."},
                    {"role": "user", "content": prompt},
                ]
            )
            explanation = resp.choices[0].message.content

            # Tokens reales si el SDK los provee; si no, aproximamos
            if hasattr(resp, "usage") and resp.usage is not None:
                chat_prompt_tokens += getattr(resp.usage, "prompt_tokens", 0) or 0
                chat_completion_tokens += getattr(resp.usage, "completion_tokens", 0) or 0
            else:
                chat_prompt_tokens += approx_tokens(prompt)
                chat_completion_tokens += approx_tokens(explanation or "")
        except Exception as e:
            explanation = f"No pude generar explicación: {e}"

        with st.expander(f"Explicación — {name}"):
            st.markdown(explanation)

    # 6) Estimación de costo
    emb_cost = (emb_tokens / 1_000_000) * EMB_PRICE_PER_MTOK
    price = CHAT_PRICE.get(MODEL_CHAT, CHAT_PRICE["gpt-5"])
    chat_cost = (chat_prompt_tokens / 1_000_000) * price["in"] + (chat_completion_tokens / 1_000_000) * price["out"]

    st.markdown("---")
    st.subheader("💵 Costo estimado (USD)")
    st.write({
        "emb_tokens_aprox": emb_tokens,
        "emb_cost": round(emb_cost, 6),
        "chat_prompt_tokens": chat_prompt_tokens,
        "chat_completion_tokens": chat_completion_tokens,
        "chat_cost": round(chat_cost, 6),
        "total_estimate": round(emb_cost + chat_cost, 6),
    })

    st.success(f"Listo ✅ — Modelo de chat usado: {MODEL_CHAT}")

# ==========================
# Footer
# ==========================
st.caption("Este MVP no almacena archivos en disco. Para PRD: agregar auth, storage y observabilidad.")

