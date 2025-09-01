# 📄 Rankeador de CVs — Streamlit (OpenAI)

MVP para subir hasta **10 PDFs** (CVs), definir **cargo** + **skills**, y obtener:
- **Ranking** por similitud semántica + cobertura de skills
- **Explicaciones** generadas con ChatGPT (modelos GPT-5 / GPT-5-mini / GPT-5-nano)

## 🚀 Demo local

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."     # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
streamlit run app.py
```

## ☁️ Deploy en Streamlit Community Cloud

1. Sube este repo a **GitHub**.
2. Ve a https://share.streamlit.io → *New app* → selecciona el repo y `app.py`.
3. En **Secrets**, agrega:
   ```toml
   OPENAI_API_KEY="sk-..."
   ```
4. Deploy y comparte tu URL pública 🎉
