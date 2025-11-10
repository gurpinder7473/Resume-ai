import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Resume AI", layout="wide")

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(file):
    try:
        text = ""
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        return text
    except:
        return ""

def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    return ""

st.title("Resume AI — Semantic Resume Ranking (Transformer Based)")

st.subheader("Step 1: Paste Job Description")
job_description = st.text_area("or upload .txt file")

job_file = st.file_uploader("Upload Job Description file (optional)", type=["txt"])
if job_file:
    job_description = job_file.read().decode()

st.subheader("Step 2: Upload Resume (multiple allowed)")
uploaded_resumes = st.file_uploader("Upload Resumes (PDF / DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

resumes = []
for resume in uploaded_resumes:
    text = extract_text(resume)
    if text.strip():
        resumes.append({"name": resume.name, "text": text})

# Avoid slider crash
if len(resumes) > 0:
    top_k = st.sidebar.slider("Show top K candidates", min_value=1, max_value=len(resumes), value=min(5, len(resumes)))
else:
    st.sidebar.warning("No readable resumes found — upload a valid PDF.")
    top_k = 0

if job_description and len(resumes) > 0:
    jd_embedding = model.encode(job_description)

    resume_embeddings = np.array([model.encode(r["text"]) for r in resumes])

    similarities = resume_embeddings @ jd_embedding / (
        np.linalg.norm(resume_embeddings, axis=1) * np.linalg.norm(jd_embedding)
    )

    ranked = sorted(zip(similarities, resumes), reverse=True)[:top_k]

    st.subheader("Ranking Result")
    for score, data in ranked:
        st.write(f"✅ **{data['name']}** — Score: `{round(score * 100, 2)}%`")
else:
    st.info("Upload resumes and add job description to start ranking.")
