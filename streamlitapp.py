import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from fpdf import FPDF

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

def extract_skills(text):
    tech_keywords = ["python", "java", "sql", "machine learning", "deep learning", "pytorch",
                     "tensorflow", "nlp", "data science", "opencv", "cnn", "lstm", "aws", "docker"]
    found = [skill for skill in tech_keywords if skill.lower() in text.lower()]
    return found

st.title("Resume AI — Semantic Ranking + Skill Matching + Report Export")

st.subheader("Step 1: Paste Job Description")
job_description = st.text_area("or upload .txt file")
job_file = st.file_uploader("Upload Job Description file (optional)", type=["txt"])
if job_file:
    job_description = job_file.read().decode()

st.subheader("Step 2: Upload Resume (multiple allowed)")
uploaded_resumes = st.file_uploader("Upload Resumes (PDF / DOCX)", type=["pdf"], accept_multiple_files=True)

resumes = []
for resume in uploaded_resumes:
    text = extract_text_from_pdf(resume)
    skills = extract_skills(text)
    if text.strip():
        resumes.append({"name": resume.name, "text": text, "skills": skills})

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

    data = []
    for score, r in ranked:
        skill_match = round((len(r["skills"]) / 10) * 100, 2)  # simple match score
        data.append([r["name"], round(score * 100, 2), ", ".join(r["skills"]), f"{skill_match}%"])
        st.write(f"✅ **{r['name']}** — Match Score: `{round(score * 100, 2)}%` — Skills: `{', '.join(r['skills'])}`")

    df = pd.DataFrame(data, columns=["Resume Name", "Match Score %", "Skills Found", "Skill Match %"])
    st.dataframe(df)

    st.subheader("📄 Export Results")

    excel = df.to_excel("resume_ranking.xlsx", index=False)
    st.download_button("Download Excel Report", data=open("resume_ranking.xlsx", "rb"), file_name="resume_report.xlsx")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Resume Ranking Report", ln=True)
    pdf.ln(5)

    for row in data:
        pdf.cell(200, 8, txt=f"{row[0]} — Match: {row[1]}% — Skills: {row[2]}", ln=True)

    pdf.output("resume_report.pdf")
    st.download_button("Download PDF Report", data=open("resume_report.pdf", "rb"), file_name="resume_report.pdf")
else:
    st.info("Upload resumes and add job description to start ranking.")
