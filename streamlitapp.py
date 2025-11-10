# streamlitapp.py
import io
import streamlit as st
import pandas as pd
import PyPDF2
import docx
import numpy as np

from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Resume AI (Semantic Ranking)", layout="wide")
st.title("📄 Resume AI — Semantic Resume Ranking (Transformer Based)")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text.append(extracted)
        return "\n".join(text)
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def extract_docx(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def extract_text(file) -> str:
    raw = file.read()
    name = file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(raw)
    elif name.endswith(".docx"):
        return extract_docx(raw)
    else:
        try:
            return raw.decode("utf-8")
        except:
            return ""

st.subheader("Step 1: Paste Job Description")
job_description = st.text_area("or upload .txt file", height=160)

job_file = st.file_uploader("Upload Job Description file (optional)", type=["txt"])
if job_file and not job_description.strip():
    job_description = job_file.read().decode("utf-8")

if not job_description.strip():
    st.stop()

st.subheader("Step 2: Upload Resume (multiple allowed)")
resumes = st.file_uploader("Upload Resumes (PDF / DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if not resumes:
    st.stop()

top_k = st.sidebar.slider("Show top K candidates", min_value=1, max_value=len(resumes), value=min(5, len(resumes)))

model = load_model()
st.info("Extracting text & generating semantic embeddings...")

resume_texts = []
resume_names = []
for r in resumes:
    txt = extract_text(r)
    resume_texts.append(txt)
    resume_names.append(r.name)

job_embedding = model.encode(job_description, convert_to_tensor=True)
resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)

similarity_scores = util.cos_sim(job_embedding, resume_embeddings)[0].cpu().numpy()

result_df = pd.DataFrame({
    "Resume": resume_names,
    "Score (%)": np.round(similarity_scores * 100, 2),
    "Extracted Text": resume_texts
}).sort_values("Score (%)", ascending=False).reset_index(drop=True)

st.subheader("Top Matching Resumes")
st.table(result_df.head(top_k)[["Resume", "Score (%)"]])

for index, row in result_df.head(top_k).iterrows():
    with st.expander(f"{index+1}. {row['Resume']} — Score: {row['Score (%)']}%"):
        st.write(row["Extracted Text"])

csv_data = result_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Results as CSV", csv_data, "resume_ranking_results.csv", "text/csv")
