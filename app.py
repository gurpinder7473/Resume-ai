import streamlit as st
import os
from resume_parser import extract_text_from_pdf
from nlp_processing import extract_skills, extract_experience, extract_education
from ranking import rank_resumes

st.title("AI-powered Resume Screening & Ranking System")

# Upload resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])
job_description = st.text_area("Enter Job Description")

if st.button("Process Resumes"):
    if not uploaded_files or not job_description:
        st.warning("Please upload resumes and enter a job description.")
    else:
        resume_texts = []
        extracted_data = []
        
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resume_texts.append(text)
            skills = extract_skills(text)
            experience = extract_experience(text)
            education = extract_education(text)
            extracted_data.append({"name": file.name, "skills": skills, "experience": experience, "education": education})
        
        ranked_resumes = rank_resumes(resume_texts, job_description)
        
        st.subheader("Ranked Resumes")
        for idx, score in ranked_resumes:
            st.write(f"**{extracted_data[idx]['name']}** - Score: {score:.2f}")
            st.write(f"- **Skills:** {', '.join(extracted_data[idx]['skills'])}")
            st.write(f"- **Experience:** {extracted_data[idx]['experience']}")
            st.write(f"- **Education:** {extracted_data[idx]['education']}")
            st.write("---")
