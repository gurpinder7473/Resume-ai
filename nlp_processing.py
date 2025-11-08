import spacy
import re

# Auto download model if not installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_skills(text):
    """Extracts skills from resume text."""
    common_skills = {"python", "java", "c++", "machine learning", "data science", "react", "sql"}
    words = set(text.lower().split())
    return common_skills.intersection(words)

def extract_experience(text):
    """Extracts years of experience."""
    match = re.search(r"(\d+)\s*(?:years|yrs) of experience", text, re.IGNORECASE)
    return f"{match.group(1)} years" if match else "Not mentioned"

def extract_education(text):
    """Extracts education details using regex."""
    education_pattern = r"(B\.?Sc|B\.?Tech|BCA|M\.?Sc|M\.?Tech|MBA|MCA|PhD|Doctorate|Diploma|Associate Degree|Bachelor of Science|Bachelor of Technology|Master of Science|Master of Technology|Master of Business Administration)"
    matches = re.findall(education_pattern, text, re.IGNORECASE)
    return ", ".join(set(matches)) if matches else "Not mentioned"
