from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(resume_texts, job_description):
    """Ranks resumes based on similarity to job description."""
    texts = [job_description] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_resumes = sorted(zip(range(len(resume_texts)), similarity_scores), key=lambda x: x[1], reverse=True)
    
    return ranked_resumes  # List of (resume_index, score)
