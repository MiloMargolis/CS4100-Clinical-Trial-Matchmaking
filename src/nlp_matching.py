"""
The purpose of this file is the implement the NLP-based similarity functions 
for matching patients to clinical trial criteria.
"""

# SKELETON CODE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(patient_keywords, eligibility_text):
    """
    Computes cosine similarity between patient keywords and eligibility text.
    """
    corpus = [patient_keywords, eligibility_text]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score

