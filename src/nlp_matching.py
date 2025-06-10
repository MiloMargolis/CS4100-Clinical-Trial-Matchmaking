from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

"""
The purpose of this file is the implement the NLP-based similarity functions 
for matching patients to clinical trial criteria. It uses cosine similarity on the TF-IDF vectors
"""

# computes the cosine similary between patient keywords and eligibility
# handles missing values by converting to empty strings
def compute_similarity(patient_keywords, eligibility_text):
    patient_keywords = str(patient_keywords) if patient_keywords else ''
    eligibility_text = str(eligibility_text) if eligibility_text else ''

    corpus = [patient_keywords, eligibility_text]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score

# computes a pairwise similarity score for each patient-trial combination
def compute_similarity_bulk(patient_df, trials_df):
    results = []

    for _, patient in patient_df.iterrows():
        patient_id = patient.get('PatientID', 'Unknown')
        patient_keywords = str(patient.get('Keywords', ''))

        for _, trial in trials_df.iterrows():
            trial_id = trial.get('NCTId', 'Unknown')
            eligibility_text = str(trial.get('Eligibility', ''))

            score = compute_similarity(patient_keywords, eligibility_text)
            results.append({
                'PatientID': patient_id,
                'TrialID': trial_id,
                'Score': score
            })

    return pd.DataFrame(results)





    

