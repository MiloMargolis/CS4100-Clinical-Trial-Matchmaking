"""
main.py

This script carries out the process of matching our mock patients to real clinical trials. 
We use the clinical trials dataset and the patient dataset. Ultimately, it will leverages NLP
to score how well each patient fits the matchmaking criteria of each trial.
"""

import pandas as pd
from src.patient_data_ingestion import load_patient_data
from src.nlp_matching import compute_similarity

def load_trial_data(filepath):
    """
    Loads clinical trials from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def match_patients_to_trials(patients_df, trials_df, top_n=5):
    """
    For each patient, rank clinical trials by similarity score.
    """
    for _, patient in patients_df.iterrows():
        patient_keywords = patient['Condition']
        # Optional: add extra keywords here

        results = []
        for _, trial in trials_df.iterrows():
            eligibility_text = trial.get('Eligibility', '')
            score = compute_similarity(patient_keywords, eligibility_text)
            results.append((trial['NCTId'], trial['Title'], score))

        # Sort by similarity
        results.sort(key=lambda x: x[2], reverse=True)
        print(f"\nTop {top_n} matches for Patient ID {patient['PatientID']}:")
        for nctid, title, score in results[:top_n]:
            print(f"  NCTId: {nctid}, Title: {title}, Score: {score:.2f}")

if __name__ == "__main__":
    patients_df = load_patient_data("data/patient_data.csv")
    trials_df = load_trial_data("data/clinical_trials.csv")
    match_patients_to_trials(patients_df, trials_df, top_n=5)
