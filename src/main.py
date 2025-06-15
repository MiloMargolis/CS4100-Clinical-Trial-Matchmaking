"""
Match mock patients to real clinical trials using keyword-based embedding vectors and KNN distance.
Saves top-N trial matches for each patient to CSV.
"""

import pandas as pd
from src.patient_data_ingestion import load_patient_data
from src.predictive_model import knn
from src.word_embedding import (
    train_word2vec,
    fit_tfidf_vect,
    weighted_embedding_bulk,
    weighted_sentence_embedding
)


def load_trial_data(filepath):
    return pd.read_csv(filepath)


def match_patients_to_trials(patients_df, trials_df, w2v_model, tfidf_vectorizer, top_n=5):
    trial_vectors = weighted_embedding_bulk(trials_df, "trial", tfidf_vectorizer, w2v_model)
    top_knn_matches = []

    for _, patient in patients_df.iterrows():
        patient_vector = weighted_sentence_embedding(
            str(patient.get('Keywords', '')),
            patient.get('PatientID', 'Unknown'),
            tfidf_vectorizer,
            w2v_model
        )

        top_results = knn(patient_vector, trial_vectors, top_n)

        matches = []
        for idx, trial in trials_df.iterrows():
            for result in top_results:
                trial_id, score_str = result.split(',', 1)
                if trial_id == trial['NCTId']:
                    matches.append((trial_id, trial['Title'], score_str.strip()))

        matches.sort(key=lambda x: float(x[2].split('=')[-1].replace('%', '').strip()), reverse=True)

        print(f"\nTop {top_n} matches for Patient {patient['PatientID']}: "
              f"{patient['Age']} y/o {patient['Sex']} seeking help with {patient['Condition']} and {patient['Keywords']}")
        for trial_id, title, score in matches:
            print(f"  NCTId: {trial_id}, Title: {title}, {score}")

            top_knn_matches.append({
                "PatientID": patient["PatientID"],
                "TrialID": trial_id,
                "Score": float(score.replace('Match Score = ', '').replace('%', ''))
            })

    return top_knn_matches


if __name__ == "__main__":
    patients_df = load_patient_data("data/patient_data.csv")
    trials_df = load_trial_data("data/clinical_trials.csv")

    patients_df["Keywords"] = patients_df["Keywords"].fillna("").astype(str)
    trials_df["Eligibility"] = trials_df["Eligibility"].fillna("").astype(str)

    print("Sample patient keywords:\n", patients_df['Keywords'].head(), "\n")
    print("Sample trial eligibility texts:\n", trials_df['Eligibility'].head(), "\n")

    corpus = patients_df["Condition"].tolist() + trials_df["Eligibility"].tolist()
    w2v_model = train_word2vec(corpus)
    tfidf_vectorizer = fit_tfidf_vect(corpus)

    top_matches = match_patients_to_trials(patients_df, trials_df, w2v_model, tfidf_vectorizer, top_n=5)

    pd.DataFrame(top_matches).to_csv("data/patient_trial_knn_top5.csv", index=False)
    print("\nSaved top 5 KNN-based matches to 'data/patient_trial_knn_top5.csv'")
