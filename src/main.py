"""
This script carries out the process of matching our mock patients to real clinical trials. 
We use the clinical trials dataset and the patient dataset. Ultimately, it will leverages NLP
to score how well each patient fits the matchmaking criteria of each trial.
"""
import numpy as np
import pandas as pd
from src.patient_data_ingestion import load_patient_data
from src.nlp_matching import compute_similarity
from src.nlp_matching import compute_similarity_bulk
from src.predictive_model import knn_mutiple_patients, knn

from src.word_embedding import (
    train_word2vec,
    fit_tfidf_vect,
    weighted_embedding_bulk, weighted_sentence_embedding
)


def load_trial_data(filepath):
    """
    Loads clinical trials from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def combine_scores(nlp_score, knn_distance, alpha=0.7, beta=0.3):
    """
    Combines NLP score and KNN distance into a single score.
    Higher is better.
    """
    knn_score = 1 / (1 + knn_distance)  # Convert distance to similarity
    return alpha * nlp_score + beta * knn_score

# function that simply computes the distances between clinical trail and patient vectors
def compute_knn_distances(patient_vector, trial_vectors):
   knn_distances = []
   patient_features = np.array(patient_vector[1:], dtype=float)


   for trial in trial_vectors:
       trial_features = np.array(trial[1:], dtype=float)
       # get the distance between the current clinical trial & patient
       distance = np.linalg.norm(patient_features - trial_features)
       # add it to a list
       knn_distances.append(distance)


   return knn_distances

def match_patients_to_trials(patients_df, trials_df, w2v_model, tfidf_vectorizer, top_n=5):
    """
    For each patient, compute NLP similarity and KNN distance to each trial.
    Combine them into a final score and rank trials by it.
    """
    # Placeholder: you would typically extract features from your data here
    trial_vectors =  weighted_embedding_bulk(trials_df, "trial", tfidf_vectorizer, w2v_model)  # placeholder feature matrix
    for _, patient in patients_df.iterrows():
        patient_keywords = patient['Condition']
        eligibility_texts = trials_df['Eligibility'].tolist()

        # Compute NLP scores
        nlp_scores = []
        for eligibility_text in eligibility_texts:
            nlp_score = compute_similarity(patient_keywords, eligibility_text)
            nlp_scores.append(nlp_score)

        # Compute KNN distances (dummy values for now)
        patient_vector = weighted_sentence_embedding(str(patient.get('Keywords', '')), patient.get('PatientID', 'Unknown'), tfidf_vectorizer, w2v_model)  # placeholder feature vector
        # embedding_with_id = np.concatenate(([patient.get('PatientID', 'Unknown')], patient_vector.astype(object)))


        knn_distances = compute_knn_distances(patient_vector, trial_vectors)

        knn_writen_results = knn(patient_vector, trial_vectors, 5)
        # print(knn_writen_results)

        # Combine both scores
        final_scores = []
        for idx, trial in trials_df.iterrows():
            combined_score = combine_scores(nlp_scores[idx], knn_distances[idx])
            final_scores.append((trial['NCTId'], trial['Title'], combined_score))

        # Sort by combined score
        final_scores.sort(key=lambda x: x[2], reverse=True)

        print(f"\nTop {top_n} matches for Patient ID {patient['PatientID']}:")
        for nctid, title, score in final_scores[:top_n]:
            print(f"  NCTId: {nctid}, Title: {title}, Combined Score: {score:.2f}")


if __name__ == "__main__":
    patients_df = load_patient_data("data/patient_data.csv")
    trials_df = load_trial_data("data/clinical_trials.csv")
    patients_df["Keywords"] = patients_df["Keywords"].fillna("").astype(str)
    trials_df["Eligibility"] = trials_df["Eligibility"].fillna("").astype(str)
    print("Sample patient keywords:", patients_df['Keywords'].head())
    print("Sample trial eligibility:", trials_df['Eligibility'].head())
    # Train Word2Vec and TF-IDF
    corpus = patients_df["Condition"].tolist() + trials_df["Eligibility"].tolist()
    w2v_model = train_word2vec(corpus)
    tfidf_vectorizer = fit_tfidf_vect(corpus)
    match_patients_to_trials(patients_df, trials_df, w2v_model, tfidf_vectorizer, top_n=5)


    print("\nRunning bulk similarity scoring...")
    similarity_df = compute_similarity_bulk(patients_df, trials_df)
    print("Sample similarity scores:", similarity_df.head())
    similarity_df.to_csv("data/patient_trial_similarity.csv", index=False)
    print("Saved patient trial data successfully.")
