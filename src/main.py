"""
This script carries out the process of matching our mock patients to real clinical trials. 
We use the clinical trials dataset and the patient dataset. Ultimately, it will leverages NLP
to score how well each patient fits the matchmaking criteria of each trial.
"""
import pandas as pd
from src.patient_data_ingestion import load_patient_data
from src.predictive_model import knn

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

'''
we did not end up using these methods - instead computed via knn method from predictive_model :)

def combine_scores(nlp_score, knn_distance, alpha=0.7, beta=0.3):
    """
    Combines NLP score and KNN distance into a single score.
    Higher is better.
    """
    knn_score = 1 / (1 + knn_distance)  # Convert distance to similarity
    return alpha * nlp_score + beta * knn_score

# function that simply computes the knn distances between clinical trail and patient vectors
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
'''
top_knn_matches = []

def match_patients_to_trials(patients_df, trials_df, w2v_model, tfidf_vectorizer, top_n=5):
    """
    For each patient, compute NLP similarity and KNN distance to each trial.
    Combine them into a final score and rank trials by it.
    """

    trial_vectors =  weighted_embedding_bulk(trials_df, "trial", tfidf_vectorizer, w2v_model)
    for _, patient in patients_df.iterrows():

        '''
        # compute NLP scores (computing cosine similarity)
        patient_keywords = patient['Condition']
        eligibility_texts = trials_df['Eligibility'].tolist()
        
        nlp_scores = []
        for eligibility_text in eligibility_texts:
            nlp_score = compute_similarity(patient_keywords, eligibility_text)
            nlp_scores.append(nlp_score)
            
        knn_distances = compute_knn_distances(patient_vector, trial_vectors) # computation for score combo  
        
        # combine both scores (cosine similarity + knn vector distances)
        final_scores = []
        for idx, trial in trials_df.iterrows():
            combined_score = combine_scores(nlp_scores[idx], knn_distances[idx])
            final_scores.append((trial['NCTId'], trial['Title'], combined_score))  
            
        # sort by combined score
        final_scores.sort(key=lambda x: x[2], reverse=True)

        # print in an easily legible manner - the top 5 matches for a given patient
        print(f"\nTop {top_n} matches for Patient ID {patient['PatientID']}:")
        for nctid, title, score in final_scores[:top_n]:
            print(f"  NCTId: {nctid}, Title: {title}, Combined Score: {score:.2f}")    
        '''

        patient_vector = weighted_sentence_embedding(str(patient.get('Keywords', '')), patient.get('PatientID', 'Unknown'), tfidf_vectorizer, w2v_model)

        knn_writen_results = knn(patient_vector, trial_vectors,top_n)  # computation for percentages (not combined with cosine similarity)

        final_scores = []
        for idx, trial in trials_df.iterrows():
            for match in knn_writen_results:
                id = match.split(',')[0]
                if id == trial['NCTId']:
                    final_scores.append((trial['NCTId'], trial['Title'], match.split(",", 1)[1]))

        # sort the results such that the highest match for the patient-trial is at the top of output
        final_scores.sort(key=lambda x: x[2], reverse=True)

        # print in an easily legible manner - the top n matches for a given patient
        print(f"\nTop {top_n} matches for Patient ID {patient['PatientID']}: a {patient['Age']} year old {patient['Sex']} seeking help with {patient['Condition']} and {patient['Keywords']}")
        for nctid, title, percent in final_scores:
            print(f"  NCTId: {nctid}, Title: {title}, {percent}")
            top_knn_matches.append({
                "PatientID": patient["PatientID"],
                "TrialID": nctid,
                "Score": float(percent.strip().replace('Match Score = ', '').replace('%', ''))
            })


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

    knn_df = pd.DataFrame(top_knn_matches)
    knn_df.to_csv("data/patient_trial_knn_top5.csv", index=False)
    print("\nSaved top 5 KNN-based matches to 'data/patient_trial_knn_top5.csv'")





