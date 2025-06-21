#
from src.predictive_model import knn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



# patients_df = pd.read_csv("data/patient_data.csv")
# trials_df = pd.read_csv("data/clinical_trials.csv")
# # #get one patient
# # patient = patients_df.iloc[0]


# corpus = patients_df["Condition"].tolist() + trials_df["Eligibility"].tolist()
# w2v_model = train_word2vec(corpus)
# tfidf_vectorizer = fit_tfidf_vect(corpus)

# #get their vector info
# patient_vector = weighted_sentence_embedding(str(patient.get('Keywords', '')), patient.get('PatientID', 'Unknown'), tfidf_vectorizer, w2v_model)
# trial_vectors =  np.array(weighted_embedding_bulk(trials_df, "trial", tfidf_vectorizer, w2v_model))

# trials = trials_df.values.tolist()



# #get the top k
# k = 5
# trail_matches = knn(patient_vector, trial_vectors, k)



# topk_vectors = [np.array(trial[2]) for trial in trail_matches]
# topk_ids = [trial[0] for trial in trail_matches]
# topk_scores = [trial[1] for trial in trail_matches]

# # Combine into single array with patient
# all_vectors = np.vstack([patient_vector] + topk_vectors)
# tsne = TSNE(n_components=2, random_state=42, perplexity=5)
# reduced = tsne.fit_transform(all_vectors)

# # Split result
# patient_2d = reduced[0]
# topk_2d = reduced[1:]

# # Plotting
# plt.figure(figsize=(10, 7))
# plt.scatter(patient_2d[0], patient_2d[1], color='red', s=120, edgecolors='black', label='Patient')

# # Plot each trial
# for i in range(k):
#     x, y = topk_2d[i]
#     trial_id = topk_ids[i]
#     score = topk_scores[i]
#     plt.scatter(x, y, color='blue', alpha=0.7, edgecolors='black')
#     plt.text(x + 0.5, y, f"{trial_id}\n{score:.1f}%", fontsize=9)

# plt.title(f"Patient vs Top {k} Closest Clinical Trials (t-SNE 2D)")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# Visualize one random patient and their top-k trials
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import numpy as np
# import random


# from src.word_embedding import train_word2vec, fit_tfidf_vect, weighted_embedding_bulk, weighted_sentence_embedding


# patients_df = pd.read_csv("data/patient_data.csv")
# trials_df = pd.read_csv("data/clinical_trials.csv")
# # #get one patient
# # patient = patients_df.iloc[0]


# # Pick a random patient
# random_patient = patients_df.sample(1).iloc[0]

# corpus = patients_df["Condition"].tolist() + trials_df["Eligibility"].tolist()
# w2v_model = train_word2vec(corpus)
# tfidf_vectorizer = fit_tfidf_vect(corpus)

# # Get their features
# patient_vector = weighted_sentence_embedding(
#     str(random_patient.get('Keywords', '')),
#     random_patient.get('PatientID', 'Unknown'),
#     tfidf_vectorizer,
#     w2v_model
# )

# # Get all trial features
# trial_vectors = weighted_embedding_bulk(trials_df, "trial", tfidf_vectorizer, w2v_model)

# # Run KNN for top-k matches
# k = 5
# trail_matches = knn(patient_vector, trial_vectors, k)

#  # Extract vectors and metadata
# topk_indices = [i for i, trial in enumerate(trial_vectors) if trials_df.iloc[i]['NCTId'] in [match[0] for match in trail_matches]]
# topk_vectors = [trial_vectors[i] for i in topk_indices]
# topk_ids = [match[0] for match in trail_matches]
# topk_scores = [match[1] for match in trail_matches]

# # Stack for t-SNE
# patient_numeric = np.array(patient_vector[1:], dtype=float)
# topk_numeric_vectors = [np.array(vec[1:], dtype=float) for vec in topk_vectors]

# # Stack for t-SNE
# all_vectors = np.vstack([patient_numeric] + topk_numeric_vectors)
# tsne = TSNE(n_components=2, perplexity=5, random_state=42)
# reduced = tsne.fit_transform(all_vectors)

# patient_2d = reduced[0]
# topk_2d = reduced[1:]

# # Plotting
# plt.figure(figsize=(10, 7))
# plt.scatter(patient_2d[0], patient_2d[1], color='red', s=120, edgecolors='black', label='Patient')

# for i in range(k):
#     x, y = topk_2d[i]
#     plt.scatter(x, y, color='blue', alpha=0.7, edgecolors='black')
#     plt.text(x + 0.4, y, f"{topk_ids[i]}\n{topk_scores[i]:.1f}%", fontsize=9)

# plt.title(f"Random Patient vs Top {k} Closest Clinical Trials (t-SNE 2D)")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# import random
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import numpy as np





# # Assuming top_knn_matches is filled (list of dicts with PatientID, TrialID, Score)
# # Pick a random patient from the matches
# random_patient_id = random.choice(list(set(m['PatientID'] for m in top_knn_matches)))

# # Filter matches for that patient
# patient_matches = [m for m in top_knn_matches if m['PatientID'] == random_patient_id]

# # Get the patient vector
# patient_row = patients_df[patients_df['PatientID'] == random_patient_id].iloc[0]
# patient_vector = weighted_sentence_embedding(
#     str(patient_row.get('Keywords', '')),
#     patient_row.get('PatientID', 'Unknown'),
#     tfidf_vectorizer,
#     w2v_model
# )


# patient_numeric = np.array(patient_vector[1:], dtype=float)

# # Get trial vectors for matched trials
# matched_trial_ids = [m['TrialID'] for m in patient_matches]
# matched_scores = [m['Score'] for m in patient_matches]

# matched_trials_rows = trials_df[trials_df['NCTId'].isin(matched_trial_ids)]
# trial_vectors_all = weighted_embedding_bulk(trials_df, "trial", tfidf_vectorizer, w2v_model)

# # Map trial IDs to their numeric vectors
# trial_id_to_vec = {vec[0]: np.array(vec[1:], dtype=float) for vec in trial_vectors_all}

# matched_trial_vectors = [trial_id_to_vec[tid] for tid in matched_trial_ids]



# all_embeds = np.vstack([patient_numeric] + matched_trial_vectors)
    
# tsne = TSNE(n_components=2, random_state=42, perplexity=5)
# reduced = tsne.fit_transform(all_embeds)
    
# patient_2d = reduced[0]
# trials_2d = reduced[1:]
    
# plt.figure(figsize=(10,7))
# plt.scatter(patient_2d[0], patient_2d[1], color='red', s=150, edgecolors='black', label=f'Patient {random_patient_id}')
    
# for i, (x, y) in enumerate(trials_2d):
#     plt.scatter(x, y, color='blue', alpha=0.7, edgecolors='black')
#     plt.text(x + 0.2, y, f"{matched_trial_ids[i]}\n{matched_scores[i]:.2f}%", fontsize=9)
    
# plt.title(f"t-SNE Visualization: Patient {random_patient_id} vs Their Top Trials")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.legend()
# plt.grid(True)
# plt.show()


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.word_embedding import train_word2vec, fit_tfidf_vect, weighted_embedding_bulk, weighted_sentence_embedding
from src.topktrails import knn  # Adjust import path if needed


if __name__ == "__main__":
    # Load CSV data
    patients_df = pd.read_csv("data/patient_data.csv")
    trials_df = pd.read_csv("data/clinical_trials.csv")

    # Prepare corpus and models
    corpus = patients_df["Condition"].tolist() + trials_df["Eligibility"].tolist()
    w2v_model = train_word2vec(corpus)
    tfidf_vectorizer = fit_tfidf_vect(corpus)

    # Precompute trial embeddings
    trial_vectors_all = weighted_embedding_bulk(trials_df, "trial", tfidf_vectorizer, w2v_model)

    # Map trial ID to numeric vector
    trial_id_to_vec = {vec[0]: np.array(vec[1:], dtype=float) for vec in trial_vectors_all}

    # Collect top knn matches for all patients
    top_knn_matches = []
    k = 5

    for _, patient in patients_df.iterrows():
        patient_vector = weighted_sentence_embedding(
            str(patient.get('Keywords', '')),
            patient.get('PatientID', 'Unknown'),
            tfidf_vectorizer,
            w2v_model
        )

        knn_results = knn(patient_vector, trial_vectors_all, k)  # List of "TrialID, Match Score = xx.xx%"

        for match_str in knn_results:
            trial_id = match_str.split(",")[0]
            score_str = match_str.split(",")[1].replace("Match Score =", "").replace("%", "").strip()
            score = float(score_str)
            top_knn_matches.append({
                "PatientID": patient["PatientID"],
                "TrialID": trial_id,
                "Score": score
            })

    # Pick a random patient from matches
    random_patient_id = random.choice(list(set(m['PatientID'] for m in top_knn_matches)))

    # Get all matches for this patient
    patient_matches = [m for m in top_knn_matches if m['PatientID'] == random_patient_id]

    # Get patient vector (numeric only)
    patient_row = patients_df[patients_df['PatientID'] == random_patient_id].iloc[0]
    patient_vector = weighted_sentence_embedding(
        str(patient_row.get('Keywords', '')),
        patient_row.get('PatientID', 'Unknown'),
        tfidf_vectorizer,
        w2v_model
    )
    patient_numeric = np.array(patient_vector[1:], dtype=float)

    # Get matched trials vectors and scores
    matched_trial_ids = [m['TrialID'] for m in patient_matches]
    matched_scores = [m['Score'] for m in patient_matches]
    matched_trial_vectors = [trial_id_to_vec[tid] for tid in matched_trial_ids]

    # Stack embeddings for t-SNE
    all_embeds = np.vstack([patient_numeric] + matched_trial_vectors)

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced = tsne.fit_transform(all_embeds)

    patient_2d = reduced[0]
    trials_2d = reduced[1:]

    # Plot
    plt.figure(figsize=(10,7))
    plt.scatter(patient_2d[0], patient_2d[1], color='red', s=150, edgecolors='black', label=f'Patient {random_patient_id}')

    for i, (x, y) in enumerate(trials_2d):
        plt.scatter(x, y, color='blue', alpha=0.7, edgecolors='black')
        plt.text(x + 0.2, y, f"{matched_trial_ids[i]}\n{matched_scores[i]:.2f}%", fontsize=9)

    plt.title(f"t-SNE: Patient {random_patient_id} vs Their Top {k} Clinical Trials")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


