"""
This file is designed to visualize the patient and the clinical trial embeddings in a 2D space.
We chose to use t-SNE; patients and trials are plotted in different colors in order 
understand clustering and proximity relationships.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne_embeddings(patient_embeds, trial_embeds, patient_ids, trial_ids):
    """
    Plots a t-SNE visualization of patient and trial embeddings.

    patient_embeds: numpy array [num_patients, embedding_dim]
    trial_embeds: numpy array [num_trials, embedding_dim]
    patient_ids: list or array of patient IDs
    trial_ids: list or array of trial IDs
    """
    # Combine embeddings
    all_embeds = np.vstack([patient_embeds, trial_embeds])

    # Fit t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced = tsne.fit_transform(all_embeds)


    # Split back
    n_patients = len(patient_embeds)
    patients_reduced = reduced[:n_patients]
    trials_reduced = reduced[n_patients:]

    # Plot
    plt.figure(figsize=(10, 7))
    plt.scatter(patients_reduced[:, 0], patients_reduced[:, 1], 
                c='blue', label='Patients', alpha=0.7, edgecolors='k')
    plt.scatter(trials_reduced[:, 0], trials_reduced[:, 1], 
                c='red', label='Clinical Trials', alpha=0.7, edgecolors='k')
    plt.legend()
    plt.title("t-SNE Visualization of Patient and Trial Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()

def load_fake_embeddings():
    """
    Loads patient and trial embeddings from CSV files.
    """
    patients_df = pd.read_csv("data/patient_embeddings.csv")
    trials_df = pd.read_csv("data/trial_embeddings.csv")
    patient_embeds = patients_df.iloc[:, 1:].values
    trial_embeds = trials_df.iloc[:, 1:].values
    patient_ids = patients_df['PatientID'].tolist()
    trial_ids = trials_df['NCTId'].tolist()
    return patient_embeds, trial_embeds, patient_ids, trial_ids

if __name__ == "__main__":
    # Load embeddings from CSV files
    patient_embeds, trial_embeds, patient_ids, trial_ids = load_fake_embeddings()

    # Plot
    plot_tsne_embeddings(patient_embeds, trial_embeds, patient_ids, trial_ids)