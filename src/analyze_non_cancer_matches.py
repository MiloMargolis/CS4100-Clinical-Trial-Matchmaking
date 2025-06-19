import pandas as pd

# load in the data
matches_df = pd.read_csv("data/patient_trial_knn_top5.csv")
trials_df = pd.read_csv("data/clinical_trials.csv")
patients_df = pd.read_csv("data/patient_data.csv")

# flag any cancer related trials (based on results, we can even add more terms here)
cancer_terms = ["cancer", "tumor", "neoplasm", "oncology", "carcinoma", "sarcoma", "Mesothelioma", "NSCLC"]
trials_df["IsCancer"] = trials_df["Condition"].str.contains('|'.join(cancer_terms), case=False, na=False)


# connect the match results with trial cancer information 
merged = matches_df.merge(trials_df[["NCTId", "IsCancer"]], left_on="TrialID", right_on="NCTId", how="left")

# calculate the overall rate for cancer vs. non-cancer trails 
cancer_rate = merged["IsCancer"].mean()
non_cancer_rate = 1 - cancer_rate
print(f"🔍 {non_cancer_rate:.2%} of top matched trials are non-cancer-related.")

# inspect the samples (via the terminal)
sample_non_cancer = merged[~merged["IsCancer"]].head(5).merge(
    trials_df[["NCTId", "Title", "Condition"]],
    left_on="TrialID",
    right_on="NCTId",
    how="left"
).merge(
    patients_df[["PatientID", "Condition"]],
    on="PatientID",
    how="left",
    suffixes=("_Trial", "_Patient")
)

print("\n Sample non-cancer trial matches:")
print(sample_non_cancer[["PatientID", "Condition_Patient", "TrialID", "Condition_Trial", "Score"]])
