import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# Load top-5 match results
matches = pd.read_csv("data/patient_trial_knn_top5.csv")

# Load trial metadata with titles
trials = pd.read_csv("data/clinical_trials.csv")
trials = trials.rename(columns={"NCTId": "TrialID"})  # For joining

# Merge matches with trial titles
merged = matches.merge(trials[["TrialID", "Title"]], on="TrialID", how="left")

# Count how many times each trial title appears
title_counts = merged["Title"].value_counts().head(10)

# Wrap long titles to fit better in the chart
wrapped_titles = [textwrap.fill(title, width=50) for title in title_counts.index]

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x=title_counts.values, y=wrapped_titles)
plt.title("Top 10 Most Frequently Matched Clinical Trials (by Title)")
plt.xlabel("Number of Patients Matched")
plt.ylabel("Trial Title")
plt.tight_layout()
plt.show()
