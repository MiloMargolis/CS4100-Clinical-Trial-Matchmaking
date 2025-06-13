import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# Load top-5 match results
matches = pd.read_csv("data/patient_trial_knn_top5.csv")

# Load trial metadata with titles
trials = pd.read_csv("data/clinical_trials.csv")
trials = trials.rename(columns={"NCTId": "TrialID"})  # Match join key

# Merge matches with trial titles
merged = matches.merge(trials[["TrialID", "Title"]], on="TrialID", how="left")

# Count most frequently matched trials by title
title_counts = merged["Title"].value_counts().head(10)

# Wrap long titles for better y-axis formatting
wrapped_titles = [textwrap.fill(title, width=40) for title in title_counts.index]

# Plot
plt.figure(figsize=(12, 10))
sns.set_style("whitegrid")
barplot = sns.barplot(x=title_counts.values, y=wrapped_titles, palette="Blues_r")

# Add value labels to the right of bars
for i, v in enumerate(title_counts.values):
    plt.text(v + 0.5, i, str(v), va='center', fontweight='bold')

# Formatting
plt.title("Top 10 Most Frequently Matched Clinical Trials (by Title)", fontsize=14)
plt.xlabel("Number of Patients Matched", fontsize=12)
plt.ylabel("Trial Title", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
