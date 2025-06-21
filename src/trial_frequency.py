"""
This file is designed to visualize the top 10 most frequently matched clinical trials in a bar chart.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# Load data
matches = pd.read_csv("data/patient_trial_knn_top5.csv")
trials = pd.read_csv("data/clinical_trials.csv").rename(columns={"NCTId": "TrialID"})
merged = matches.merge(trials[["TrialID", "Title"]], on="TrialID", how="left")
title_counts = merged["Title"].value_counts().head(10)

# Wrap long titles onto multiple lines
wrapped_titles = [textwrap.fill(title, width=50) for title in title_counts.index]

# Normalize counts for soft gradient mapping
norm = plt.Normalize(title_counts.min(), title_counts.max())
base_colors = sns.color_palette("Blues", n_colors=100)[30:]  
mapped_colors = [base_colors[int(norm(v) * (len(base_colors) - 1))] for v in title_counts.values]

# Plot
plt.figure(figsize=(13, 9))
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

barplot = sns.barplot(
    x=title_counts.values,
    y=wrapped_titles,
    palette=mapped_colors
)

# Add value labels to bars
for i, v in enumerate(title_counts.values):
    plt.text(v + 0.5, i, str(v), va='center', fontsize=9, fontweight='bold')

# Titles and labels
plt.title("Top 10 Most Frequently Matched Clinical Trials", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Number of Patients Matched", fontsize=12)
plt.ylabel("") 

# Axis formatting
plt.xticks(fontsize=10)
plt.yticks(fontsize=9)
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()
