import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# Wrap long titles
wrapped_titles = [textwrap.fill(title, width=40) for title in title_counts.index]

# Plot setup
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
barplot = sns.barplot(x=title_counts.values, y=wrapped_titles, palette="Blues_d")

# Add value labels to the right of bars
for i, v in enumerate(title_counts.values):
    plt.text(v + 0.5, i, str(v), va='center', fontweight='bold')

plt.title("Top 10 Most Frequently Matched Clinical Trials (by Title)", fontsize=14)
plt.xlabel("Number of Patients Matched", fontsize=12)
plt.ylabel("Trial Title", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
