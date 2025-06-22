# Clinical Trial Matcher

This project aims to improve the process of matching **patients** to clinical trials by leveraging **Natural Language Processing (NLP)** and **predictive modeling**. The system is designed to work with **anonymized patient datasets** (e.g. from hospitals or research partners) and match them in bulk to relevant clinical trials. This approach aims to streamline recruitment, reduce delays, and make trial access more equitable.

## Project Overview

Modern clinical trials often struggle with recruitment failures and design mismatches, delaying medical advancements. This system seeks to:

- Parse **clinical trial eligibility criteria** from [ClinicalTrials.gov](https://clinicaltrials.gov/)
- Ingest **bulk patient data** (anonymized)
- Use **NLP techniques** to compare patient data with eligibility criteria
- Rank trials based on **fit** and (optionally) **success probability**

## Document Overview
- requirements.txt: project dependencies

**data** directory:
  - clinical_trials.csv: clinical trials data
  - patient_data.csv: patient data
  - patient_trial_knn_top5.csv: match results from running main.py

**src** directory:
- main.py: match patients to trials (using NLP and KNN)
- word_embedding.py: create word embeddings, weightings, and weighted embeddings
- predictive_model.py: knn match-making predictions
- data_ingestion.py: load clinical trial data
- visualize_data.py: visualize embeddings
- patient_data_ingestion.py: load patient data
- nlp_matching.py (not used in final product): computes cosine similarity (not used in final product)
- analyze_non_cancer_matches.py: analyzes non-cancer matches
- trial_frequency.py: visualize 10 most frequently matched trials

## Features

- **Data Integration**: Fetch clinical trial data from ClinicalTrials.gov (API).
- **Patient Matching**: Match patients to trials using:
  - Structured fields (age, sex, condition)
  - NLP-based keyword matching for eligibility criteria.
  - KNN matchmaking algorithmn
- **Predictive Modeling**: (Future steps) Predict the likelihood of trial success using a Decision Tree. 
- **Visualizations**: Display relevant plots (e.g. matching distributions).

## Data Sources

- **ClinicalTrials.gov**: Clinical trial data including eligibility criteria.
- **Patient Data**: Anonymized datasets (e.g. CSV) with fields such as:
  - Age
  - Sex
  - Condition
  - Additional keywords (optional)
- **Trial Success Labels**: ClinicalTrials.gov status (e.g. Completed, Terminated).

## Methodology

1. **Data Ingestion**
   - Use `requests` and `pandas` to load clinical trial data.
   - Load patient data from CSV or database.
2. **NLP Preprocessing**
   - For preprocessing: use `regex` (tokenization and lowercasing) and `gensim` (stopword removal)
   - After preprocessing, a Word2Vec model and TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer are trained on full corpus of all trials and patients
     - Use `gensim` for Word2Vec (word embedding - by semantic meaning and context)
     - Use `scikit-learn` for TF-IDF Vectorizer (word weighing - by importance and rarity)
   - The final embeddings are computed as TF-IDF weighted averages of the Word2Vec vectors and normalized to unit vectors. 
     - Essentially converting all patient conditions and trial eligibilities to vectors
3. **Similarity Matching**
   - Use **K-Nearest Neighbors (KNN)** to find the top k clinical trails that match for the patient using the patient's and trails' vectors.
   - Calculate **Euclidean distance** between the patient and trails' embeddings and convert it to a match score using 100 / (1 + distance).
4. **Ranking**
   - Use the Match scores calculated which helps to assign higher scores to closer matches and output the top K best fit trails.

## Next Steps

### Bias Towards Cancer Trials
Our current analysis reveals a strong bias toward cancer related trials in the top match results:
- Only 3.8% of the top matched trials were non cancer related
- The model consistently matched patients to a small subset of non-cancer trials (for example scoliosis)
- This is most likely due to the Word2Vec and TF-IDF embeddings being trained on a cancer heavy corpus which resulted in a narrow semantic space.

### Improvements for Future
- Diversify the training corpus by including more non-cancer trials
- Separate embeddings or models by condition category to avoid domain contamination
- Refactor eval process to track match diversity and clinical relevance by each medical field

### Trial Likelihood Scoring 
- Training a DecisionTreeClassifier using metadata such as the trial phase, sponsor type, location, completion rate, and so on
- Include a success likelihood alongside the match score for a more holistic view
- This would help prioritize not only relevant trials, but also trials that are likely to be completed.

## Running the project
- Clone the repository with: git clone <repo_url>
- Install the dependencies listed in requirements.txt with: pip install -r requirements.txt 
- Run python src/main.py
  - This will load the data, train the embeddings preforms the matching, and save the output to a CSV (patient_trial_knn_top5.csv).

## AI Citations:
- ChatGPT-4o
- Claude Opus 4
- Github Copilot
