# Clinical Trial Matcher

This project aims to improve the process of matching **patients** to clinical trials by leveraging **Natural Language Processing (NLP)** and **predictive modeling**. The system is designed to work with **anonymized patient datasets** (e.g. from hospitals or research partners) and match them in bulk to relevant clinical trials. This approach aims to streamline recruitment, reduce delays, and make trial access more equitable.

## Project Overview

Modern clinical trials often struggle with recruitment failures and design mismatches, delaying medical advancements. This system seeks to:

- Parse **clinical trial eligibility criteria** from [ClinicalTrials.gov](https://clinicaltrials.gov/)
- Ingest **bulk patient data** (anonymized)
- Use **NLP techniques** to compare patient data with eligibility criteria
- Rank trials based on **fit** and (optionally) **success probability**

## Document Overview
**data** directory:
  - clinical_trials.csv
  - patient_data.csv
  - patient_embeddings.csv
  - trial_embeddings.csv
  - patient_trial_similarity.csv

**src** directory:
- main.py
- word_embedding.py
- predictive_model.py
- data_ingestion.py
- visualize_data.py
- word_embedding.py
- patient_data_ingestion.py
- nlp_matchking.py (original test, not used in final product)

## Features

- **Data Integration**: Fetch clinical trial data from ClinicalTrials.gov (API).
- **Patient Matching**: Match patients to trials using:
  - Structured fields (age, sex, condition)
  - NLP-based keyword matching for eligibility criteria.
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
   - Use `gensim` for preprocessing, tokenization, and Word2Vec (word embedding).
   - Afer tokenization and stopword removal, a Word2Vec model is trainied on the full corpus. Then the TF-IDF is used to weigh the word by importance and rarity.
   - The final sentence embedings are computed as TF-IDF weighted averages of the Word2Vec vectors and normalized to unit vectors. 
3. **Similarity Matching**
   - Use `scikit-learn` for TF-IDF (Term Frequency-Inverse Document Frequency)
   - Convert patient profiles and trial eligibility criteria to vectors using `TfidfVectorizer`.
   - Calculate **Euclidean distance** between the vector embeddings and convert it to a match score using 100 / (1 + distance).
4. **Predictive Modeling (Future steps)**
   - Use a `DecisionTreeClassifier` from `scikit-learn` to estimate trial success probability.
5. **Ranking**
   - Combine similarity scores and success probabilities to rank trials for each patient. Match scores are calculated using the formula score = 100 / (1 + distance) which helps to assign higher scores to closer matches.

## Running the project
- Run python src/main.py
- This will load the data, train the embeddings preforms the matching, and save the output to a CSV.

## AI Citations:
- ChatGPT-4o
- Claude Opus 4
- Github Copilot
