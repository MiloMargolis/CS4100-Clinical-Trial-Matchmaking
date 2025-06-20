"""
This file uses Word2Vec and TFIDFVectorizer to compute word embeddings, word weightings, and weighted embeddings.
These computed vectors are then used in KNN predictions.
"""

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
import re

# tokenizer to preprocess the text but retain numbers in medical terms
def preprocess_text(text):
    # use regex to split the text into tokens (words and number) and convert to lowercase
    tokens = re.findall(r'\b\w+\b', text.lower())
    # remove stopword (for example common words like "the", "and", etc) from the list of tokens
    return [token for token in tokens if token not in STOPWORDS]

# trains the word2vec given corpus (the text) by turning words in corpus to vectors
# first preprocess text using preprocess_text above
def train_word2vec(corpus):
    # preprocess the list of strings - and removes any stopwords
    tokenized_corpus = [preprocess_text(text) for text in corpus]
    # train the w2v model with vector size 100, window 5 (# of words on either side of target to consider as context)
    # and the min_count of words of 1 - meaning that even if it only appears once it is added to vocab
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1)
    return w2v_model

# to give more weight to the less common and more significant terms (i.e., medical terminology)
# trains a tfidf vectorizer with the given text!
def fit_tfidf_vect(text):
    # use preprocess text function for consistent tokenization
    return TfidfVectorizer(tokenizer=preprocess_text).fit(text)

# given a string, the id (name) of patient or trial, a trained tfidf_vect, and trained word2vec model
# to compare semantic meaning of sentences - returns single vector (weighted average of word vectors) for sentence
# uses tf-idf scores as weights
def weighted_sentence_embedding(text, id, tfidf_vectorizer, w2v_model):
    # preprocess text
    words = preprocess_text(text)

    tfidf_weights = tfidf_vectorizer.transform([" ".join(words)]) # transform into feature vector (sparse matrix)
    tfidf_vocab = tfidf_vectorizer.get_feature_names_out()  # retrieves vocab learned by tfidf model
    # makes a dict (key = word, value = tfidf weight)
    word_to_weight = dict(zip(tfidf_vocab, tfidf_weights.toarray()[0]))

    embeddings = [] # for w2v word vectors
    weights = [] # for tfidf weights
    # to add the clinical trial or patient name at start of vector array
    string_arr_setup = np.array([id], dtype=object)

    for word in words:
        if word in w2v_model.wv and word in word_to_weight: # if in w2v vocab & tfidf weight
            vec = w2v_model.wv[word] # retrieve word's vector from w2v
            weight = word_to_weight[word]
            embeddings.append(vec) # store word embedding
            weights.append(weight) # store the weight of the word for being in the sentence

    if not embeddings: # for edge case
        return np.concatenate((string_arr_setup, np.zeros(w2v_model.vector_size).astype(object)))

    # to compute weighted average for sentence (a vector for sentence)
    embeddings = np.array(embeddings)
    weights = np.array(weights)
    # compute the weighted avg embedding
    sentence_embedding = np.average(embeddings, axis=0, weights=weights)
    # normalize the embedding to unit vec
    norm = np.linalg.norm(sentence_embedding)
    if norm > 0:
        sentence_embedding = sentence_embedding / norm

    return np.concatenate((string_arr_setup, sentence_embedding.astype(object)))

# compute the weighted embedding for multiple data trails or multiple patients - and put them all
# in a list to be used by knn predictive_model
# computes the weighted embedding for each patient or trial individually, then appends to result list
# returns results as [[trialID, num, num, num], [trial2ID, num, num, num]]
# inputs: dataframe of patients or trials, string of "patient" or "trial" to indicate who to get info for,
#           the trained tfidf vect, and trained w2v model
def weighted_embedding_bulk(specific_df, what, tfidf_vectorizer, w2v_model):
    results = []

    if what == "patient":
        for _, patient in specific_df.iterrows():
            patient_id = patient.get('PatientID', 'Unknown')
            patient_text = str(patient.get('Keywords', '')) # fit into a list below to work with weighted_emb function
            results.append(weighted_sentence_embedding(patient_text, patient_id, tfidf_vectorizer, w2v_model))


    if what == "trial":
        for _, trial in specific_df.iterrows():
            trial_id = trial.get('NCTId', 'Unknown')
            trial_text = str(trial.get('Eligibility', '')) # fit into a list below to work with weighted_emb function
            results.append(weighted_sentence_embedding(trial_text, trial_id, tfidf_vectorizer, w2v_model))

    return results

# save the trained word2vec model 
def save_w2v_model(model, path):
    model.save(path)

# load in a previously saved word2vec model from the disk
def load_w2v_model(path):
    return Word2Vec.load(path)