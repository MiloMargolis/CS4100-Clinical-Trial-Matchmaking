from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess
import numpy as np
import pandas as pd
import re

<<<<<<< Updated upstream
# for the preprocessing of text
def preprocess(text):
    text = text.lower() # converts to lowercase
    text = re.sub(r'[^a-z\s]', '', text) # removes any characters that are not alphabetical
    return text.strip().split() # splits it into a list of words

# word2vec usage - to represent each word as a vector
#chsnge
=======
# first preprocess text using simple_preprocess (makes lowercase, cuts non-alphabetical,
# cuts word of length < 2 and > 50, and turns into list)
# then trains by turning words in corpus to vectors
>>>>>>> Stashed changes
def train_word2vec(corpus, vector_size=100, window=5, min_count=1): # can change this ??
    tokenized_corpus = [simple_preprocess(text, max_len=50) for text in corpus] # preprocess the list of strings (corpus)
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count)
    return w2v_model

# usage example
# corpus = patients_df["Condition"].tolist() + trials_df["Eligibility"].fillna("").tolist()
# model = train_word2vec(corpus)

def get_weighted_sentence_embedding(text, tfidf_vectorizer, w2v_model):
    # Preprocess text
    words = simple_preprocess(text, max_len=50)

    # TF-IDF transformation
    tfidf_weights = tfidf_vectorizer.transform([" ".join(words)])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    word_to_weight = dict(zip(feature_names, tfidf_weights.toarray()[0]))

    # Weighted embeddings
    embeddings = []
    weights = []

    for word in words:
        if word in w2v_model.wv and word in word_to_weight:
            vec = w2v_model.wv[word]
            weight = word_to_weight[word]
            embeddings.append(vec)
            weights.append(weight)

    if not embeddings:
        return np.zeros(w2v_model.vector_size)

    # Compute weighted average
    embeddings = np.array(embeddings)
    weights = np.array(weights)
    return np.average(embeddings, axis=0, weights=weights)

# compute cosine similarity between sentence embeddings?
def compute_similarity_w2v(patient_text, trial_text, w2v_model, tfidf_vectorizer):
    vec1 = get_weighted_sentence_embedding(patient_text, tfidf_vectorizer, w2v_model)
    vec2 = get_weighted_sentence_embedding(trial_text, tfidf_vectorizer, w2v_model)
    return cosine_similarity([vec1], [vec2])[0][0]
