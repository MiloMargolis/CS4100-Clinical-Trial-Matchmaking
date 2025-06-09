from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess
import numpy as np
import pandas as pd
import re

# first preprocess text using simple_preprocess (makes lowercase, cuts non-alphabetical,
# cuts word of length < 2 and > 50, and turns into list)
# then trains by turning words in corpus to vectors
def train_word2vec(corpus, vector_size=100, window=5, min_count=1):  # can change this ??
    tokenized_corpus = [simple_preprocess(text, max_len=50) for text in corpus]  # preprocess the list of strings (corpus)
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count)
    return w2v_model

# usage example
# corpus = patients_df["Condition"].tolist() + trials_df["Eligibility"].fillna("").tolist()
# model = train_word2vec(corpus)

# to give more weight to the less common and more significant terms (i.e., medical terminology)
def get_tfidf_vectorizer(corpus_texts):
   return TfidfVectorizer().fit(corpus_texts)

# given a piece of text (string),
# to compare the semantic meaning of sentences - returns a single vector for the sentence
def get_weighted_sentence_embedding(text, tfidf_vectorizer, w2v_model):
    # Preprocess text
    words = simple_preprocess(text, max_len=50)

    tfidf_weights = tfidf_vectorizer.transform([" ".join(words)]) # merges words back into singular string
    feature_names = tfidf_vectorizer.get_feature_names_out() # retrieves vocab learned by tfidf model
    # makes a dict (key = word, value = tfidf weight)
    word_to_weight = dict(zip(feature_names, tfidf_weights.toarray()[0]))

    embeddings = [] # for w2v word vectors
    weights = [] # for tfidf weights

    for word in words:
        if word in w2v_model.wv and word in word_to_weight: # if in w2v vocab & tfidf weight
            vec = w2v_model.wv[word] # retrieve word's vector from w2v
            weight = word_to_weight[word]
            embeddings.append(vec) # store word embedding
            weights.append(weight) # store the weight of the word for being in the sentence

    if not embeddings: # for edge case
        return np.zeros(w2v_model.vector_size)

    # to compute weighted average for sentence (a vector for sentence)
    embeddings = np.array(embeddings)
    weights = np.array(weights)
    return np.average(embeddings, axis=0, weights=weights)


# compute cosine similarity between sentence embeddings??
def compute_similarity_w2v(patient_text, trial_text, w2v_model, tfidf_vectorizer):
    vec1 = get_weighted_sentence_embedding(patient_text, tfidf_vectorizer, w2v_model)
    vec2 = get_weighted_sentence_embedding(trial_text, tfidf_vectorizer, w2v_model)
    return cosine_similarity([vec1], [vec2])[0][0]
