from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re

# for the preprocessing of text
def preprocess(text):
    text = text.lower() # converts to lowercase
    text = re.sub(r'[^a-z\s]', '', text) # removes any characters that are not alphabetical
    return text.strip().split() # splits it into a list of words

# word2vec usage - to represent each word as a vector
#chsnge
def train_word2vec(corpus, vector_size=100, window=5, min_count=1): # can change this ??
    tokenized_corpus = [preprocess(text) for text in corpus] # preprocess the list of strings (corpus)
    model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count)
    return model

# usage example
# corpus = patients_df["Condition"].tolist() + trials_df["Eligibility"].fillna("").tolist()
# model = train_word2vec(corpus)

# to get weighted average vector (for the sentences?)
def get_weighted_avg_vector(text, model, tfidf_vectorizer, tfidf_matrix, feature_names):
    words = preprocess(text)
    word_vecs = []
    weights = []

    for word in words:
        if word in model.wv and word in feature_names:
            vec = model.wv[word]
            tfidf_weight = tfidf_vectorizer.vocabulary_.get(word, 0)
            word_vecs.append(vec)
            weights.append(tfidf_weight)

    if not word_vecs: # in the case of no words matching - return a vector of 0
        return np.zeros(model.vector_size)

    word_vecs = np.array(word_vecs)
    weights = np.array(weights)
    weighted_avg = np.average(word_vecs, axis=0, weights=weights)
    return weighted_avg

# compute cosine similarity between sentence embeddings?
def compute_similarity_w2v_tfidf(patient_text, trial_text, model, tfidf_vectorizer, tfidf_matrix, feature_names):
    vec1 = get_weighted_avg_vector(patient_text, model, tfidf_vectorizer, tfidf_matrix, feature_names)
    vec2 = get_weighted_avg_vector(trial_text, model, tfidf_vectorizer, tfidf_matrix, feature_names)
    return cosine_similarity([vec1], [vec2])[0][0]






'''
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Option 2: Train your own Word2Vec (example corpus below)
def train_word2vec(corpus_texts, vector_size=100, window=5, min_count=1):
    tokenized_corpus = [simple_preprocess(doc) for doc in corpus_texts]
    return Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count)

# Step 1: Fit TF-IDF on the entire text corpus
def get_tfidf_vectorizer(corpus_texts):
    return TfidfVectorizer().fit(corpus_texts)

# Step 2: Create a TF-IDF-weighted sentence embedding
def sentence_embedding(text, tfidf_vectorizer, w2v_model):
    words = simple_preprocess(text)
    tfidf_weights = tfidf_vectorizer.transform([" ".join(words)])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    word_to_weight = dict(zip(feature_names, tfidf_weights.toarray()[0]))

    # Weighted average
    embeddings = []
    for word in words:
        if word in w2v_model.wv and word in word_to_weight:
            weight = word_to_weight[word]
            embeddings.append(weight * w2v_model.wv[word])

    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        # Fallback to zeros if no embeddings
        return np.zeros(w2v_model.vector_size)

# Step 3: Cosine similarity using sentence embeddings
def compute_similarity(patient_keywords, eligibility_text, tfidf_vectorizer, w2v_model):
    vec1 = sentence_embedding(patient_keywords, tfidf_vectorizer, w2v_model)
    vec2 = sentence_embedding(eligibility_text, tfidf_vectorizer, w2v_model)
    return cosine_similarity([vec1], [vec2])[0][0]
'''