from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np

# helper function to be more consistent with text preeprocessing  (lowercase and stopword removed)
def preprocess_text(text):
    return [word for word in simple_preprocess(text, max_len=50) if word not in STOPWORDS]

# trains the word2vec given corpus (the text) by turning words in corpus to vectors
# first preprocess text using simple_preprocess (makes lowercase, cuts non-alphabetical,
# cuts word of length < 2 and > 50, and turns into list)
# * note it currently removes all numbers...
# * also how are we passing the strings - as a document of 1 string, as a list of individual sentences, as a list of
# paragraphs each pertaining to a different category in the data, etccc
def train_word2vec(corpus):  # should we change the window, vector size, or mincount??
    # preprocess the list of strings - and removes any stopwords
    tokenized_corpus = [preprocess_text(text) for text in corpus]
    # train the w2v model with vector size 100, window 5 (# of words on either side of target to consider as context)
    # and the min_count of words of 1 - meaning that even if it only appears once it is added to vocab
    w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1)
    return w2v_model

# corpus = patients_df["Condition"].tolist() + trials_df["Eligibility"].tolist()
# model = train_word2vec(corpus)

'''
a note of some functions available to use on a w2v model:
# w2v_model.wv.most_similar('raynauds') -- computes cosine similarities (returns words closest)
'''

# to give more weight to the less common and more significant terms (i.e., medical terminology)
# trains a tfidf vectorizer with the given text!
def fit_tfidf_vect(text):
    # use preprocess text fucntion for consistent tokenization
    return TfidfVectorizer(tokenizer=preprocess_text(text).fit(text))

# given a piece of text (string), a trained tfidf_vect, and trained word2vec model
# to compare semantic meaning of sentences - returns single vector (weighted average of word vectors) for sentence
# uses tf-idf scores as weights
def weighted_sentence_embedding(text, tfidf_vectorizer, w2v_model):
    # preprocess text - but only < 50 and if not a stopword
    words = [word for word in simple_preprocess(text, max_len=50) if word not in STOPWORDS]

    tfidf_weights = tfidf_vectorizer.transform([" ".join(words)]) # transform into feature vector (sparse matrix)
    tfidf_vocab = tfidf_vectorizer.get_feature_names_out()  # retrieves vocab learned by tfidf model
    # makes a dict (key = word, value = tfidf weight)
    word_to_weight = dict(zip(tfidf_vocab, tfidf_weights.toarray()[0]))

    embeddings = [] # for w2v word vectors
    weights = [] # for tfidf weights

    for word in words:
        if word in w2v_model.wv and word in word_to_weight: # if in w2v vocab & tfidf weight dict
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


# compute cosine similarity between sentence embedding vectors (score between -1 and 1)
# -1 = least similar, 1 = most similar
# computes cosine similarity between patient keywords and eligibility text.
def compute_similarity_w2v(patient_text, trial_text, w2v_model, tfidf_vect):
    vec1 = weighted_sentence_embedding(patient_text, tfidf_vect, w2v_model)
    vec2 = weighted_sentence_embedding(trial_text, tfidf_vect, w2v_model)
    return cosine_similarity([vec1], [vec2])[0][0]

# save the trained word2vec model 
def save_w2v_model(model, path):
    model.save(path)

# load in a previusly saved word2vec model from the disk 
def load_w2v_model(path):
    return Word2Vec.load(path)
