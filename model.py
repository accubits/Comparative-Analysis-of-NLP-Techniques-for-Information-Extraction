# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:44:51 2017

@author: Vishnu
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from smalltalk import SmallTalk
import distance
from nltk.corpus import wordnet
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import gensim
from gensim.models.doc2vec import LabeledSentence
from data import Data
from Preprocess import preprocess
import os
import glob

os.chdir('path/to/your/data folder/')

path = glob.glob("*.pdf")

nlp = spacy.load('en')
st = SmallTalk(nlp)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
svd = TruncatedSVD(n_components=100)

# =============================================================================
# TF-IDF to identify most similar documents(page here) and retuns the index of the document
# =============================================================================

def indexTFIDF(doc):
    tfidf_matrix = tfidf_vectorizer.fit_transform(doc)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    sim = similarity[0].tolist()
    refined = sorted(range(len(sim)), key=lambda i: sim[i])[-3:]
    sim = sorted(((value, index) for index, value in enumerate(sim)), reverse=True)[:3]
    refined = [i[1] for i in sim if i[0] > 0.07]
    refined.remove(0)
    refined[:] = [i-1 for i in refined]
    return refined   

# =============================================================================
# Latent semantic Indexing to identify most similar documents(page here) and retuns the index of the document
# =============================================================================

def indexLSA(doc):
    tfidf_matrix = tfidf_vectorizer.fit_transform(doc)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    similarity = cosine_similarity(lsa_matrix[0:1], lsa_matrix)
    sim = similarity[0].tolist()
    refined = sorted(range(len(sim)), key=lambda i: sim[i])[-3:]
    sim = sorted(((value, index) for index, value in enumerate(sim)), reverse=True)[:3]
    refined = [i[1] for i in sim if i[0] > 0.1]
    refined.remove(0)
    refined[:] = [i-1 for i in refined]
    return refined  

'''
One can use either tf-idf or lsa for document retrieval whichever is more accurate to their data

'''

# =============================================================================
# Module for query expansion and focus detection
# =============================================================================

def QueryProcess(query):
    data = Data(path)
    cleaned_query = preprocess(query)
    txtn = nlp(cleaned_query)
    txtp = nlp(query)
    np = [np.text for np in txtn.noun_chunks]
    ner = [ent.text for ent in txtp.ents]
    tokens = cleaned_query.split()
    keywords = [token.text for token in txtn if token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'NOUN' or token.pos_ == 'PROPN']
    synonyms = list(set([l.name() for i in keywords for syn in wordnet.synsets(i) for l in syn.lemmas()]))
    synonyms = [i for i in synonyms if '_' not in i]
    new_text = tokens + synonyms + np + [i.lower() for i in ner] + [query]
    final_query = ' '.join(new_text)
    cleaned_data = [''.join(preprocess(i) + i) for i in data]
    doc = [final_query] + cleaned_data
    indx = indexLSA(doc) # or indexTFIDF(doc)
    return final_query, indx, data

# =============================================================================
# Spacy similarity to extract answers from most similar documents (10 sentences)
# =============================================================================

def FinalIndexSpacy(final_query, desc):
    desc = [preprocess(i) for i in desc if i != ''  and len(i.split()) > 10]
    list_indx = []
    for indx, i in enumerate(desc):
        dict_indx = {}
        dict_indx['index'] = indx
        dict_indx['similarity'] = nlp(final_query).similarity(nlp(i))
        if dict_indx['similarity'] > .6:
            list_indx.append(dict_indx)
    refined = sorted(range(len(list_indx)), key=lambda index: list_indx[index]['similarity'], reverse=True)[:10]
    return refined

# =============================================================================
# Jaccard distance to extract answers from most similar documents (10 sentences)
# =============================================================================

def FinalIndexJaccard(final_query, desc):
    desc = [preprocess(i) for i in desc if i != ''  and len(i.split()) > 10]
    list_indx = []
    for indx, i in enumerate(desc):
        dict_indx = {}
        dict_indx['index'] = indx
        dict_indx['similarity'] = 1 - distance.jaccard(final_query, i)
        if dict_indx['similarity'] > .5:
            list_indx.append(dict_indx)
    refined = sorted(range(len(list_indx)), key=lambda index: list_indx[index]['similarity'], reverse=True)[:10]
    return refined

# =============================================================================
# doc2vec similarity to extract answers from most similar documents (10 sentences)
# =============================================================================

def FinalIndexDoc2Vec(final_query, desc):
    desc = [preprocess(i) for i in desc if i != ''  and len(i.split()) > 10]
    sentences= []
    for item_no, line in enumerate(desc):
        sentences.append(LabeledSentence(line,[item_no]))
    dm = 1
    size = 300
    context_window = 50
    seed = 42
    min_count = 1
    alpha = 0.5
    max_iter = 200
    model = gensim.models.doc2vec.Doc2Vec(documents = sentences, dm = dm, 
                                          alpha = alpha, seed = seed,
                                          min_count = min_count,
                                          max_vocab_size = None,
                                          window = context_window,
                                          size = size, sample = 1e-4,
                                          negative = 5, iter = max_iter)
    tokens = final_query.split()
    new_vector = model.infer_vector(tokens)
    sims = model.docvecs.most_similar([new_vector], topn = 10)
    refined = [i[0] for i in sims if i[1] > 0]
    return refined

# =============================================================================
# module to join most similar sentences and return the answer
# =============================================================================
    
def FinalAnswer(query):
    final_query, indx, data = QueryProcess(query)
    desc_answer = [data[i] for i in indx]
    desc_answer = ' '.join(desc_answer)
    if desc_answer != '':
        desc = re.split('[.]', desc_answer)
        desc = [i for i in desc if i != '' and len(i.split()) > 10]
        final_indexJaccard = FinalIndexJaccard(final_query, desc)
        final_answerJaccard = '.'.join(desc[i] for i in final_indexJaccard) + '.'
        final_indexSpacy = FinalIndexSpacy(final_query, desc)
        final_answerSpacy = '.'.join(desc[i] for i in final_indexSpacy) + '.'
        final_indexDoc2Vec = FinalIndexDoc2Vec(final_query, desc)
        final_answerDoc2Vec = '.'.join(desc[i] for i in final_indexDoc2Vec) + '.'
        return final_answerJaccard, final_answerSpacy, final_answerDoc2Vec
    else:
        return st.get_reply(query), st.get_reply(query), st.get_reply(query)
