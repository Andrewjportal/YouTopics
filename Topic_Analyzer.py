


from pytube import YouTube
from pytube import Playlist
import pandas as pd
import numpy as np

from xml.etree import ElementTree as ET
import bleach
import re

import gensim
from gensim import corpora, models, similarities, matutils
from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize
from gensim.parsing.preprocessing import remove_stopwords

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# In[56]:


from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))


# Helper Functions

# In[57]:


def get_transcript(url):
    path=url
    try:
        yt = YouTube(path)
    except ValueError:
        print('cannot find video')
    caption = yt.captions.get_by_language_code('en')
    try:
        xml=caption.xml_captions
    except AttributeError:
        print('no captions or transcripts')

    root = ET.fromstring(xml)
    #gets the transcripts
    doc=''
    for child in root:
        try:
            doc=doc+" "+(child.text)
        except TypeError:
            pass
    return doc.replace('\n',' ')


# In[58]:


def make_corpus(url_list):
    corpus=[]
    for url in url_list:
        x=bleach.clean(get_transcript(url), tags=[], attributes={}, styles=[], strip=True)
        y=re.sub(r'&#39;', '', x)
        z=re.sub(r'\[inaudible]', '', y)
        doc=re.sub(r'\[Music]', '', z)
       
        corpus.append(doc)
    
    return corpus


# In[59]:


def oov(keys):
    keys2 = []
    for key in keys:
        if key in model.vocab:
            keys2.append(key)
    x = len(keys)-len(keys2)
    y = x*(sum(list(map(model.word_vec, keys2)))/len(keys2))
    vector = sum(list(map(model.word_vec, keys2)))+y
    return vector


# In[60]:


def get_topic_space(url_list):
    docs=make_corpus(url_list)
    vectors_list=[]
    for i in range(len(docs)):
        clean_doc=remove_stopwords(docs[i])
        keys=keywords(clean_doc, words=5,pos_filter=('NN','NNS','NNPS','NNP',),lemmatize=True, split=True)
    
        try:
            vector=sum(list(map(model.word_vec,keys)))
        except KeyError:
            vector=oov(keys)
                
        vectors_list.append(vector)
   
    return (sum(vectors_list)/len(docs))


# In[75]:


def keywords_to_vect(keys):

    try:
        vector=sum(list(map(model.word_vec,keys)))
    except KeyError:
        vector=oov(keys)
    return vector


# In[77]:


def topic_analyze(url,topic_dict):
    doc=make_corpus([url])
    x=get_topic_space([url])
    topic_dict_vectors={}
    analysis={}

    for key, value in topic_dict.items():
        topic_dict_vectors[f'{key}']=keywords_to_vect(value)
    for key, value in topic_dict_vectors.items():
        analysis[f'{key}'] = str(round(cos_sim(x,value),3))
    
    clean_doc=remove_stopwords(doc[0])
    keys=keywords(clean_doc,words=5,pos_filter=('NN','NNS','NNPS','NNP',),lemmatize=True,split=True)

    summary=summarize(doc[0],word_count=50, split=True)
    return {'Keys':keys, 'Summary':summary,'Analysis':analysis}


# In[10]:


from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

glove_file = datapath('/Users/andrewportal/Downloads/glove/glove.6B.100d.txt')
tmp_file = get_tmpfile("glove_word2vec.txt")

# call glove2word2vec script
# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)    





