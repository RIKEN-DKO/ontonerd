# %%
# debug
# %load_ext autoreload
# %autoreload 2
# from logging import debug
# from re import T
# import sys

# sys.path.append("..")
# import os
# import sys

# %%
import config
import networkx as nx
from pathlib import Path
import obonet
from utils import (create_ner_sentences_children)
import os
import copy
import random
import pickle
import spacy
from spacy.lang.en import English
#%%
#bert SUPPORT 512
MAX_NUM_TOKEN_SEN = 512
ONTO_PATH = config.ONTOLOGY_FILES_PATH
ONTOLOGIES=config.WORKING_ONTOLOGIES
# ONTOLOGIES=['go']
SAVING_DIR =os.path.join('data','ner') 
Path(SAVING_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_FILE = os.path.join(SAVING_DIR,'train.txt')
TEST_FILE = os.path.join(SAVING_DIR,'test.txt')
VALID_FILE = os.path.join(SAVING_DIR,'valid.txt')

# tokenizer = nlp.tokenizer
# %%


sentences = create_ner_sentences_children(  ONTOLOGIES, ONTO_PATH, MAX_NUM_WORDS_ENTITY=5)
       

#%% 
#
random.shuffle(sentences)
print("Creating the train,test and valid datasets ")
size_dataset = len(sentences)
size_train=int(size_dataset*0.90)
size_valid=int(size_dataset*0.05)
size_test=int(size_dataset*0.05)
# random.shuffle(lines)

_train = sentences[:size_train]
_valid = sentences[size_train:size_train+size_valid]
_test = sentences[size_train+size_valid:]
_dataset = [_train,_valid,_test]
#Creating lines
# lines = []
train,valid,test=[],[],[]
dataset = [train,valid,test]

for i,_sentences in enumerate(_dataset):
    for sentence in _sentences:
        if len(sentence) > MAX_NUM_TOKEN_SEN:
            dataset[i].extend(sentence[:MAX_NUM_TOKEN_SEN])
        else:
            dataset[i].extend(sentence)

#%%
print("Saving to file...")
_files = [TRAIN_FILE,VALID_FILE,TEST_FILE]
for i,jfile in enumerate(_files):
    handle = open(jfile,'w')
    for item in dataset[i]:
        handle.write("{}\n".format(item))

handle.close()
print("Finished saving to file")

# %%
