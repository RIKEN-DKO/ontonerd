# %%
# debug
%load_ext autoreload
%autoreload 2
from logging import debug
from re import T
import sys

sys.path.append("..")
import os
import sys

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
#bert SUPPORT 512, but bioBERT was finetuned with 128
MAX_NUM_TOKEN_SEN = 128
ONTO_PATH = config.ONTOLOGY_FILES_PATH
ONTOLOGIES=config.WORKING_ONTOLOGIES
ONTOLOGIES=['go']
SAVING_DIR =os.path.join('data','ner_bert') 
Path(SAVING_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_FILE = os.path.join(SAVING_DIR,'train.txt')
TRAINDEV_FILE = os.path.join(SAVING_DIR,'train_dev.txt')
TEST_FILE = os.path.join(SAVING_DIR,'test.txt')
DEVEL_FILE = os.path.join(SAVING_DIR,'devel.txt')

# tokenizer = nlp.tokenizer
# %%


sentences = create_ner_sentences_children(
    ONTOLOGIES, 
    ONTO_PATH, 
    MAX_NUM_WORDS_ENTITY=5,
    char_space=' ')
       

#%% 
#
random.shuffle(sentences)
print("Creating the train,test and devel datasets ")
size_dataset = len(sentences)
size_train=int(size_dataset*0.45)
size_train_dev=int(size_dataset*0.45)
size_test=int(size_dataset*0.05)
size_devel=int(size_dataset*0.05)
# random.shuffle(lines)

_train = sentences[:size_train]
_train_dev = sentences[size_train:size_train+size_train_dev]
_test = sentences[size_train+size_train_dev : size_train+size_train_dev+size_test]
_devel = sentences[size_train+size_train_dev+size_test:]
_dataset = [_train, _train_dev, _test,_devel]
#Creating lines
# lines = []
train,train_dev,test,devel=[],[],[],[]
dataset = [train,train_dev,test,devel]

for i,_sentences in enumerate(_dataset):
    for sentence in _sentences:
        if sentence[-2] != '. O':
            sentence[-2] = '. O'
            sentence[-1] = ' '
        if len(sentence) > MAX_NUM_TOKEN_SEN:
            print('Sentence was truncated',sentence)
            #TODO add [.  O]
            dataset[i].extend(sentence[:MAX_NUM_TOKEN_SEN - 2] + ['. O',' '])
        else:
            dataset[i].extend(sentence)

#%%
print("Saving to file...")
_files = [TRAIN_FILE, TRAINDEV_FILE,TEST_FILE,DEVEL_FILE]
for i,jfile in enumerate(_files):
    handle = open(jfile,'w')
    for item in dataset[i]:
        handle.write("{}\n".format(item))

handle.close()
print("Finished saving to file")

# %%
nlp = English()
nlp.add_pipe("sentencizer")

# %%
child_def = 'This is one damn sentence. This is other fockin sentence'
def_sents = nlp(child_def)
def_sents = [sen.text
             for sen in list(def_sents.sents)]

# %%
