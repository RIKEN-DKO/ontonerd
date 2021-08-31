# %%
# debug
%load_ext autoreload
%autoreload 2
from re import T
import sys

sys.path.append("..")
# import os
# import sys

# %%
import config
import networkx as nx
from pathlib import Path
import obonet
from utils import (get_children_ids, get_synonyms_formatted,
                   preprocess, create_spacy_line, create_ner_sentence)
import os
import copy
import random
import pickle
import spacy
from spacy.lang.en import English
#%%
MAX_NUM_WORDS_ENTITY = 2 
ONTO_PATH = config.ONTOLOGY_FILES_PATH
ONTOLOGIES=config.WORKING_ONTOLOGIES
SAVING_DIR =os.path.join('data','spacy') 
Path(SAVING_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_FILE = os.path.join(SAVING_DIR,'train.pickle')
TEST_FILE = os.path.join(SAVING_DIR,'test.pickle')
VALID_FILE = os.path.join(SAVING_DIR,'valid.pickle')
# %%

lines = []
for ontology in ONTOLOGIES:
    obo_file = os.path.join(ONTO_PATH,ontology,ontology+".obo")
    print('Processing:  ', obo_file)
    graph = obonet.read_obo(obo_file)
    #first when construct a list of documents to search for entity names
    docs = []
    for qid, data in graph.nodes(data=True):

        if 'name' in data:
            name = preprocess(data['name'])
            doc.append(name)

        if 'def' in data:
            definition = preprocess(data['def'])
            doc.append(definition)


        synonyms = get_synonyms_formatted(graph, data)
        docs.extend(synonyms)

    #We search entity name in each document
    for qid, data in graph.nodes(data=True):
        if 'name' in data:
            name = preprocess(data['name'])

        for doc in docs:
            lines.extend(create_ner_sentence(name,doc))
              

print("Finished processing ontologies")
 #%% 
print("Creating the train,test and valid datasets ")
size_dataset = len(lines)
size_train=int(size_dataset*0.90)
# size_test=int(size_dataset*0.10)
random.shuffle(lines)

train = lines[:size_train]
# valid = lines[size_train:size_train+size_valid]
test = lines[size_train:]
dataset = [train,test]
#%%
print("Saving to file...")
_files = [TRAIN_FILE,TEST_FILE]
for i,jfile in enumerate(_files):
    handle = open(jfile,'wb')
    pickle.dump(dataset[i],handle)

print("Finished saving to file")


#%%
nlp = English()
tokenizer = nlp.tokenizer
# %%
create_ner_sentence('bien chula!','esa chula chica super esta bien chula!',tokenizer)
# %%
doc = tokenizer("rosita fresita")
# %%
tokens = [token.text for token in tokenizer("rosita fresita")]
# %%
tokens
# %%
