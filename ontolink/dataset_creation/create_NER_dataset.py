# %%
# debug
# %load_ext autoreload
# %autoreload 2
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
MIN_LEN_SENTENCE = 3
MAX_NUM_SENTENCE_PERDOC = 1000
ONTO_PATH = config.ONTOLOGY_FILES_PATH
ONTOLOGIES=config.WORKING_ONTOLOGIES
# ONTOLOGIES=['go']
SAVING_DIR =os.path.join('data','ner') 
Path(SAVING_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_FILE = os.path.join(SAVING_DIR,'train.txt')
TEST_FILE = os.path.join(SAVING_DIR,'test.txt')
VALID_FILE = os.path.join(SAVING_DIR,'valid.txt')
nlp = English()
nlp.add_pipe("sentencizer")
# tokenizer = nlp.tokenizer
# %%

lines = []
sentences = []
for ontology in ONTOLOGIES:
    obo_file = os.path.join(ONTO_PATH,ontology,ontology+".obo")
    print('Processing:  ', obo_file)
    graph = obonet.read_obo(obo_file)
    #first when construct a list of documents to search for entity names
    docs = []
    for qid, data in graph.nodes(data=True):
        for_process = []
        if 'name' in data:
            name = preprocess(data['name'])
            for_process.append(name)
            # docs.append(name)

        if 'def' in data:
            definition = preprocess(data['def'])
            # docs.append(definition)
            for_process.append(definition)


        synonyms = get_synonyms_formatted(graph, data)
        for_process.extend(synonyms)
        # docs.extend(synonyms)

        for text in for_process:
            doc = nlp(text)
            sentence_tokens = [[token.text for token in sent] for sent in doc.sents]

            for sentence in sentence_tokens:
                if len(sentence) >= MIN_LEN_SENTENCE:
                    docs.append(sentence)

    #We search entity name in each document
    for qid, data in graph.nodes(data=True):
        if 'name' in data:
            name = preprocess(data['name'])

        num_sentences=0
        for doc in docs:
            if num_sentences > MAX_NUM_SENTENCE_PERDOC:
                break
            sentence = create_ner_sentence(name, doc, nlp)
            if len(sentence)>0:
                sentences.append(sentence)
                num_sentences+=1

       

print("Finished processing ontologies")
#%% 
#
random.shuffle(sentences)
#Creating lines
for sentence in sentences:
    lines.extend(sentence)

print("Creating the train,test and valid datasets ")
size_dataset = len(lines)
size_train=int(size_dataset*0.90)
size_valid=int(size_dataset*0.05)
size_test=int(size_dataset*0.05)
# random.shuffle(lines)

train = lines[:size_train]
valid = lines[size_train:size_train+size_valid]
test = lines[size_train+size_valid:]
dataset = [train,valid,test]
#%%
print("Saving to file...")
_files = [TRAIN_FILE,VALID_FILE,TEST_FILE]
for i,jfile in enumerate(_files):
    handle = open(jfile,'w')
    for item in dataset[i]:
        handle.write("{}\n".format(item))

handle.close()
print("Finished saving to file")
