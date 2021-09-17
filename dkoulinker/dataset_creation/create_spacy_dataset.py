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
from utils import get_children_ids, get_synonyms_formatted,preprocess,create_spacy_line
import os
import copy
import random
import pickle
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
    for qid, data in graph.nodes(data=True):
        
        if 'name' in data:
            name = preprocess(data['name'])
        else:
            continue
        
        if 'def' in data:
            definition = preprocess(data['def'])
        else:
            continue
        name_len = len(name) 
        words_name = name.split(' ')
        if len(words_name)> MAX_NUM_WORDS_ENTITY:
            continue
        
        #find children
        children = list(set(get_children_ids(qid,graph)))
        for child_id in children:
            if 'name' in graph.nodes[child_id]:
                child_name = preprocess(graph.nodes[child_id]['name'])
            else:
                continue

            if 'def' in graph.nodes[child_id]:
                child_def = preprocess(graph.nodes[child_id]['def'])
            else:
                continue
            #Searching for ocurrences of the parent name into the names
            # and definitions of children            
            #search in name
            line = create_spacy_line(child_name,name,qid,insert_space=True)
            if line is not None:
                lines.append(line)
            #def
            line = create_spacy_line(child_def, name, qid, insert_space=True)
            if line is not None:
                lines.append(line)
            #TODO add synonyns
        # if len(lines)>2:
        #     break
              

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


# #%%
# a = 'The offsets of theannotationsfor `links` could not be aligned to token boundaries.'
# strs = 'annotations'

# # %%
# s = create_spacy_line(a, strs, 111, insert_space=True)
# print(s)
# b = s[0]
# # %%
