# %%
%load_ext autoreload
%autoreload 2
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
from utils import get_children_ids, get_synonyms_formatted
import os
import copy
import random
#%%
MAX_NUM_WORDS_ENTITY = 2 
ONTO_PATH = config.ONTOLOGY_FILES_PATH
ONTOLOGIES=['go'] 
SAVING_DIR =os.path.join('data','blink') 
Path(SAVING_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_FILE = os.path.join(SAVING_DIR,'train.jsonl')
TEST_FILE = os.path.join(SAVING_DIR,'test.jsonl')
VALID_FILE = os.path.join(SAVING_DIR,'valid.jsonl')
# %%

lines = []
for ontology in ONTOLOGIES:
    obo_file = os.path.join(ONTO_PATH,ontology,ontology+".obo")
    print('Processing:  ', obo_file)
    graph = obonet.read_obo(obo_file)
    for id, data in graph.nodes(data=True):
        
        if 'name' in data:
            name = data['name'].lower()
        else:
            continue

        # if 'def' in data:
        #     definition = data['def']
        # else:
        #     definition = ''

        # try:
        #     synonyms = ' '.join(get_synonyms_formatted(graph, data))
        # except KeyError:
        #     synonyms = ''
        words_name = name.split('')
        if len(words_name)> MAX_NUM_WORDS_ENTITY:
            continue:

        #find parents
        parents = get_children_ids(id,graph)
        if len(parents) < 1:
            continue
        # print(name)
        for parent_id in parents:
            jsonline = {}
            jsonline["world"]=ontology
            mention = graph.nodes[parent_id]['name']
            if 'def' in graph.nodes[parent_id]:
                label = graph.nodes[parent_id]['def']
            else:
                continue

            jsonline["context_left"]=context
            jsonline["context_right"]=context
            jsonline["mention"]=mention
            jsonline["label"]=label
            jsonline["label_id"]=int(label_id.split(':')[1])
            jsonline["label_title"]=label_title


            #Create line
            lines.append(jsonline)    
        # print(data)
        # break
print("Finished processing ontologies")
#%% 
print("Creating the train,test and valid datasets ")
size_dataset = len(lines)
size_train=int(size_dataset*0.80)
size_valid=int(size_dataset*0.10)
random.shuffle(lines)

train = lines[:size_train]
valid = lines[size_train:size_train+size_valid]
test = lines[size_train+size_valid:]
dataset = [train,valid,test]
#%%
print("Saving to file...")
json_files = [TRAIN_FILE,VALID_FILE,TEST_FILE]
for i,jfile in enumerate(json_files):
    writer = jsonlines.open(jfile, mode='w')
    writer.write_all(dataset[i])
    writer.close()

print("Finished saving to file")


#%%
ontology='go'
obo_file = os.path.join(ONTO_PATH,ontology,ontology+".obo")
print('Processing:  ', obo_file)
graph = obonet.read_obo(obo_file)

# %%
children = get_children_ids('GO:0016049',graph)
# %%
id = 'GO:0016049'
for child, parent, key in graph.in_edges(id, keys=True):
    print(f'• {parent} ⟵ {key} ⟵ {child}')
# %%
