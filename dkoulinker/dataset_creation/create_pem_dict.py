#%%
%load_ext autoreload
%autoreload 2
from re import T
import sys

sys.path.append("..")
# %%
from utils import (create_pem_dictionary)
import config
import os
import obonet
import networkx as nx
from pathlib import Path
import pickle
# %%
ONTO_PATH = config.ONTOLOGY_FILES_PATH
# ONTOLOGIES = config.WORKING_ONTOLOGIES
ONTOLOGIES=['go']
SAVING_DIR = os.path.join('data', 'pem')
Path(SAVING_DIR).mkdir(parents=True, exist_ok=True)


# %%
pem,mention_freq = create_pem_dictionary(
    ONTOLOGIES, ONTO_PATH)
# %%
PEM_FILE = os.path.join(SAVING_DIR, 'pem.pickle')
handle = open(PEM_FILE,'wb')
pickle.dump(pem,handle)
MENTION_FILE = os.path.join(SAVING_DIR, 'mention_freq.pickle')
handle = open(MENTION_FILE,'wb')
pickle.dump(mention_freq,handle)

#%%
from utils import get_children_ids
ontology = 'go'
obo_file = os.path.join(ONTO_PATH, ontology, ontology+".obo")
print('Reading ontology:  ', obo_file)
graph = obonet.read_obo(obo_file)

# %%
get_children_ids('GO:0000003', graph)

# %%
