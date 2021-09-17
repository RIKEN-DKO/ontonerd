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
from pathlib import Path
import obonet
from utils import  get_nodes_description
import os
import random
import pickle
#%%
MAX_NUM_WORDS_ENTITY = 2
ONTO_PATH = config.ONTOLOGY_FILES_PATH
ONTOLOGIES = config.WORKING_ONTOLOGIES
# ONTOLOGIES = ['go']
SAVING_DIR = os.path.join('data', 'pem')
Path(SAVING_DIR).mkdir(parents=True, exist_ok=True)

DESC_FILE = os.path.join(SAVING_DIR, 'entity2description.pickle')
# %%

entity2description = get_nodes_description(ONTOLOGIES,ONTO_PATH)

print("Finished processing ontologies")

#%%
print("Saving to file...")

handle = open(DESC_FILE, 'wb')
pickle.dump(entity2description, handle)

print("Finished saving to file")


#%%
# ontology='go'
# obo_file = os.path.join(ONTO_PATH, ontology, ontology+".obo")
# print('Reading ontology:  ', obo_file)
# graph = obonet.read_obo(obo_file)

# # %%
# graph.nodes['GO:0031012']

# # %%

# graph.nodes['GO:0005578']

# # %%

# for qid, data in graph.nodes(data=True):
#     if qid == 'GO:0005578':
#         print('the data',data)

# print('My search ended')
# %%
