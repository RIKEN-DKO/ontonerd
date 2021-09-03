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
from pathlib import Path
# %%
ONTO_PATH = config.ONTOLOGY_FILES_PATH
# ONTOLOGIES = config.WORKING_ONTOLOGIES
ONTOLOGIES=['go']
SAVING_DIR = os.path.join('data', 'ner_bert')
Path(SAVING_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_FILE = os.path.join(SAVING_DIR, 'train.txt')

# %%
pem,mention_freq = create_pem_dictionary(
    ONTOLOGIES, ONTO_PATH)

# %%
