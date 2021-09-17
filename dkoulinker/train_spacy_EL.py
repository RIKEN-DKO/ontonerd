# %%
#debug
# %load_ext autoreload
# %autoreload 2
# from re import T

# sys.path.append(".")

# %%
import sys
import spacy
import time
from spacy.kb import KnowledgeBase
from spacy.ml.models import load_kb
from spacy.training import Example
import config
from config import SPACY_DATASET
from pathlib import Path
import obonet
import os
import pickle
import random
from dataset_creation.utils import get_synonyms_formatted
from spacy.util import minibatch, compounding

#%%
spacy.require_gpu()

NUM_ITERATIONS = 500
if len(sys.argv)>1:
    NUM_ITERATIONS = int(sys.argv[1])
if len(sys.argv)>2:
    mode = sys.argv[2]
else :
    mode ='new'

print('Training Spacy dataset for {}'.format(NUM_ITERATIONS))
output_dir = Path.cwd()/"data"/"spacy"
print('Data dir: ', output_dir)
# %%
#Loading dataset
# kb = KnowledgeBase.from_disk(output_dir / "my_kb")
if mode == 'continue':
    nlp = spacy.load(output_dir / "my_nlp")
else:
    nlp = spacy.load("en_core_web_lg")

handle = open(os.path.join(config.SPACY_DATASET,'train.pickle'),'rb')
train_dataset = pickle.load(handle)
handle = open(os.path.join(config.SPACY_DATASET,'test.pickle'),'rb')
test_dataset = pickle.load(handle)
# %%
#Creating the spacy 3 dataset

TRAIN_EXAMPLES = []
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")
sentencizer = nlp.get_pipe("sentencizer")
for text, annotation in train_dataset:
    # print(i)
    example = Example.from_dict(nlp.make_doc(text), annotation)
    example.reference = sentencizer(example.reference)
    TRAIN_EXAMPLES.append(example)

print('[FINISHED] Creating examples')
# %%
#create a new Entity Linking component and add it to the pipeline.
if mode == 'continue':
    entity_linker = nlp.get_pipe("entity_linker")
else:
    entity_linker = nlp.add_pipe("entity_linker", config={
                             "incl_prior": False}, last=True)
entity_linker.initialize(get_examples=lambda: TRAIN_EXAMPLES,
                         kb_loader=load_kb(output_dir / "my_kb"))

#$$
# %%

start = time.time()
# train only the entity_linker
with nlp.select_pipes(enable=["entity_linker"]):
    optimizer = nlp.resume_training()
    for itn in range(NUM_ITERATIONS):   # 500 iterations takes about a minute to train
        random.shuffle(TRAIN_EXAMPLES)
        batches = minibatch(TRAIN_EXAMPLES, size=compounding(
            4.0, 32.0, 1.001))  # increasing batch sizes
        losses = {}
        for batch in batches:
            nlp.update(
                batch,
                drop=0.2,      # prevent overfitting
                losses=losses,
                sgd=optimizer,
            )
        if itn % 50 == 0:
            print(itn, "Losses", losses)   # print the training loss
print(itn, "Losses", losses)

end = time.time()
print('Training took :',end - start)
# %%
nlp.to_disk(output_dir / "my_nlp")

# %%
