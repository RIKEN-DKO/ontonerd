#%%
import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
from pathlib import Path
import pickle
import os
#%%
output_dir = os.path.join('data', 'spacy')
nlp = spacy.blank("en")  # load a new spacy model
db = DocBin()  # create a DocBin object

handle = open(output_dir+"/train.pickle",'rb')
TRAIN_DATA = pickle.load(handle) 

handle = open(output_dir+"/test.pickle", 'rb')
TEST_DATA = pickle.load(handle)

#%%
for text, annot in tqdm(TRAIN_DATA):  # data in previous format
    doc = nlp.make_doc(text)  # create doc object from text
    ents = []
    for start, end, label in annot["entities"]:  # add character indexes
        span = doc.char_span(start, end, label=label,
                             alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents  # label the text with the ents
    db.add(doc)

db.to_disk(output_dir+"/train.spacy")  # save the docbin object

#TODO Ugly but Im lazy
for text, annot in tqdm(TEST_DATA):  # data in previous format
    doc = nlp.make_doc(text)  # create doc object from text
    ents = []
    for start, end, label in annot["entities"]:  # add character indexes
        span = doc.char_span(start, end, label=label,
                             alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents  # label the text with the ents
    db.add(doc)

db.to_disk(output_dir+"/test.spacy")  # save the docbin object
