# %%
#debug
%load_ext autoreload
%autoreload 2
from re import T
import sys

sys.path.append(".")
import os
import sys

# %%
import spacy
from spacy.kb import KnowledgeBase
from spacy.training import Example
import config
from config import SPACY_DATASET
from pathlib import Path
import obonet
import os
import pickle
from dataset_creation.utils import get_synonyms_formatted

#%%
ONTO_PATH = config.ONTOLOGY_FILES_PATH
ONTOLOGIES = ['go']
nlp = spacy.load("en_core_web_lg")
#%%
def load_entities():

    for ontology in ONTOLOGIES:
        obo_file = os.path.join(ONTO_PATH, ontology, ontology+".obo")
        print('Processing:  ', obo_file)
        graph = obonet.read_obo(obo_file)

        names = dict()
        descriptions = dict()
        synonyms = dict()
        for id, data in graph.nodes(data=True):
            if 'name' in data:
                names[id] = data['name']
            else:
                continue

            if 'def' in data:
                descriptions[id] = data['def']
            else:
                continue

            if 'synonym' in data:
                synonyms[id] = get_synonyms_formatted(graph,data)

    return names, descriptions, synonyms


# %%
print('Creating the Knowledge base from ontology files...')
name_dict, desc_dict,synonyms_dict = load_entities()
for QID in name_dict.keys():
    print(f"{QID}, name={name_dict[QID]}, desc={desc_dict[QID]},syn={synonyms_dict[QID]}")
    break
print('[FINISHED]Creating the Knowledge base from ontology files...')

# %%
#TODO nlp.vocab create own?
kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)

# To add each record to the knowledge base, we encode its description using the built-in 
# word vectors of our `nlp` model. The `vector` attribute of a document is the average 
# of its token vectors. We also need to provide a frequency, which is a raw count of how many
#  times a certain entity appears in an annotated corpus. Here 
#  we're not using these frequencies, so we're setting them to an arbitrary value.
print('Adding vectors to the Knowledge base')
for qid, desc in desc_dict.items():
    desc_doc = nlp(desc)
    desc_enc = desc_doc.vector  # Embedding
    kb.add_entity(entity=qid, entity_vector=desc_enc,
                  freq=342)   # 342 is an arbitrary value he
print('[FINISHED]Adding vectors to the Knowledge base')

# %%
#Add names as alias
print('Adding Alias(names and synonims) to the Knowledge base')
for qid, name in name_dict.items():
    # 100% prior probability P(entity|alias)
    kb.add_alias(alias=name, entities=[qid], probabilities=[1])

    if qid in synonyms_dict:
        synonyms = synonyms_dict[qid]
        for syn in synonyms:
            kb.add_alias(alias=syn, entities=[qid], probabilities=[1])


print('[FINISHED]Adding  Alias to the Knowledge base')


# %%
# print(f"Entities in the KB: {kb.get_entity_strings()}")
# print(f"Aliases in the KB: {kb.get_alias_strings()}")
print(f"Candidates for 'reproduction': {[c.entity_ for c in kb.get_alias_candidates('reproduction')]}")

# %%
#Save Knowledge base to disk
output_dir = Path.cwd()/"data"/"spacy"
Path(output_dir).mkdir(parents=True, exist_ok=True)
nlp.to_disk(output_dir / "my_nlp")
kb.to_disk(output_dir / "my_kb")
# %%
#Loading dataset
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
for i,(text, annotation) in enumerate(train_dataset):
    print(i)
    example = Example.from_dict(nlp.make_doc(text), annotation)
    example.reference = sentencizer(example.reference)
    TRAIN_EXAMPLES.append(example)

# %%
