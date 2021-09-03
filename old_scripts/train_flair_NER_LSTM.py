#%%
# %load_ext autoreload
# %autoreload 2
# from logging import debug
# from re import T
# import sys
#%%
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from typing import List
from flair.trainers import ModelTrainer
# %%
# define columns

EPOCHS=150
#%%
columns = {0: 'text', 1: 'ner'}  # directory where the data resides
data_folder = 'data/ner/'  # initializing the corpus
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='valid.txt')
print('Finished load datasets')
# %%
print(len(corpus.train))
print(corpus.train[0].to_tagged_string('ner'))
# %%
# tag to predict
tag_type = 'ner'  # make tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
# %%
embedding_types = [
    WordEmbeddings('glove'),
    ## other embeddings
]
embeddings: StackedEmbeddings = StackedEmbeddings(
    embeddings=embedding_types)

# %%
tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                       embeddings=embeddings,
                                       tag_dictionary=tag_dictionary,
                                       tag_type=tag_type,
                                       use_crf=True)
print(tagger)

#%%
trainer: ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train('resources/taggers/example-ner',
            learning_rate=0.1,
            mini_batch_size=32,
            max_epochs=EPOCHS) #150

# %%
