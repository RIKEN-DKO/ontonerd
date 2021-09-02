#%%
%load_ext autoreload
%autoreload 2
from logging import debug
from re import T
import sys
#%%
from flair.data import Corpus
from flair.datasets import ColumnCorpus


# %%
# define columns
columns = {0: 'text', 1: 'ner'}  # directory where the data resides
data_folder = '../data/ner/'  # initializing the corpus
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')
print('Finished load datasets')
# %%
print(len(corpus.train))
# %%
print('asd')
# %%
