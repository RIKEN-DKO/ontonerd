from flair.datasets import CONLL_03
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ColumnCorpus
import torch
from torch.optim.lr_scheduler import OneCycleLR

# 1. get the corpus
# corpus = CONLL_03()
columns = {0: 'text', 1: 'ner'}  # directory where the data resides
data_folder = 'data/ner/'  # initializing the corpus
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='valid.txt')
print('Finished load datasets')
print(corpus)

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize fine-tuneable transformer embeddings WITH document context
embeddings = TransformerWordEmbeddings(
    model='xlm-roberta-large',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)

# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type='ner',
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
)

# 6. initialize trainer with AdamW optimizer
trainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)

# 7. run training with XLM parameters (20 epochs, small LR, one-cycle learning rate scheduling)
trainer.train('resources/taggers/sota-ner-flert',
              learning_rate=5.0e-5, #5.0e-6
              mini_batch_size=8,
              # remove this parameter to speed up computation if you have a big GPU
              #mini_batch_chunk_size=1,
              max_epochs=20,  # 10 is also good
              scheduler=OneCycleLR,
              embeddings_storage_mode='gpu', #none,cpu
              weight_decay=0.,
              checkpoint=True
              )
