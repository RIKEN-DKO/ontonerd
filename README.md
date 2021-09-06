## Creating blink dataset:

```
PYTHONPATH=ontolink python ontolink/dataset_creation/create_blink_dataset.py
```
# SPACY
## Creating spacy dataset:

```
PYTHONPATH=ontolink python ontolink/dataset_creation/create_spacy_dataset.py
```

## Spacy 3 dataset

```
PYTHONPATH=ontolink python ontolink/dataset_creation/create_spacy3_dataset.py
``` 

Fill the config file from:
https://spacy.io/usage/training#quickstart



```
python -m spacy init fill-config ontolink/dataset_creation/base_config.cfg ontolink/dataset_creation/config.cfg
```


```
python -m spacy train ontolink/dataset_creation/config.cfg --output data/spacy --paths.train data/spacy/train.spacy --paths.dev data/spacy/test.spacy --gpu-id 0
```


## Creating spacy KB:

```
PYTHONPATH=ontolink python ontolink/create_spacy_KB.py
```

## Training spacy model

```
PYTHONPATH=ontolink python ontolink/train_spacy_EL.py
```

# NER

## Creating NER dataset
```
PYTHONPATH=ontolink python ontolink/dataset_creation/create_NER_dataset_flair.py
```

```
PYTHONPATH=ontolink python ontolink/dataset_creation/create_NER_dataset_BERT.py
```
## train 

```
PYTHONPATH=ontolink python ontolink/train_flair_NER_transformers.py
```

```
PYTHONPATH=ontolink python ontolink/train_flair_NER_LSTM.py
```

The dataset created for BERT is being trained on repo ../merge_NER_data

# P(e|m)

creating dictionaries

```
PYTHONPATH=ontolink python ontolink/dataset_creation/create_pem_dict.py
```