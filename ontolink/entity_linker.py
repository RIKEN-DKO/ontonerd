from typing import List


from dataset_creation.utils import get_clean_tokens,preprocess
from nltk.corpus import stopwords
from spacy.lang.en import English
from flair.data import Sentence
import time
import numpy as np
from entity_ranking import EntityRanking

class EntityLinker:
    """
    TODO add doc
    """
    def __init__(self, 
    mention2pem, 
    # entity2description, 
    # mention_freq,
    # collection_size_terms,
    ranking_strategy: EntityRanking,
    ner_model=None,
    ner_model_type='flair'):

        self.mention2pem = mention2pem
        # self.entity2description = entity2description
        # self.entities_list = list(entity2description.keys())
        # self.entities_list_np = np.asarray(self.entities_list)
        # print(self.entities_list_np.shape)
        # self.mention_freq = mention_freq
        #TODO is necesary?
        # self.collection_size_terms = collection_size_terms
        self.ranking_strategy = ranking_strategy
        #TODO maybe strategy pattern is better here
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")

        
        self.ner_model = ner_model
        # ncpu = cpu_count()
        # print('Creating multiprocessing pool of {} size '.format(ncpu))
        # self.pool = Pool(int(ncpu/2))


    def link_entities(self,text,use_ner=True):
        """
        Process the query, find mentions and for each mention show the top-k 
        possible entities for each mention. 
        """
        text = preprocess(text)
        print(text)
        text_tokens = get_clean_tokens(text,self.nlp)

        mentions_ner = get_mentions_ner(text,self.ner_model,model_type='flair')


        #For each token find if some is a mention. Search the dictionary of mentions. 
        mention2pem = self.mention2pem
        mentions = [m for m in mentions_ner if m in mention2pem]
        
        print("Analizing mentions:",mentions)
        #Score entities for each mention

        return self.ranking_strategy.get_interpretations(text_tokens,mentions)
        
        # return entities_scores_mentions


def get_mentions_ner(text:str,nlp,model_type='flair') -> List[str]:

    if model_type=='flair':
        return get_mentions_flair(text,nlp)
        


def get_mentions_flair(text,nlp):
    sentence = Sentence(text)
    nlp.predict(sentence)

    mentions = []
    last_end = -3
    i = 0
    for entity in sentence.to_dict(tag_type='ner')['entities']:
        if (last_end+1) == entity['start_pos']:
            mentions[i-1] += ' ' + entity['text']
        else:
            mentions.append(entity['text'])
            i += 1
        last_end = entity['end_pos']

    return mentions
