from typing import List,Dict


from dataset_creation.utils import get_clean_tokens,preprocess
from nltk.corpus import stopwords
from spacy.lang.en import English
from flair.data import Sentence
import time
import numpy as np
from entity_ranking import EntityRanking
from utils import _print_colorful_text
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
        # _text = preprocess(text)
        # doc = self.nlp(text)


        # [{'end_pos': 12, 'start_pos': 2, 'text': 'quaternary'},
        #  {'start_pos': 13, 'end_pos': 21, 'text': 'ammonium'}]
        ner_mentions = [] #a list of dicts

        #The text is divided into sentences and NER object search for mentions in each one
        #Neither spacy or nltk are give the correct boundaries,  they remove trailing spaces.
        # split() for now 
        last_sen_size = 0
        for sent in text.split('.'):
            # print('Analysing sentence:',sent)
            ner_mentions_textonly_sentence,ner_mentions_sentence = get_mentions_ner(sent,self.ner_model,model_type='flair')
            if len(ner_mentions_sentence)>0:
                for ment in ner_mentions_sentence:
                    ment['start_pos']+=last_sen_size
                    ment['end_pos']+=last_sen_size
                ner_mentions.extend(ner_mentions_sentence)
            #Last text lenght plus 1 for accounting the '.'
            last_sen_size += len(sent) + 1 
            # print('NER mentions:',ner_mentions)

        #For each token find if some is a mention. Search the dictionary of mentions. 
        #TODO find text_tokens positions in text
        clean_text_tokens = get_clean_tokens(text,self.nlp)
        tokendict_mentions = self.get_mentions_by_tokens_and_dict(text)
        # print('token mentions',tokendict_mentions)
        #combine the mentions found by the NER system and the ones found by
        #tokenization and dict searching. 
        mentions = ner_mentions + tokendict_mentions
        #Also delete repetitions: https://stackoverflow.com/questions/11092511/python-list-of-unique-dictionaries
        mentions = [dict(s) for s in set(frozenset(d.items())
                                        for d in mentions)]

        # print("Analizing mentions:")
        # _print_colorful_text(text,mentions)
        #Score entities for each mention

        interpretations = self.ranking_strategy.get_interpretations(clean_text_tokens,mentions)
        return interpretations

    def get_mentions_by_tokens_and_dict(self, text:str)->Dict:
        #tokenize and check if mentions exist in the mention dictionary
        nlp = self.nlp
        text_tokens = []
        doc = nlp(text)
        all_stopwords = nlp.Defaults.stop_words
        for token in doc:
            if ((not token.is_punct) 
            and (token.text not in all_stopwords)
            and (token.text in self.mention2pem)):
                text_tokens.append({
                    'text': token.text,
                    'start_pos': token.idx,
                    'end_pos': token.idx+len(token.text),
                })
        
        return text_tokens



def get_mentions_ner(text:str,nlp,model_type='flair') -> List[str]:

    if model_type=='flair':
        return get_mentions_flair(text,nlp)
        


def get_mentions_flair(text,nlp):
    sentence = Sentence(text)
    nlp.predict(sentence)

    mentions = []
    start_poss=[]
    end_poss = []

    last_end = -3
    i = 0
    for entity in sentence.to_dict(tag_type='ner')['entities']:
        #combine
        if (last_end+1) == entity['start_pos']:
            mentions[i-1] += ' ' + entity['text']
            end_poss[i-1] = entity['end_pos']
        else:
            mentions.append(entity['text'])
            i += 1
            start_poss.append(entity['start_pos'])
            end_poss.append(entity['end_pos'])
        last_end = entity['end_pos']

    #Pack all ina dict
    results = []
    for i,mention in enumerate(mentions): 
        results.append({
            'text':mention,
            'start_pos':start_poss[i],
            'end_pos':end_poss[i],
        })

    return mentions,results
