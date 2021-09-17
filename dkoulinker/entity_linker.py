from typing import List,Dict


from dkoulinker.dataset_creation.utils import get_clean_tokens, preprocess
from nltk.corpus import stopwords
from spacy.lang.en import English
from flair.data import Sentence
import time
import numpy as np
from dkoulinker.entity_ranking import EntityRanking
from dkoulinker.utils import (_print_colorful_text, is_overlaping, log)
import pprint

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
                ner_model_type='flair',
                prune_overlapping_method='best_score'):

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
        self.prune_overlapping_method = prune_overlapping_method

        
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
            if sent == '': #text ending with '.', give empty sentence 
                continue
            log('Analysing sentence:',sent)
            ner_mentions_textonly_sentence,ner_mentions_sentence = get_mentions_ner(sent,self.ner_model,model_type='flair')
            if len(ner_mentions_sentence)>0:
                for ment in ner_mentions_sentence:

                    ment['start_pos']+=last_sen_size
                    ment['end_pos']+=last_sen_size

                #Skip mentions not in dictionary
                    if ment['text'] in self.mention2pem:
                        ner_mentions.append(ment)
            #Last text lenght plus 1 for accounting the '.'
            last_sen_size += len(sent) + 1 
            # log('NER mentions:',ner_mentions)

        #For each token find if some is a mention. Search the dictionary of mentions. 
        #TODO find text_tokens positions in text
        clean_text_tokens = get_clean_tokens(text,self.nlp)
        tokendict_mentions = self.get_mentions_by_tokens_and_dict(text)
        # log('token mentions',tokendict_mentions)
        #combine the mentions found by the NER system and the ones found by
        #tokenization and dict searching. 
        mentions = ner_mentions + tokendict_mentions
        #Also delete repetitions: https://stackoverflow.com/questions/11092511/python-list-of-unique-dictionaries
        mentions = [dict(s) for s in set(frozenset(d.items())
                                        for d in mentions)]

        log("Analizing mentions:")
        # _print_colorful_text(text,mentions)
        #Score entities for each mention

        interpretations = self.ranking_strategy.get_interpretations(clean_text_tokens,mentions)

        # log(interpretations)
        # return interpretations
        return self.prune_overlapping_entities(interpretations,method= self.prune_overlapping_method)
    
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
    
    def prune_overlapping_entities(self,interpretations:List[Dict],method='best_score')->List[Dict]:
        """Detect overlapping entities by it position in text. 


        :param interpretations: [description]
        :type interpretations: List[Dict]
        :param method:
        Methods:
            'best_score' choose only the entity with best score over two overlapping text.
            'large_text' choose only the entity with large text score over two overlapping text.
        :type method: str, optional
        :return: [description]
        :rtype: List[Dict]
        """
        new_interpretations=[]
        overlapping_indices=[]
        interpretations.sort(key=lambda x: x['start_pos'])
        for i in range(0,len(interpretations)):
            if i in overlapping_indices:
                continue
            best_interp = interpretations[i]
            log(overlapping_indices)
            log(i, 'best', best_interp['text'], best_interp['best_entity'])
            for j in range(i+1,len(interpretations)):

                other_interp = interpretations[j]
                log(j, 'other', other_interp['text'], other_interp['best_entity'])
                best_interval = [best_interp['start_pos'],best_interp['end_pos']]
                other_interval = [other_interp['start_pos'],other_interp['end_pos']]

                #Is overlapping
                if is_overlaping(best_interval,other_interval) and j not in overlapping_indices:
                    log('overlap!')
                    overlapping_indices.append(j)
                    overlapping_indices.append(i)
                    #comparing scores
                    if method =='best_score':             
                        if best_interp['best_entity'][1] < other_interp['best_entity'][1]:
                            best_interp = other_interp
                    elif method =='large_text':
                        if len(best_interp['text']) < len(other_interp['text']):
                            best_interp = other_interp
                    else:
                        raise('Unknow method')
                        


            new_interpretations.append(best_interp)
        
        return new_interpretations

# def prune_overlapping_entities2(self, interpretations: List[Dict], method='best_score') -> List[Dict]:
#     if len(interpretations) < 1:
#         return interpretations
#     #sort the intervals by its first value
#     interpretations.sort(key=lambda x: x['start_pos'])

#     merged_list = []
#     merged_list.append(interpretations[0])
#     for i in range(1, len(interpretations)):
#         pop_element = merged_list.pop()

#         pop_interval = [pop_element['start_pos'], pop_element['end_pos']]
#         other_interval = [interpretations[i]['start_pos'],
#                           interpretations[i]['end_pos']]

#         if is_overlaping(pop_element['start_pos'], other_interval):

#             # new_element = pop_element[0], max(pop_element[1], interpretations[i][1])
#             if method == 'best_score':
#                 if best_interp['best_entity'][1] < other_interp['best_entity'][1]:
#                     best_interp = other_interp
#             elif method == 'large_text':
#                 if len(best_interp['text']) < len(other_interp['text']):
#                     best_interp = other_interp
#             else:
#                 raise('Unknow method')
#             merged_list.append(new_element)
#         else:
#             merged_list.append(pop_element)
#             merged_list.append(interpretations[i])
#     return merged_list


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
