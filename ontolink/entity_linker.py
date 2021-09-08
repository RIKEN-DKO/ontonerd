from typing import List
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from utils import insert_if_anybig
from dataset_creation.utils import get_clean_tokens
from itertools import repeat
from nltk.corpus import stopwords
from spacy.lang.en import English
from flair.data import Sentence
import time
import numpy as np

class EntityLinker:
    """
    TODO add doc
    """
    def __init__(self, 
    mention2pem, 
    entity2description, 
    mention_freq,
    collection_size_terms,
    ner_model=None,
    ner_model_type='flair'):
        self.mention2pem = mention2pem
        self.entity2description = entity2description
        self.entities_list = list(entity2description.keys())
        self.entities_list_np = np.asarray(self.entities_list)
        print(self.entities_list_np.shape)
        self.mention_freq = mention_freq
        #TODO is necesary?
        self.collection_size_terms = collection_size_terms

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

        text_tokens = get_clean_tokens(text,self.nlp)

        mentions_ner = get_mentions_ner(text,self.ner_model,model_type='flair')


        #For each token find if some is a mention. Search the dictionary of mentions. 
        mention2pem = self.mention2pem
        mentions = [m for m in mentions_ner if m in mention2pem]
        
        print("Analizing mentions:",mentions)
        #Score entities for each mention

        #TODO Slow bootleneck   

        entities_scores_mentions = {}
        for mention in mentions:

            # scored_entities = self.score_E_q_m_filterby_pem(text, mention)
            scored_entities = self.score_E_q_m(text_tokens, mention)
            entities_scores_mentions[mention] = scored_entities


        interpretations = self.gen_interpretations(entities_scores_mentions)

        return interpretations
        
        # return entities_scores_mentions


    def gen_interpretations(self, entities_scores_mentions:dict,method='max'):
        """
        A dict with mention as keys, the values contains the top-k entities assigment and scores
        """

        if method == 'max':
            return self.get_max_interpretation(entities_scores_mentions)


        return self.get_max_interpretation(entities_scores_mentions)

    def get_max_interpretation(self, entities_scores_mentions:dict, eps=1e-6):
        """
        Generate an interpretation by selection the one with the biggest score
        """
        new_entities_scores_mentions = {}
        for mention,entity_and_score in entities_scores_mentions.items():
            #unzip the tuples of entities and scores
            #https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists
            # entities,scores = list(map(list, zip(*entity_and_score)))
            #The list comes in ascending scores, last is the max
            best_entity,score = list(entity_and_score)[-1] 
            
            if score > eps :
                new_entities_scores_mentions[mention] = (best_entity,score)

        
        return new_entities_scores_mentions
    #Slow
    def score_E_q_m(self,text_tokens,m,k_top=10):
        """ Score  in the question (q) given mention(m)   
            From Balog 7.3.3.1
            Returns a list of entities ID in decreasion order 
        """

        # entities = self.entities_descriptions
        e_scores = [-1] * k_top
        topk_entities = [0] * k_top
        
        for entity in self.entities_list:
            # print(entity,desc)
            # Ids equal to zero are not useful
            if entity == 0 or entity == '0':
                continue
            score = self.score_e_q_m(entity, text_tokens, m)
            #Try to insert scores if bigger than any element
            #TODO insert_if_anybig is slow?
            i,e_scores = insert_if_anybig(e_scores,score)
            #If a insertion happen
            if i != -1:
                topk_entities[i] = entity

        return zip(topk_entities,e_scores)

    #TODO slow
        """ es ID in decreasion order 
            k_pem: 
        """
    def score_E_q_m_filterby_pem(self, q, m,k_pem=100, k_top=10,mode='sorted'):
        """Score  in the question (q) given mention(m)   
            From Balog 7.3.3.1
            This version first computes P(e|m) for all entities since is fast. Then computes P(q|e)
            on the top-K from p(e|m)
            
        Args:
            q ([type]): query
            m ([type]): mention
            k_pem (int, optional): p_q_e is calculated only for the top k_pem entities
            k_top (int, optional): The number of entities to return. Defaults to 10.
            mode (str, optional):  Defaults to 'sorted'.

        Returns:
            tuple: Returns two one of scores and one of entities ids
        """
        start = time.time()

        #TODO map np ? 
        # pem_scores = [self.p_e_m(entity, m) for entity in self.entities_list]
        pem_scores = self.mention2pem[m].values()

        if mode== 'sorted': 
            _pem_scores = np.asarray(pem_scores)
            ind = _pem_scores.argsort()
            pem_scores = _pem_scores[ind][-k_pem:]
            pem_entities = self.entities_list_np[ind][-k_pem:]
            
            # function p_q_e is very costly, We only calculate for the best entites by higher pem.
            e_scores = [pem_scores[i]*self.p_q_e(q, pem_entities[i]) for i in range(k_pem) ]
            _e_scores = np.asarray(e_scores)
            ind = _e_scores.argsort()
            e_scores = _e_scores[ind][-k_top:]
            topk_entities = pem_entities[ind][-k_top:]
            return zip(topk_entities.tolist(), e_scores.tolist())
        elif mode == 'max':

            # e_scores = [self.p_e_m(entity, m) * self.p_q_e(q, entity)
            #             for entity in self.entities_list]
            e_scores = []
            for entity in self.entities_list:
                pem = self.p_e_m(entity, m)
                pqe = self.p_q_e(q, entity)
                e_scores.append(pem*pqe)

            imax = e_scores.index(max(e_scores))

            print("Bootleneck  took: {} s".format(time.time() - start))

            return (self.entities_list[imax],e_scores[imax])



    def score_e_q_m(self, e: int, text_tokens, m):
        """ Score the given entity (e) in the question (q) given mention(m)   
            From Balog 7.3.3.1
        """
        #TODO change p_m_m e to string
        P_e_m = self.p_e_m(e, m)
        P_q_e = self.p_q_e(text_tokens, e)   

        return P_e_m*P_q_e



    def p_e_m(self, e: int, m: str,return_entity = False):
        """Commonness P(e|m)
        P(e|m) = n(e,m)/Total

        `m`:mention
        `e`:entity ID
        """
        #mention2pem['reproduction'] = {'GO:0000003': 1.0}
        #mention2pem['mention mention'] = {'ID': p(e|m)}
        pem = 0
        if e in self.mention2pem[m]:
            pem = self.mention2pem[m][e]



        if return_entity:
            return e,pem
        else:
            return pem 

        #commoness

    #TODO change entity e to int or make str in all cases
    def c(self, e: int,t:str,return_len=False):
        """
            Frequency of term t in the entity’s description and le is the length of the entity’s 
            description (measured in the number of terms).

            entities_desc , dictionary of entities descriptions
        """
        entities_desc = self.entity2description
        count = 0
        tokens = ['']

        #TODO there' a case with no description?
        tokens = entities_desc[e]
        if len(tokens)<1:
            # print('Entity with no desc:',e)
            tokens = ['']
        # tokens = word_tokenize(description)
        # print(tokens)
        # print(description)
        for token in tokens:
            if token == t:
                count += 1


        if return_len:
            return len(tokens), count

        return count

    def p_t_thetae(self, e, t,return_parts = False):
        """ P (t|θe)
        `entity_catalog`: A dict of entities and descriptions

        Jelinek-Mercer Smooting
        Eq 3.9 Balog
        P (t|θe) = (1 − λ)P (t|e) + λP (t|E) , entity language models 
        P(t|e) = c(t;e)/le 
        
        """
        terms_freq = self.mention_freq
        collection_size_terms = self.collection_size_terms
        entity_catalog = self.entity2description
        le, cte = self.c(e, t, return_len=True)
        p_t_e = cte/le
        lamb_ = 0.5
        # sum_cte =0
        # sum_le =0
        #TODO: ...
        #search for term in all the entities catalog
        # for entity,desc in entity_catalog.items():
        #     t_le,t_cte = c(entity, t, entity_catalog,return_len=True)
        #     sum_le += t_le
        #     sum_cte += t_cte
        try:
            p_t_E = terms_freq[t]/collection_size_terms
        except KeyError:
            p_t_E = 0


        # print("P(t|Epsilon)=", p_t_E)

        if return_parts:
            return p_t_e, p_t_E

        return (1-lamb_) * p_t_e + lamb_ * p_t_E


    def p_q_e(self, text_tokens, e):
        """
            P(q|e) probability of questions give an entity
            Equation 7.5 Balog

            q:question
            e:entity
        """
        # tokens = word_tokenize(q)
        #get the counter of each term
        
        #TODO change to spacy
        terms = FreqDist(text_tokens)
        lq = len(text_tokens)
        lamb_ = 0.5
        mul_up = 1
        mul_down = 1
        for t, c_tq in terms.items():

            p_t_e, p_t_E = self.p_t_thetae(e, t, return_parts=True)
            p_t_theta = (1-lamb_) * p_t_e + lamb_ * p_t_E
            if p_t_E == 0 or p_t_e == 0:
                continue
            mul_up *= (pow(p_t_theta, c_tq/lq)) 
            mul_down *= (pow(p_t_E, c_tq/lq))

        return mul_up/mul_down


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
