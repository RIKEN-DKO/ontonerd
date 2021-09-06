import nltk
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from utils import insert_if_anybig
# from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
from itertools import repeat
from nltk.corpus import stopwords
# from multiprocessing import cpu_count
# from functools import partial
# from pathos.multiprocessing import ProcessingPool as Pool

import time

import numpy as np

class EntityLinker:
    """
    TODO add doc
    """
    def __init__(self, commonness_dict, entities_descriptions, terms_frequency,collection_size_terms,ner_model=None):
        self.commonness_dict = commonness_dict
        self.entities_descriptions = entities_descriptions
        self.entities_list = list(entities_descriptions.keys())
        self.entities_list_np = np.asarray(self.entities_list)
        print(self.entities_list_np.shape)
        self.terms_frequency = terms_frequency
        self.collection_size_terms = collection_size_terms


        self.ner_model = ner_model
        # ncpu = cpu_count()
        # print('Creating multiprocessing pool of {} size '.format(ncpu))
        # self.pool = Pool(int(ncpu/2))


    def link_entities_query(self,query,use_ner=True):
        """
        Process the query, find mentions and for each mention show the top-k 
        possible entities for each mention. 
        """
        #tokenize
        tokens = word_tokenize(query)
        stop_words = set(stopwords.words("english"))
        filtered_tokens = []
        for w in tokens:
            #do not take stop words
            if w not in stop_words:
                filtered_tokens.append(w)

        #Obtain mentions give by the BERT model
        if use_ner and self.ner_model != None:
            bert_mentions = self.ner_model(query)
            #Return a list of dictionaries
            # [{'entity_group': 'bio',
            #   'score': 0.9997287392616272,
            #   'word': 'hiv',
            #   'start': 22,
            #   'end': 25},
            for men_group in bert_mentions:
                mention = men_group['word']
                if mention not in filtered_tokens:
                    filtered_tokens.append(mention)

        #For each token find if some is a mention. Search the dictionary of mentions. 
        mentions_dict = self.commonness_dict
        mentions = [m for m in filtered_tokens if m in mentions_dict]
        
        print("Analizing mentions:",mentions)
        #Score entities for each mention

        #TODO Slow bootleneck    
        entities_scores_mentions = {}
        for mention in mentions:

            scored_entities = self.score_E_q_m_filterby_pem(query, mention)
            entities_scores_mentions[mention] = scored_entities


        interpretations = self.gen_interpretations(entities_scores_mentions)
        return interpretations

        return entities_scores_mentions


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
    def score_E_q_m(self,q,m,k_top=10):
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
            score = self.score_e_q_m(entity, q, m)
            #Try to insert scores if bigger than any element
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
        pem_scores = [self.p_e_m(entity, m) for entity in self.entities_list]
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



    def score_e_q_m(self, e: int, q, m):
        """ Score the given entity (e) in the question (q) given mention(m)   
            From Balog 7.3.3.1
        """
        #TODO change p_m_m e to string
        P_e_m = self.p_e_m(e, m)
        P_q_e = self.p_q_e(q, e)   

        return P_e_m*P_q_e



    def p_e_m(self, e: int, m: str,return_entity = False):
        """Commonness P(e|m)
        P(e|m) = n(e,m)/Total

        n(e,m) = Number of times e it is used as a link destination for m

        `m`:mention
        `e`:entity ID
        dic_comm[Mention]
        e.g
        mentions['chloride'] = {'entities': [
            4167203, 254493001], 'num_entity': [78637, 2], 'num': 78639}

        `entities`: An array of entities related with mention
        `num_entity`: The number of times the entity in `entities` is used
        `num`:the Total number of times mention(m) is linked with  entity(e)

        """
        
        #ignored entity ids equal to zero
        if e == 0 or e == '0':
            return 0

        mentions = self.commonness_dict
        mention = m
        EntityID = e
        n_em = 0
        total_n = 1
        if mention in mentions:
            #search for the index of the entityID 
            index_entity = -1
            entities = mentions[mention]["entities"]
            for i,e in enumerate(entities):
                if e == EntityID:
                    index_entity = i
            
            if index_entity != -1 :
                n_em = mentions[mention]["num_entity"][index_entity]
                total_n = mentions[mention]["num"]


                #If the entityID exist inside
                # index_entity = mentions[mention]["entities"].index(EntityID)
            # except ValueError:
            #     #EntityID do not exist
            #     #TODO: Check where the Error comes index or []
            #     # print('Entity not found')
            #     pass
        pem = n_em / total_n
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
        entities_desc = self.entities_descriptions
        count = 0
        tokens = ['']
        try:
            tokens = entities_desc[e]
            # tokens = word_tokenize(description)
            # print(tokens)
            # print(description)
            for token in tokens:
                if token == t:
                    count += 1
        except KeyError:
            # print('Entity {} not found in dictionary'.format(e))
            pass

        if return_len:
            return len(tokens), count

        return count

    def p_t_thetae(self, e:int, t,return_parts = False):
        """ P (t|θe)
        `entity_catalog`: A dict of entities and descriptions

        Jelinek-Mercer Smooting
        Eq 3.9 Balog
        P (t|θe) = (1 − λ)P (t|e) + λP (t|E) , entity language models 
        P(t|e) = c(t;e)/le 
        
        """
        terms_freq = self.terms_frequency
        collection_size_terms = self.collection_size_terms
        entity_catalog = self.entities_descriptions
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


    def p_q_e(self, q, e:int):
        """
            P(q|e) probability of questions give an entity
            Equation 7.5 Balog

            q:question
            e:entity
        """

        tokens = word_tokenize(q)
        #get the counter of each term
        terms = FreqDist(tokens)
        lq = len(tokens)
        lamb_ = 0.5
        mul_up = 1
        mul_down = 1
        for t, c_tq in terms.items():

            p_t_e, p_t_E = self.p_t_thetae(e, t, return_parts=True)
            p_t_theta = (1-lamb_) * p_t_e + lamb_ * p_t_E
            if p_t_E == 0 and p_t_e == 0:
                continue
            mul_up *= (pow(p_t_theta, c_tq/lq)) 
            mul_down *= (pow(p_t_E, c_tq/lq))

        return mul_up/mul_down

