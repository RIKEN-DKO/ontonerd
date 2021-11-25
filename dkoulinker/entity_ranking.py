
from abc import abstractmethod
from typing import Dict, List
import operator
from nltk.probability import FreqDist
from dkoulinker.utils import log
from operator import itemgetter

class EntityRanking:
    """Manage how we will rank the entities

    """
    @abstractmethod
    def get_interpretations(self, text_tokens: List[str], mentions: List[str])->Dict:
        return {}


class DictionaryRanking(EntityRanking):
    """
    Just ranks by returning the best entity ranked by p(e|m). It doesn't considers context.
    """

    def __init__(self, 
    mention2pem:Dict) -> None:

        super().__init__()

        self.mention2pem = mention2pem

    @abstractmethod
    def get_interpretations(self,text_tokens:List[str],mentions:List[str])->Dict:
        mention2pem = self.mention2pem
        interpretations = {}

        for mention in mentions:

            if mention in mention2pem:
                interpretations[mention]=max(mention2pem[mention].items(),
                                   key=operator.itemgetter(1))


        return interpretations


class QueryEntityRanking(EntityRanking):
    """
    Score  in the query (q) given mention(m)   
            From Balog 7.3.3.1
    """

    def __init__(self,
                 entity2description,
                 mention_freq,
                 mention2pem: Dict,
                 min_score_for_be_ranked=1e-6,  # entity with score too small should be ignored
                 ) -> None:

        super().__init__()
        self.min_score_for_be_ranked = min_score_for_be_ranked
        self.mention2pem = mention2pem
        self.entity2description = entity2description
        self.mention_freq = mention_freq
        self.len_terms_collection = len(mention_freq)


    @abstractmethod
    def get_interpretations(self, text_tokens: List[str], mentions: List[Dict]) -> List[Dict]:


        # entities_scores_mentions = {}
        for mention in mentions:

            scored_entities = self.score_E_q_m(text_tokens, mention['text'])
            # entities_scores_mentions[mention] = scored_entities
            mention['entities'] = scored_entities

        interpretations = self.gen_interpretations(mentions)

        return interpretations
        # return mentions

    def gen_interpretations(self, entities_scores_mentions: List[Dict], method='max'):
        """
        A dict with mention as keys, the values contains the top-k entities assigment and scores
        """

        if method == 'max':
            return self.get_max_interpretation(entities_scores_mentions)

        return self.get_max_interpretation(entities_scores_mentions)

    def get_max_interpretation(self, entities_scores_mentions: List[Dict]):
        """
        Generate an interpretation by selection the one with the biggest score
        """
        eps = self.min_score_for_be_ranked
        new_entities_scores_mentions = {}
        for mention in entities_scores_mentions:
            # print(mention)


            entity_and_score = mention['entities']
            # log(entity_and_score)
            sorted_by_second = sorted(
                entity_and_score, key=itemgetter(1), reverse=True)
            # log(sorted_by_second)
            best_entity, score = sorted_by_second[0]
            # best_entity, score = list(entity_and_score.sort(key=lambda x: x[1]))[0]

            if score > eps:
                mention['best_entity'] = (best_entity, score)

        return entities_scores_mentions
  
    def score_E_q_m(self, text_tokens:List[str], mention:str):
        """ Score  in the question (q) given mention(m)   
            From Balog 7.3.3.1 
            score(e;q,m) = P (e|q,m) ∝ P(e|m) P(q|e)
            Returns a list of entities ID in decreasion order 
        """

        # entities = self.entities_descriptions
        #We search in the relaed entities of the mention
        e_scores=[]
        for entityid,pem in self.mention2pem[mention].items():
            # print(entity,desc)
            # Ids equal to zero are not useful
            #This is for pudmed
            if entityid == 0 or entityid == '0':
                continue
            score = self.score_e_q_m(entityid, text_tokens, mention)

            e_scores.append((entityid,score))


        return e_scores

    def score_e_q_m(self, entityid, text_tokens, mention):
        """ Score the given entity (e) in the question (q) given mention(m)   
            From Balog 7.3.3.1
        """
        
        P_e_m = self.p_e_m(entityid, mention)
        P_q_e = self.p_q_e(text_tokens, entityid)

        return P_e_m*P_q_e

    def p_e_m(self, entityid, mention: str, return_entity=False):
        """Commonness P(e|m)
        P(e|m) = n(e,m)/Total

        `m`:mention
        `e`:entity ID
        """
        #mention2pem['reproduction'] = {'GO:0000003': 1.0}
        #mention2pem['mention mention'] = {'ID': p(e|m)}
        pem = 0
        if entityid in self.mention2pem[mention]:
            pem = self.mention2pem[mention][entityid]

        if return_entity:
            return entityid, pem
        else:
            return pem

        #commoness

    def p_q_e(self, text_tokens, entityid):
        """
            P(q|e) probability of questions give an entity
            Equation 7.5 Balog

            q:question
            e:entity
        """
        # tokens = word_tokenize(q)
        #get the counter of each term

        #TODO change to spacy?
        terms = FreqDist(text_tokens)
        lq = len(text_tokens)
        lamb_ = 0.5
        mul_up = 1
        mul_down = 1
        #c_term count of terms in the query
        for term, c_term in terms.items():

            p_t_e, p_t_Eps = self.p_t_thetae(entityid, term, return_parts=True)
            p_t_theta = (1-lamb_) * p_t_e + lamb_ * p_t_Eps
            if p_t_Eps == 0 or p_t_e == 0:
                continue
            mul_up *= (pow(p_t_theta, c_term/lq))
            mul_down *= (pow(p_t_Eps, c_term/lq))

        return mul_up/mul_down

    #TODO change entity e to int or make str in all cases
    def c(self, entityid, term: str, return_len=False):
        """
            Frequency of term t in the entity’s description and len is the length of the entity’s 
            description (measured in the number of terms).

            entities_desc , dictionary of entities descriptions
        """
        entities_desc = self.entity2description
        count = 0
        tokens = ['']

        #TODO there' a case with no description?
        tokens = entities_desc[entityid]
        if len(tokens) < 1:
            # print('Entity with no desc:',e)
            tokens = ['']
        # tokens = word_tokenize(description)
        # print(tokens)
        # print(description)
        for token in tokens:
            if token == term:
                count += 1

        if return_len:
            return len(tokens), count

        return count

    def p_t_thetae(self, entityid, term, return_parts=False):
        """ P (t|θe)
        `entity_catalog`: A dict of entities and descriptions

        Jelinek-Mercer Smooting
        Eq 3.9 Balog
        P (t|θe) = (1 − λ)P (t|e) + λP (t|E) , entity language models 
        P(t|e) = c(t;e)/le 
        
        """
        terms_freq = self.mention_freq
        le, cte = self.c(entityid, term, return_len=True)
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
        if term in terms_freq:
            p_t_Eps = terms_freq[term]/self.len_terms_collection
        else:
            p_t_Eps = 0

        # print("P(t|Epsilon)=", p_t_E)

        if return_parts:
            return p_t_e, p_t_Eps

        return (1-lamb_) * p_t_e + lamb_ * p_t_Eps
