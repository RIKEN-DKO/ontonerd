import re
from typing import List
import networkx
import itertools
from spacy.lang.en import English
import spacy
import obonet
import os
from tqdm import tqdm

def get_synonyms_formatted(graph, data):
    
    if 'synonym' not in data:
        return []

    res = data['synonym']
    synonyms = []
    for syn in res:
        syn_ = re.findall('"([^"]*)"', syn)
        if len(syn_) == 0:
            continue
        synonyms.append(preprocess(syn_[0]))

    return synonyms


def get_parents_ids(id,graph):
    """Returns a list with the ids of the parents of the node
    with `id`
    """
    parents=[]
    for child, parent, key in graph.out_edges(id, keys=True):
        if key == 'is_a':
            parents.append(parent)
            break # just one for now
    
    if len(parents) < 1:
        return []
    else:
        return (parents + get_parents_ids(parents[0],graph))

def get_children_ids(id,graph,deep=0,max_deep=3):
    """Returns a list with the ids of the children of the node
    with `id`
    """
    if deep>max_deep:
        return []

    children=[]
    for child,parent, key in graph.in_edges(id, keys=True):
        if key == 'is_a':
            children.append(child)
            # break # just one for now
    
    if len(children) < 1:
        return []
    else:
        new_children=[]
        for child in children:
            new_children.append(get_children_ids(child,graph,deep+1))

    return (children + flatten(new_children))

#https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
def flatten(a):
    return list(itertools.chain.from_iterable(a))

def preprocess(str):
    """Common preprocess the strings found in the ontologies
    """
    commas = str.split('"')
    #Cleaning when string :
    #'"very large definition blah blah." [goc:pamgo_curators]'
    #to get only: 'very large definition blah blah.'
    if len(commas)>1:
        str = commas[1]

    return str.lower().strip()


def create_spacy_line(context, words,qid,insert_space=False):

    start = context.find(words)
    end = start + len(words)
    new_context = context

    if insert_space and start != -1:
        # if context[start - 1] != ' ':
        #     new_context = context[:start-1] + ' ' + context[start-1:]
        #     start +=1
        #     end +=1
        # if context[end] != ' ':
        #     new_context = context[:end] + ' ' + context[end:]
        #     # start +=1
        #     # end +=1
        new_context = context[:start] + ' ' + context[start:end]+ ' ' +context[end:]
        start +=1
        end +=1

    line = None
    if start != -1:
        line = (new_context,
                {'links': {(start, end): {qid: 1.0}},
                 'entities': [(start, end, 'ONTO')]}
                )
    
    return line


def create_ner_sentence(string:str, context:str,nlp, insert_last_space=True,char_space=' '):
    """Creates one ner sentences in BIO format from a `string` and `context`. It search for `string` in the `context`.
    
    token by token.
    word_context O
    word_string B
    word_string I
    word_context O

    :param string: [description]
    :type string: str
    :param context: [description]
    :type context: str
    :param nlp: Spacy model
    :type nlp: [type]
    :param insert_last_space: [description], defaults to True
    :type insert_last_space: bool, optional
    :param char_space: [description], defaults to ' '
    :type char_space: str, optional
    :return: [description]
    :rtype: [type]
    """
 

    lines = []
    temp_lines = []

    tokens_str = [token.text for token in nlp(string)]
    doc = nlp(context)
    sentence_tokens = [[token.text for token in sent] for sent in doc.sents]
    j=0
    for sentence in sentence_tokens:
        found_token = False
        temp_lines=[]
        for i,token in enumerate(sentence):
            if tokens_str[j] == token:
                if j == 0:
                    temp_lines.append(token + ' ' + 'B')
                if j > 0:
                    temp_lines.append(token + ' ' + 'I')

                j += 1
                #the str was found
                #TODO sometimes a sentence doesnt have B I
                #TODO found several tokens_str
                if j >= len(tokens_str):
                    found_token = True
                    j = 0 

            else:
                j = 0
                temp_lines.append(token + ' ' + 'O')

        if found_token:
            lines.extend(temp_lines)
            if lines[-1] !='. O':
                lines.append('. O')
            if insert_last_space:
                lines.append(char_space)

    return lines


def create_ner_sentences_all(ONTOLOGIES:List[str], ONTO_PATH:str, MIN_LEN_SENTENCE=3, MAX_NUM_SENTENCE_PERDOC=1000):
    """This one first create a catalog of documents. Then for each node name search in every dcoument. 

    """
    nlp = English()
    nlp.add_pipe("sentencizer")
    sentences = []
    for ontology in ONTOLOGIES:
        obo_file = os.path.join(ONTO_PATH, ontology, ontology+".obo")
        print('Processing:  ', obo_file)
        graph = obonet.read_obo(obo_file)
        #first when construct a list of documents to search for entity names
        docs = []
        for qid, data in graph.nodes(data=True):
            for_process = []
            if 'name' in data:
                name = preprocess(data['name'])
                for_process.append(name)
                # docs.append(name)

            if 'def' in data:
                definition = preprocess(data['def'])
                # docs.append(definition)
                for_process.append(definition)

            synonyms = get_synonyms_formatted(graph, data)
            for_process.extend(synonyms)
            # docs.extend(synonyms)

            for text in for_process:
                doc = nlp(text)
                sentence_tokens = [[token.text for token in sent]
                                for sent in doc.sents]

                for sentence in sentence_tokens:
                    if len(sentence) >= MIN_LEN_SENTENCE:
                        docs.append(sentence)

        #We search entity name in each document
        for qid, data in graph.nodes(data=True):
            if 'name' in data:
                name = preprocess(data['name'])

            num_sentences = 0
            for doc in docs:
                if num_sentences > MAX_NUM_SENTENCE_PERDOC:
                    break
                sentence = create_ner_sentence(name, doc, nlp)
                if len(sentence) > 0:
                    sentences.append(sentence)
                    num_sentences += 1

    return sentences


def create_ner_sentences_children(
        ONTOLOGIES: List[str], 
        ONTO_PATH: str, 
        MAX_NUM_WORDS_ENTITY=10,
        debug=False,
        char_space=' '):
    """Creates the sentences from the ontology files. For each node it explored their children
    and checks if the name of the node is contained into the name and descriptions of the children.

    :param ONTOLOGIES: [description]
    :type ONTOLOGIES: List[str]
    :param ONTO_PATH: [description]
    :type ONTO_PATH: str
    :param MAX_NUM_WORDS_ENTITY: [description], defaults to 10
    :type MAX_NUM_WORDS_ENTITY: int, optional
    :param debug: [description], defaults to False
    :type debug: bool, optional
    :return: [description]
    :rtype: [type]
    """
    nlp = English()
    nlp.add_pipe("sentencizer")
    sentences = []
    for ontology in ONTOLOGIES:
        obo_file = os.path.join(ONTO_PATH, ontology, ontology+".obo")
        print('Reading ontology:  ', obo_file)
        graph = obonet.read_obo(obo_file)
        #TODO add bar tqdm
        print('Exploring nodes..')
        for qid, data in graph.nodes(data=True):

            if 'name' in data:
                name = preprocess(data['name'])
            else:
                continue
            #we insert the name of the entity as one single sentence
            sentences.append(create_ner_sentence(
                name, name, nlp, char_space))

            name_len = len(name)
            words_name = name.split(' ')
            #Very long names shouldn't be searched 
            if len(words_name) > MAX_NUM_WORDS_ENTITY:
                continue
            
            #TODO search for the synonyms names too... 
             
            #find the name in the children
            children = list(set(get_children_ids(qid, graph)))
            for child_id in children:
                docs = []
                child_data = graph.nodes[child_id]
                if 'name' in child_data:
                    child_name = preprocess(child_data['name'])
                    docs.append(child_name)

                if 'def' in child_data:
                    child_def = preprocess(child_data['def'])
                    #We extract sentences form the definition
                    def_sents = text_to_sentences(child_def,nlp)
                    docs.extend(def_sents)

                synonyms = get_synonyms_formatted(graph, child_data)
                docs.extend(synonyms)
                for doc in docs:
                    sentence = create_ner_sentence(
                        name, doc, nlp, char_space)
                    if len(sentence) > 0:
                        sentences.append(sentence)

            if debug and len(sentences)>30:
                print('debug')
                break

    print("Finished processing ontologies")
    return sentences

def create_pem_dictionary(ONTOLOGIES: List[str], ONTO_PATH: str):

    pem={}
    mention_freq = {}
    for ontology in ONTOLOGIES:
        obo_file = os.path.join(ONTO_PATH, ontology, ontology+".obo")
        print('Reading ontology:  ', obo_file)
        graph = obonet.read_obo(obo_file)
        #TODO add bar tqdm
        print('Exploring nodes..')
        for qid, data in graph.nodes(data=True):
            mentions = []
            if 'name' in data:
                name = preprocess(data['name'])
                mentions.append(name)

            synonyms = get_synonyms_formatted(graph, data)
            mentions.extend(synonyms)
            children = list(set(get_children_ids(qid, graph)))
            num_links = len(children)
            for mention in mentions:
                if mention not in mention_freq:
                    mention_freq[mention] = num_links + 1
                else:
                    mention_freq[mention] += 1

                if mention in pem:
                    if qid not in pem[mention]:
                        pem[mention][qid] = num_links + 1
                    else:
                        pem[mention][qid] += 1
                else:
                    pem[mention] = {qid: num_links + 1}

    #Computing pem
    for mention,count in mention_freq.items():
        if mention in pem:
            for qid,freq in pem[mention].items():
                pem[mention][qid]=freq/count

    print('[FINISHED]Exploring nodes..')
    return pem,mention_freq

            
def get_nodes_description(ONTOLOGIES: List[str], ONTO_PATH: str):
    
    nlp = English()
    # nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe("sentencizer")
    all_stopwords = nlp.Defaults.stop_words
    
    entity2description = {}
    for ontology in ONTOLOGIES:
        obo_file = os.path.join(ONTO_PATH, ontology, ontology+".obo")
        print('Reading ontology:  ', obo_file)
        graph = obonet.read_obo(obo_file)
        #TODO add bar tqdm
        print('Exploring nodes..')
        pbar = tqdm(total=len(graph))
        for qid, data in graph.nodes(data=True):
            pbar.update(1)
            desc = []
            if 'name' in data:
                name = preprocess(data['name'])
                desc.append(name)

            if 'def' in data:
                definition = preprocess(data['def'])
                desc.append(definition)

            synonyms = get_synonyms_formatted(graph, data)
            for synonym in synonyms:
                desc.append(synonym)
            sentence_tokens = []
            for des in desc:
                doc = nlp(des)
                sentence_tokens.append([token.text for token in doc if not token.is_punct ])

            sentence_tokens = list(set(flatten(sentence_tokens)))
            sentence_tokens = [
                word for word in sentence_tokens 
                if not word in all_stopwords]

            sentence_tokens = tuple(sentence_tokens)
            #preventing empty descriptions
            if len(sentence_tokens)>0:
                entity2description[qid] = sentence_tokens
            else: 
                entity2description[qid] = ('#UNK')
    
    return entity2description

        
def get_clean_tokens(text,nlp):
    text_tokens = []
    doc = nlp(text)
    all_stopwords = nlp.Defaults.stop_words
    text_tokens.append([token.text for token in doc if not token.is_punct ])
    text_tokens = list(set(flatten(text_tokens)))
    text_tokens = [
        word for word in text_tokens
        if not word in all_stopwords]
    
    return text_tokens


def text_to_sentences(text,nlp=None)->List[str]:
    #Case from spacy sentecizer
    if nlp is not None:
        sents = nlp(text)
        sents = [sen.text for sen in list(sents.sents)]
        return sents
    #fast split
    return text.split('.')