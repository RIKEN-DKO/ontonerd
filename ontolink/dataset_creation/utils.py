import re
import networkx
import itertools

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


def create_ner_sentence(str, context,nlp, insert_last_space=True):

    # start = context.find(str)
    # end = start + len(str)
    # new_context = context
    # if start == -1:
    #     return []

    lines = []
    temp_lines = []

    tokens_str = [token.text for token in nlp(str)]
    # doc = nlp(context)
    # sentence_tokens = [[token.text for token in sent] for sent in doc.sents]
    j=0
    found_token=False
    # for sentence in sentence_tokens:
    for i,token in enumerate(context):
        if tokens_str[j] == token:
            if j == 0:
                temp_lines.append(token + ' ' + 'B')
            if j > 0:
                temp_lines.append(token + ' ' + 'I')

            j += 1

            if j >= len(tokens_str):
                found_token = True
                lines.extend(temp_lines)
                temp_lines=[]
                j = 0 

        else:
            j = 0
            lines.append(token + ' ' + 'O')

    if insert_last_space:
        lines.append(' ')

    if found_token:
        return lines
    else:
        return []
