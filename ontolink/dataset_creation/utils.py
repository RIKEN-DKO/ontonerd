import re
import networkx
import itertools

def get_synonyms_formatted(graph, data):
    
    res = data['synonym']
    synonyms = []
    for syn in res:
        syn_ = re.findall('"([^"]*)"', syn)
        if len(syn_) == 0:
            continue
        synonyms.append(syn_[0])

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
                'entities': [(start, end, None)]}
                )
    
    return line

