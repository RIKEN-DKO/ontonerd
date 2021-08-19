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
    return str.lower().strip()