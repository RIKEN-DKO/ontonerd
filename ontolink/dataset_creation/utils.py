import re
import networkx

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