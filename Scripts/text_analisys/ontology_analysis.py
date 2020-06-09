import rdflib #requirement
from rdflib import plugin, URIRef
from rdflib.namespace import Namespace
from rdflib.graph import Graph
import urllib

from nltk.corpus import wordnet as wn
from pprint import pprint

from queue import Queue

cso_nt_path = "C:\\Users\\GIORGIO-DESKTOP\\Desktop\\CSO.3.1.12.nt"

g = Graph()
g.parse(cso_nt_path, format="nt")

CSO_SUPER_TOPIC = URIRef("http://cso.kmi.open.ac.uk/schema/cso#superTopicOf")

def rdf_bfs(triple,max_depth=5,specific_lvl=None):

    Q = Queue()
    visitSet = set()
    resultSet = dict()

    Q.put_nowait(triple)
    visitSet.add(triple)
    if (not specific_lvl or specific_lvl==0): resultSet.update({triple: 0})
    
    while( not Q.empty() ):
        node = Q.get_nowait()

        if(resultSet[node] <= max_depth):
            node_lvl = resultSet[node]
            f.write("\t"*node_lvl+str(node[2])+"\n")
            for s, p, o in g.triples((node[2],CSO_SUPER_TOPIC,None)):
                t_child = (s,p,o)
                child_lvl = node_lvl+1

                if not (t_child in visitSet):
                    f.write("\t"*child_lvl)
                    f.write(str(t_child[0])+"\t")
                    f.write(str(t_child[1])+"\t")
                    f.write(str(t_child[2])+"\t")
                    f.write("\n")
                    Q.put_nowait(t_child)
                    visitSet.add(t_child)
                    if (not specific_lvl or specific_lvl==child_lvl): resultSet.update({t_child: child_lvl})

    return resultSet


def isInOntology(topic,space_separator="#"):
    ont_topic = urllib.parse.quote("_".join(topic.split(space_separator)))
    cso_topic_prefix = "https://cso.kmi.open.ac.uk/topics/"
    triples = g.triples((None,None,URIRef(cso_topic_prefix+ont_topic)))
    
    print(cso_topic_prefix+ont_topic)

    try: next(triples)
    except StopIteration: return False 
    
    return True

def getWnTerm(term):
    synset = wn.synsets(term)
    if(not synset): 
        return None
    syn = synset[0]
    return syn

def showTree(synset):
    hyp = lambda s:s.hyponyms()
    pprint(synset.tree(hyp,depth=5))

if __name__ == "__main__":
    # syn = wn.synsets("dog")[0]

    with open("C:\\Users\\GIORGIO-DESKTOP\\Desktop\\query_res.txt", "w") as f:

        # <https://cso.kmi.open.ac.uk/topics/communication_channels_%28information_theory%29>
        # <http://cso.kmi.open.ac.uk/schema/cso#superTopicOf>
        # <https://cso.kmi.open.ac.uk/topics/antenna_array> .

        # s = URIRef("https://cso.kmi.open.ac.uk/topics/communication_channels_%28information_theory%29")
        # p = URIRef("http://cso.kmi.open.ac.uk/schema/cso#superTopicOf")
        o = URIRef("https://cso.kmi.open.ac.uk/topics/antenna_arrays")

        paren_topic = g.value(subject=None,predicate=CSO_SUPER_TOPIC,object=o)
        print(paren_topic)
        f.write(str(rdf_bfs((None,CSO_SUPER_TOPIC,paren_topic),max_depth=2)))


        # print(isInOntology("antenna#arrays"))


