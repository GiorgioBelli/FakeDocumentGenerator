
import rdflib #requirement
from rdflib import plugin, URIRef
from rdflib.namespace import Namespace
from rdflib.graph import Graph
from rdflib.paths import evalPath, MulPath, SequencePath, OneOrMore, ZeroOrMore, ZeroOrOne
import urllib

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from pprint import pprint

from queue import Queue

import itertools as it

cso_nt_path = "..\\datasets\\CSO.3.1.12.nt"

# g = Graph()
# g.parse(cso_nt_path, format="nt")
g = None

CSO_SUPER_TOPIC = URIRef("http://cso.kmi.open.ac.uk/schema/cso#superTopicOf")

def rdf_bfs(triple,max_depth=5,specific_lvl=None,asTree=True,fd=None):

    Q = Queue()
    visitSet = set()
    resultSet = dict()

    predicate = CSO_SUPER_TOPIC if asTree else None

    Q.put_nowait(triple)
    visitSet.add(triple)
    if (not specific_lvl or specific_lvl==0): resultSet.update({triple: 0})
    
    while( not Q.empty() ):
        node = Q.get_nowait()

        if(resultSet[node] <= max_depth):
            node_lvl = resultSet[node]
            if fd: fd.write("\t"*node_lvl+str(node[2])+"\n")
            for s, p, o in g.triples((node[2],predicate,None)):
                t_child = (s,p,o)
                child_lvl = node_lvl+1

                if not (t_child in visitSet):
                    if fd:
                        f.write("\t"*child_lvl)
                        f.write(str(t_child[0])+"\t")
                        f.write(str(t_child[1])+"\t")
                        f.write(str(t_child[2])+"\t")
                        f.write("\n")
                    Q.put_nowait(t_child)
                    visitSet.add(t_child)
                    if (not specific_lvl or specific_lvl==child_lvl): resultSet.update({t_child: child_lvl})

    return resultSet

def rdf_get_sibligs(concept):
    o = URIRef("https://cso.kmi.open.ac.uk/topics/%s" % concept)

    paren_topic = g.value(subject=None,predicate=CSO_SUPER_TOPIC,object=o)
    return [child[2].split("/")[-1] for child in rdf_bfs((None,CSO_SUPER_TOPIC,paren_topic),max_depth=1, asTree=True)]


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

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
 
def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count, best_score = 0.0, 0, 0.0
 
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        score_list = list(filter(lambda ps: ps, [synset.path_similarity(ss) for ss in synsets2]))
        best_score = max(score_list) if score_list else 0
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
 
    # Average the values
    if(count==0): return 0
    score /= count
    return score

def computeDistance(sent1, sent2):
    wn_sent1 = []
    wn_sent2 = []
    for t in sent1.split():
        wn_term = getWnTerm(t)
        if wn_term: wn_sent1.append(wn_term)
    for t in sent2.split():
        wn_term = getWnTerm(t)
        if wn_term: wn_sent2.append(wn_term)

    for tup in it.product(wn_sent1,wn_sent2):
        sim = tup[0].path_similarity(tup[1])

        print(tup, "->", sim)


    
# if __name__ == "__main__":
    # syn = wn.synsets("dog")[0]

    # with open("C:\\Users\\GIORGIO-DESKTOP\\Desktop\\query_res.txt", "w") as f:

        # <https://cso.kmi.open.ac.uk/topics/communication_channels_%28information_theory%29>
        # <http://cso.kmi.open.ac.uk/schema/cso#superTopicOf>
        # <https://cso.kmi.open.ac.uk/topics/antenna_array> .

        # s = URIRef("https://cso.kmi.open.ac.uk/topics/communication_channels_%28information_theory%29")
        # p = URIRef("http://cso.kmi.open.ac.uk/schema/cso#superTopicOf")
        # o = URIRef("https://cso.kmi.open.ac.uk/topics/case-based_reasoning_approaches")

        # n1 = URIRef("https://cso.kmi.open.ac.uk/topics/computer_science")
        # n2 = URIRef("https://cso.kmi.open.ac.uk/topics/artificial_intelligence")
        # n3 = URIRef("https://cso.kmi.open.ac.uk/topics/decision_analysis")

        # paren_topic = g.value(subject=None,predicate=CSO_SUPER_TOPIC,object=o)
        # f.write(str(rdf_bfs((None,CSO_SUPER_TOPIC,paren_topic),max_depth=4, asTree=False)))

        # p = Namespace("http://cso.kmi.open.ac.uk/schema/cso#")
        # t = Namespace("https://cso.kmi.open.ac.uk/topics/")


        # mp = MulPath(e.superTopicOf,OneOrMore)
        # sp = SequencePath(e.superTopicOf)


        # tuples = sp.eval(g,n1,n3)
        
        # tuples = g.triples()

        # for j in tuples:
        #     print(j)
        # print(isInOntology("binary#decision#diagram#(bdd)"))


        # computeDistance("Eventually, a huge cyclone hit the entrance of my house.", "Finally, a massive hurricane attacked my home.")
