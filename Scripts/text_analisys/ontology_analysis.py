import rdflib #requirement
from rdflib import plugin, URIRef
from rdflib.namespace import Namespace
from rdflib.graph import Graph

from queue import Queue

path = "C:\\Users\\GIORGIO-DESKTOP\\Desktop\\CSO.3.1.12.nt"

g = Graph()
g.parse(path, format="nt")

i = 0

def rdf_bfs(triple,max_depth=5):

    Q = Queue()
    visitSet = set()
    resultSet = dict()

    Q.put_nowait(triple)
    visitSet.add(triple)
    resultSet.update({triple: 0})
    
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
                    resultSet.update({t_child: child_lvl})

    return resultSet

CSO_SUPER_TOPIC = URIRef("http://cso.kmi.open.ac.uk/schema/cso#superTopicOf")

with open("C:\\Users\\GIORGIO-DESKTOP\\Desktop\\query_res.txt", "w") as f:
    # for s, p, o in g:
    #     if(i==1): break
    #     i += 1
        # f.write(str(s)+"\n")

    # <https://cso.kmi.open.ac.uk/topics/communication_channels_%28information_theory%29>
    # <http://cso.kmi.open.ac.uk/schema/cso#superTopicOf>
    # <https://cso.kmi.open.ac.uk/topics/antenna_array> .

    s = URIRef("https://cso.kmi.open.ac.uk/topics/communication_channels_%28information_theory%29")
    p = URIRef("http://cso.kmi.open.ac.uk/schema/cso#superTopicOf")
    o = URIRef("https://cso.kmi.open.ac.uk/topics/antenna_arrays")


    f.write(str(rdf_bfs((s,p,o),max_depth=2)))

        ## neighbours
        # for sp, pp, op in g.triples((s,CSO_SUPER_TOPIC,None)):
        #     f.write("\t")
        #     f.write(str(sp)+"\t")
        #     f.write(str(pp)+"\t")
        #     f.write(str(op)+"\t")
        #     f.write("\n")
# g.parse("http://bigasterisk.com/foaf.rdf")


# qres = g.query(
#     """SELECT ?rel ?t2
#        WHERE {
#            csot:automated_pattern_recognition ?rel ?t2 .
#        }""",
#     initNs=dict(
#         csot=Namespace("https://cso.kmi.open.ac.uk/topics/"),
#         csor=Namespace("https://cso.kmi.open.ac.uk/schema/")
#         )
#     ) csor:cso#superTopicOf

# parentTopicQuery = g.query(
#     """SELECT ?superClass
#        WHERE {
#            ?superClass <http://cso.kmi.open.ac.uk/schema/cso#superTopicOf> <https://cso.kmi.open.ac.uk/topics/reinforcement_learning> .
#        }""",
#     initNs=dict(
#         csot=Namespace("https://cso.kmi.open.ac.uk/topics/"),
#         csor=Namespace("https://cso.kmi.open.ac.uk/schema/")
#         )
#     )

    

# p_topic = parentTopicQuery.result[0][0].split("/")[-1]

# print(p_topic)

# childrenTopicsQuery = g.query(
#     '''SELECT ?child
#        WHERE {
#            <https://cso.kmi.open.ac.uk/topics/%s> <http://cso.kmi.open.ac.uk/schema/cso#superTopicOf> ?child .
#        }''' % p_topic,
#     initNs=dict(
#         csot=Namespace("https://cso.kmi.open.ac.uk/topics/"),
#         csor=Namespace("https://cso.kmi.open.ac.uk/schema/")
#         )
#     )

# parentTopicQuery = g.query(
#     """SELECT ?superClass
#        WHERE {
#            ?superClass <http://cso.kmi.open.ac.uk/schema/cso#superTopicOf> <https://cso.kmi.open.ac.uk/topics/reinforcement_learning> .
#        }""",
#     initNs=dict(
#         csot=Namespace("https://cso.kmi.open.ac.uk/topics/"),
#         csor=Namespace("https://cso.kmi.open.ac.uk/schema/")
#         )
#     )

# with open("C:\\Users\\GIORGIO-DESKTOP\\Desktop\\query_res.txt", "w") as f:
    # for row in childrenTopicsQuery.result:
    #     f.write(" ".join([str(x).split("/")[-1] for x in row])+"\n")