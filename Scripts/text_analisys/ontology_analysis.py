from rdflib.graph import Graph #requirement

path = "C:\\Users\\GIORGIO-DESKTOP\\Desktop\\CSO.3.1.12.nt"

g = Graph()
g.parse(path, format="nt")
# g.parse("http://bigasterisk.com/foaf.rdf")

import rdflib
from rdflib import plugin
from rdflib.namespace import Namespace

qres = g.query(
    """SELECT ?rel ?t2
       WHERE {
           <https://cso.kmi.open.ac.uk/topics/computer_science> ?rel ?t2 .
       }""",
    initNs=dict(
        csot=Namespace("https://cso.kmi.open.ac.uk/topics/"),
        csor=Namespace("https://cso.kmi.open.ac.uk/schema/")
        )
    )

# qres = g.query(
# """SELECT DISTINCT ?aname ?bname
#     WHERE {
#         ?a foaf:knows ?b .
#         ?a foaf:name ?aname .
#         ?b foaf:name ?bname .
#     }""",
# initNs=dict(
#     foaf=Namespace("http://xmlns.com/foaf/0.1/")))

with open("C:\\Users\\GIORGIO-DESKTOP\\Desktop\\query_res.txt", "w") as f:
    for row in qres.result:
        rel = str(row[0]).split("/")[-1]
        t2 = str(row[1]).split("/")[-1]
        f.write("computer_science {} {}\n".format(rel,t2))