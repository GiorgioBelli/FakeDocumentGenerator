# import stanza
# from stanza.server import CoreNLPClient
# from nltk.tree import Tree
# from itertools import chain  

# import os
# os.environ["CORENLP_HOME"] = "/home/user/gbelli/FakeDocumentGenerator/models/corenlp40"

# def extract_phrase1(tree_str, label):
#     phrases = []
#     trees = Tree.fromstring(tree_str)
#     for tree in trees:
#       phrases.extend([' '.join(subtree.leaves()) for subtree in tree.subtrees() if subtree.label() == label])
#     return phrases

# def stanza_phrases(matches,pattern):                                                                                                                                              
#     Nps = []                                                                                                                                                                
#     #for match in matches:                                                                                                                                                  
#     for its in matches['sentences']:                                                                                                                                          
#         Nps.extend(extract_phrase1('(ROOT\n'+ values['match']+')', pattern)  for _,values in its.items())
#     return set(list(chain(*Nps)))     

# def noun_phrases(_client, _text, _annotators=None):
#     pattern = 'NP'
#     matches = _client.tregex(_text,pattern,annotators=_annotators)
#     print("\n".join(["\t"+sentence[match_id]['spanString'] for sentence in matches['sentences'] for match_id in sentence]))

# if __name__ == "__main__":

#     # English example
#     with CoreNLPClient(classpath="/home/user/gbelli/FakeDocumentGenerator/models/corenlp40/*",timeout=30000, memory='16G') as client:
        
#         englishText = '''The year of 2006 was exceptionally cruel to me – almost all of my papers submitted for that year conferences have been rejected.'''
        
#         noun_phrases(client,englishText,_annotators="tokenize,ssplit,pos,lemma,parse")


# =================

# from nltk.parse.corenlp import CoreNLPParser, CoreNLPServer
# import os

# os.environ['STANFORD_PARSER'] = '/home/user/gbelli/FakeDocumentGenerator/models/corenlp40/stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = '/home/user/gbelli/FakeDocumentGenerator/models/corenlp40/stanford-parser-4.0.0-models.jar'

# java_path = "/usr/bin/java"
# os.environ['JAVAHOME'] = java_path

# _JAR_PATH = "/home/user/gbelli/FakeDocumentGenerator/models/corenlp40/stanford-parser.jar"
# _MODEL_PATH = '/home/user/gbelli/FakeDocumentGenerator/models/corenlp40/stanford-parser-4.0.0-models.jar'

# def parse_sents_example():
#     sents = [['Latest', 'corporate', 'unbundler', 'reveals', 'etc', 'talks', 'to', 'Frank', 'Kane'],
#              ['By', 'FRANK', 'KANE'],
#              ['IT', 'SEEMS', 'that', 'Roland', 'Franklin', 'etc', 'packaging', 'group', 'DRG', '.'],
#              ['He', 'has', 'not', 'properly', 'investigated', 'the', 'target', "'s", 'dining', 'facilities', '.']]
#     parse_trees = []
#     server = CoreNLPServer(path_to_jar=_JAR_PATH, path_to_models_jar=_MODEL_PATH)
#     server.start()
#     parser = CoreNLPParser(url='http://localhost:9000')

#     # Default
#     parse_results = parser.parse_sents(sents)
#     for parse in parse_results:
#         parse_trees.append(next(parse))
#     print(len(parse_trees))   # Outputs 2 - first 3 sentences are concatenated

#     parse_trees.clear()

#     # Using correct property
#     parse_results = parser.parse_sents(sents, properties={'ssplit.eolonly': 'true'})
#     for parse in parse_results:
#         parse_trees.append(next(parse))
#     print(len(parse_trees))   # Outputs 4

#     server.stop()

# parse_sents_example()

# =============================



from nltk.parse.corenlp import CoreNLPServer
import os

os.environ['STANFORD_PARSER'] = '/home/user/gbelli/FakeDocumentGenerator/models/corenlp392/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/user/gbelli/FakeDocumentGenerator/models/corenlp392/stanford-parser-3.9.2-models.jar'
os.environ['CLASSPATH'] = '/home/user/gbelli/FakeDocumentGenerator/models/corenlp392/*'

java_path = "/usr/bin/java"
os.environ['JAVAHOME'] = java_path

# The server needs to know the location of the following files:
#   - stanford-corenlp-X.X.X.jar
#   - stanford-corenlp-X.X.X-models.jar
# STANFORD = '/home/user/gbelli/FakeDocumentGenerator/models/corenlp392/'

# # Create the server
# server = CoreNLPServer(
#    os.path.join(STANFORD, "stanford-parser.jar"),
#    os.path.join(STANFORD, "stanford-parser-3.9.2-models.jar"),    
# )

# # Start the server in the background
# server.start()

from  nltk.parse.corenlpnltk.pa  import CoreNLPParser

parser = CoreNLPParser()

parse = next(parser.raw_parse("I put the book in the box on the table."))

print(parse)