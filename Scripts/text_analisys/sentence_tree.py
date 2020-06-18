# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:53:30 2020

@author: GIORGIO-DESKTOP
"""

import sys
import nltk #requiment
import pandas #requirement
from multi_rake import Rake #requirement
import spacy #requirement
import re
import os

from nltk.tree import Tree
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer

# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer

from nltk.parse import stanford
from ontology_analysis import rdf_get_sibligs, isInOntology, getWnTerm, showTree, sentence_similarity

stopwords_en_set=set(stopwords.words('english'))


nlp = spacy.load("en_core_web_sm")
rake = Rake(language_code="en")

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # if( iteration == total): print()


with open("C:\\Users\GIORGIO-DESKTOP\\Documents\\Universita\\Tesi\\datasets\\AI_glossary.txt","r") as ai_glossary_fd:
    field_words = set()
    for word in ai_glossary_fd.readlines():
        field_words.add(word[:-1].lower())

def isSmallestNP(tree):
    adj_factor = 1 if tree.label()=="NP" else 0
    return (tree.label() == "NP") and (len(list(tree.subtrees(filter= lambda t: t.label()=="NP")))-adj_factor == 0)
    
def discardTopics(concepts_set):
    clean_set = set()
    for c in concepts_set:
        canAdd = True
        for cp in concepts_set:
            if((c.lower() != cp.lower()) and (c.lower() in cp.lower())): canAdd = False
        if(canAdd):
            stopwords_clean = list(filter(lambda w: (not w.lower() in stopwords_en_set,c.split()) or w.lower() in field_words,c.split()))
            if(stopwords_clean): clean_set.add(" ".join(stopwords_clean))
    return clean_set

def forge_jaccard_distance(label1, label2):
    """Distance metric comparing set-similarity.

    """
    if(not bool(label1) and not bool(label2)): return 0.0
    return len(label1.intersection(label2))/len(label1.union(label2))

class Paper():
    def __init__(self,text,fulltext,parser):
        self.text = text.replace("ï¬�","fi")
        self.fulltext = fulltext.replace("ï¬�","fi")
        self.topics = set()
        self.contexts = dict()
        self.proximityFrequences = {}
        self.jcMatrix = dict()
        # self.sentencesTrees = self.createTree(parser)

        self.transformedFullText = ""
        self.transformedTopics = set()

        # self.spacy_extractTopics()
        self.rake_extractTopics()

    def createTree(self,parser):
        return parser.raw_parse_sents(self.text.split(". "))

    def stanford_extractTopics(self):
        for sentenceTree in self.sentencesTrees:
            for t in sentenceTree: root = t

            partial_topics = set()
        
            for child in root.subtrees(filter= lambda t: isSmallestNP(t)):
                partial_topics.add(" ".join(child.leaves()).replace(" -RRB-",")").replace("-LRB- ","("))
                self.topics.add(" ".join(child.leaves()).replace(" -RRB-",")").replace("-LRB- ","("))
                self.transformedTopics.add("#".join(child.leaves()).replace(" -RRB-",")").replace("-LRB- ","("))
                self.contexts["#".join(child.leaves()).replace(" -RRB-",")").replace("-LRB- ","(")] = set()
                
            # print(partial_topics)
        self.cleanRedoundantTopics()
    
    def spacy_extractTopics(self):
        doc = nlp(self.text)
        for token in doc.noun_chunks:
            token = str(token)
            self.topics.add(token)
            self.transformedTopics.add("#".join(token.split()))
            self.contexts["#".join(token.split())] = set()
        self.cleanRedoundantTopics()

    def rake_extractTopics(self,threshold=5.0):
        keywords = rake.apply(self.text)
        for keyword in keywords:
            k_rating = keyword[1]
            k_text = keyword[0]
            if(k_rating > threshold):
                self.topics.add(k_text)
                self.transformedTopics.add("#".join(k_text.split()))
                self.contexts["#".join(k_text.split())] = set()


    def cleanRedoundantTopics(self):
        self.topics = discardTopics(self.topics)

    def replaceTopicsInText(self,extraTopics = set()):
        self.transformedFullText = self.fulltext
        for topic in self.topics.union(extraTopics):
            t_topic = '#'.join(topic.split())
            self.transformedFullText = self.transformedFullText.replace(topic,t_topic)

    def computeContextAndFrequencies(self,window=5,extraTransformedTopics = set()):
        for t_sentence in self.transformedFullText.split(". "):
        
            sentenceWords=t_sentence.split()
           
            # create context and compute frequency with window size (only for current sentence)
            allTopics = self.transformedTopics.union(extraTransformedTopics)

            for i,word in enumerate(sentenceWords):
                if(word in allTopics):
                    contextWords = sentenceWords[max(0,i-window):i]+sentenceWords[i+1:min(i+window+1,len(sentenceWords))]
                    if(word not in self.contexts.keys()): self.contexts[word] = set(contextWords)
                    else: self.contexts[word].update(contextWords)
                    for j in range(i+1,min((i+window+1),len(sentenceWords)-1)):
                        if(sentenceWords[j]!=word and sentenceWords[j] in allTopics):
                            new_val = self.proximityFrequences.get((word,sentenceWords[j]),0)+1
                            self.proximityFrequences.update({(word,sentenceWords[j]): new_val}) 
    
    def getSubstituteConcept(self,focus_topic,alternatives):
        best_candidate = (None, None, -1)
        topic_set = alternatives-self.topics    
        for topic in topic_set:
            t = (focus_topic,topic,sentence_similarity(focus_topic,topic))
            if(best_candidate[2]<t[2]): best_candidate = t

        return best_candidate

        
    def getCandicateConcepts(self,focus_topic,alternatives):
        distances = []
        topic_set = alternatives-self.topics
        for topic in topic_set:
            distances.append((focus_topic,topic,sentence_similarity(focus_topic,topic)))
        distances.sort(reverse=True,key=lambda x: x[2])
        return distaces




class Repository():
    def __init__(self,paper_list = []):
        self.papers = paper_list

        self.generalContexts = dict()
        self.generalTopics = set()
        self.generalTransformedTopics = set()
        self.jcMatrix = dict()

        print("\tinitTopics...",end="")
        self.initTopics()
        print("\t[done]")
        print("\treplacePapersTopics...",end="")
        self.replacePapersTopics()
        print("\t[done]")
        print("\tcomputePapersContext...",end="")
        self.computePapersContext()
        print("\t[done]")
        print("\tinitGeneralContexts...",end="")
        self.initGeneralContexts()
        print("\t[done]")
        # print("\tcomputeJC...",end="")
        # self.computeJC()
        # print("\t[done]")
        
    def initTopics(self):
        for paper in self.papers:
            self.generalTopics = self.generalTopics.union(paper.topics)
            self.generalTransformedTopics = self.generalTransformedTopics.union(paper.transformedTopics)
    
    def initGeneralContexts(self):
        for paper in self.papers:
            keys = self.generalContexts.keys()
            for k in paper.contexts.keys():
                if k in keys: self.generalContexts[k].update(paper.contexts[k])
                else: self.generalContexts[k] = paper.contexts[k]

    def replacePapersTopics(self):
        for paper in self.papers:
            paper.replaceTopicsInText(extraTopics=self.generalTopics)

    def computePapersContext(self):
        for paper in self.papers:
            paper.computeContextAndFrequencies(extraTransformedTopics=self.generalTransformedTopics)

    # compute repo jaccard [slow]
    def computeJC(self):
        for tp in self.generalTransformedTopics:
            for tpp in self.generalTransformedTopics:
                if(tp != tpp):
                    jc = forge_jaccard_distance(self.generalContexts[tp],self.generalContexts[tpp])
                    if(jc != 0 and jc > (self.jcMatrix.get(tp,(tp,0)))[1]): self.jcMatrix[tp] = (tpp,jc)

    def computeJCforPaper(self,paper):
        jcMatrix = dict()
        for tp in paper.transformedTopics:
            for tpp in self.generalTransformedTopics:
                if(tp != tpp):
                    jc = forge_jaccard_distance(paper.contexts[tp],self.generalContexts[tpp])
                    if(jc != 0 and jc > (paper.jcMatrix.get(tp,(tp,0)))[1]): 
                        paper.jcMatrix[tp] = (tpp,jc)
        return paper.jcMatrix
        

def main():

    os.environ['STANFORD_PARSER'] = 'C:\\Users\\GIORGIO-DESKTOP\\Documents\\Universita\\FakeDocumentGenerator\\models\\stanford-parser-full-2018-10-17\\stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = 'C:\\Users\\GIORGIO-DESKTOP\\Documents\\Universita\\FakeDocumentGenerator\\models\\stanford-parser-full-2018-10-17\\stanford-parser-3.9.2-models.jar'
    
    java_path = "C:\\Program Files\\Java\\jre-10.0.1\\bin\\java.exe"
    os.environ['JAVAHOME'] = java_path

    parser = stanford.StanfordParser(model_path="C:\\Users\\GIORGIO-DESKTOP\\Documents\\Universita\\FakeDocumentGenerator\\models\\stanford-parser-full-2018-10-17\\models\\englishPCFG.ser.gz")
    
    csv_path = "C:\\Users\\GIORGIO-DESKTOP\\Documents\\Universita\\FakeDocumentGenerator\\datasets\\arxiv\\4500_summaries_trainingSet.csv"
    csv_path = "C:\\Users\\GIORGIO-DESKTOP\\Desktop\\intros.csv"
    dataset = pandas.read_csv(csv_path, delimiter = '\f\n', engine="python")

    # print(dataset.head(5))
    
    # exit(0)
    
    
    
    # document_text = '''The intelligence community relies on human-machine-based analytic strategies that 1) access and integrate vast amounts of information from disparate sources, 2) continuously process this information, so that, 3) a maximally comprehensive understanding of world actors and their behaviors can be developed and updated. Herein we describe an approach to utilizing outcomes-based learning (OBL) to support these efforts that is based on an ontology of the cognitive processes performed by intelligence analysts.'''
    # document_text = '''The close connection between reinforcement learning (RL) algorithms and dynamic programming algorithms has fueled research on RL within the machine learning community. Yet, despite increased theoretical understanding, RL algorithms remain applicable to simple tasks only. In this paper I use the abstract framework afforded by the connection to dynamic programming to discuss the scaling issues faced by RL researchers. I focus on learning agents that have to learn to solve multiple structured RL tasks in the same environment. I propose learning abstract environment models where the abstract actions represent “intentions” of achieving a particular state. Such models are variable temporal resolution models because in different parts of the state space the abstract actions span different number of time steps. The operational definitions of abstract actions can be learned incrementally using repeated experience at solving RL tasks. I prove that under certain conditions solutions to new RL tasks can be found by using simulated experience with abstract actions alone.'''
    # document_text1 = '''A surfactant composition for agricultural chemicals containing fatty acid polyoxyalkylene alkyl ether expressed by the following formula (I): ABCD.'''
    
    paper_list = []
    paper_count = len(dataset['intros'])
    printProgressBar(0,paper_count,prefix="Creating papers [{},{}]".format(0,paper_count),suffix="",length=50)
    for i,entry in enumerate(dataset['intros']):
        # if(i==100): break
        paper_list.append(Paper(entry,entry,parser))
        printProgressBar(i+1,paper_count,prefix="Creating papers [{},{}]".format(i+1,paper_count),suffix="",length=50)

    print("\ncreating repo...")
    repo = Repository(paper_list=paper_list)
    print()

    out_file = "C:\\Users\\GIORGIO-DESKTOP\\Desktop\\result.out"

    p_test = paper_list[-3]

    
    # found = []
    # for t in p_test.transformedTopics:
    #     isInOnt = isInOntology(t)
    #     if(isInOnt):
    #         for term in t.split("#"):
    #             showTree(getWnTerm(term))
    
    focus_topic = p_test.topics.pop()
    print("finding substitution for: ",focus_topic)
    distances = []

    treplace = None
    # for topic in p_test.topics:
    #     if isInOntology(topic,space_separator=" "):
    #         print("ontology concept: "+topic)
    #         siblings = rdf_get_sibligs(topic)
    #         print(siblings)
    #         treplace=max(siblings.keys(),key=lambda x: sentence_similarity(topic,x))
    #         break

    substitutions = []
    num_concepts = len(p_test.topics)
    printProgressBar(0,num_concepts,prefix="Finding substitution [{}/{}]".format(0,num_concepts),suffix="",length=50)

    for i,focus_topic in enumerate(p_test.topics):
        printProgressBar(i+1,num_concepts,prefix="Finding substitutions [{}/{}]".format(i+1,num_concepts),suffix="computing: "+focus_topic,length=50)
        substitutions.append(p_test.getSubstituteConcept(focus_topic,repo.generalTopics))


    newIntro = p_test.text
    for old, new, score in substitutions:
        newIntro = newIntro.replace(old,new)

    with open(out_file,"wb") as result:
        print("writing results...",end="")

        text = ""

        text += "="*20+"KEYWORDS"+"="*20+"\n"
        text += "\n"+str(p_test.topics)
        text += "\n"+"="*40+"\n"

        # text += "="*20+"REPLACEMENT"+"="*20+"\n"
        # text += "\n"+"\n".join([str(s) for s in distances[1:5]])
        # text += "\n"
        # text += "\n"+str(treplace)
        # text += "\n"+"="*40+"\n"

        matrix = repo.computeJCforPaper(p_test)
        l = [(k,matrix[k][0],matrix[k][1])for k in matrix.keys()]
        l.sort(reverse=True,key=(lambda x: x[2]))
        text += "="*20+"JC"+"="*20+"\n"
        text += "\n"+str(l)
        text += "\n"+"="*40+"\n"

        text += "="*20+"ORIGINAL INTRO"+"="*20+"\n"
        text += p_test.text
        text += "\n"+"="*40+"\n"

        text += "="*20+"NEW INTRO"+"="*20+"\n"
        text += newIntro
        text += "\n"+"="*40+"\n"

        result.write(text.encode("utf-8"))
        print("\t[done]")
    
    return    
    

import timeit

print(timeit.Timer(main).repeat(1, 1))