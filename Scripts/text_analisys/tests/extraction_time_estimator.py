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
from nltk.corpus import wordnet as wn

# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer

from nltk.parse import stanford
from ontology_analysis import rdf_get_sibligs, isInOntology, getWnTerm, showTree, sentence_similarity, ic_sentence_similarity, order_by_cosine_similarity
from pdf_parsing.paper_to_txt import RepositoryExtractor, RawPaper, PaperSections, removeEOL, removeWordWrap, escape_semicolon
from pdf_parsing.txt2pdf import PDFCreator, Args, Margins

import operator
import json
import itertools

import multiprocessing as mp
import psutil

stopwords_en_set=set(stopwords.words('english'))

os.environ['STANFORD_PARSER'] = '/home/user/gbelli/FDG_Data/models/corenlp400/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/user/gbelli/FDG_Data/models/corenlp400/stanford-parser-4.0.0-models.jar'
os.environ['CLASSPATH'] = '/home/user/gbelli/FDG_Data/models/corenlp400/*'

java_path = "/usr/bin/java"
os.environ['JAVAHOME'] = java_path

nlp = spacy.load("en_core_web_lg")
rake = Rake(max_words=5,min_freq=1,language_code="en")
stan_parser = stanford.StanfordParser(model_path="/home/user/gbelli/FDG_Data/models/corenlp400/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")


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

def replace_multi_regex(text,rep_dict):
    rep_dict = dict((re.escape(k), v) for k, v in rep_dict.items()) 
    pattern = re.compile("|".join(rep_dict.keys()))
    text = pattern.sub(lambda m: rep_dict[re.escape(m.group(0))], text)
    return text

# with open("../datasets/AI_glossary.txt","r") as ai_glossary_fd:
#     field_words = set()
#     for word in ai_glossary_fd.readlines():
#         field_words.add(word[:-1].lower())

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

    ## TODO should implement lemmatization and stemming

    if(not bool(label1) and not bool(label2)): return 0.0
    return len(label1.intersection(label2))/len(label1.union(label2))

def removePunctualization(text):
    return re.sub(r'\?|,|:','',text)

def computeSynonymsDict(paper,topics, synsets_dict):
    res_dict = dict()
    for t in topics:
        for w,lst in synsets_dict.items(): 
            topic_set = set(wn.synsets("_".join(t.split())))
            word_set = set(lst)
            if topic_set and word_set and not topic_set.isdisjoint(word_set):
                item = res_dict.setdefault(t,[[],0])
                item[0].append(w)
                item[1]+=paper.fulltext.lower().count(" "+w.lower()+" ")
    return res_dict

class StructuredPaper():
    def __init__(self,sections,fulltext,info={},max_topics=None,parser=None):
        str_rep = {"ï¬�":"fi","ï¬":"fi","ï¬‚":"fl","ﬁ":"fi","ﬀ":"ff","ﬂ":"fl"}

        self.sections = sections
        # self.fulltext = re.sub("ï¬�|ï¬","fi",fulltext)
        self.fulltext = replace_multi_regex(fulltext,str_rep)
        self.topics = set()
        self.contexts = dict()
        self.proximityFrequences = {}
        self.jcMatrix = dict()
        self.info = info

        self.transformedFullText = ""
        self.transformedTopics = set()

        max_topics = None

        # self.spacy_extractTopics()
        for section in self.sections.keys() :
            # self.sections[section] = re.sub("ï¬�|ï¬","fi",self.sections[section])
            self.sections[section] = replace_multi_regex(self.sections[section],str_rep)
            if self.sections[section] in ["",None]: continue 
            if(fulltext == "" or fulltext is None): self.fulltext += section+"\n"+self.sections[section]+"\n"
            section_sents = self.sections[section].split(". ")
            section_sents_count = len(section_sents)
            offset = 3
            # for idx in range(0,section_sents_count,3):
            #     if(idx+3 > section_sents_count-1): subtext = ". ".join(section_sents[idx:]).lower()
            #     else: subtext = ". ".join(section_sents[idx:idx+offset]).lower()
            #     self.rake_extractTopics(subtext,limit=max_topics)
            
            print("extracting topics in section \"{}\": {}".format(section,timeit.Timer(lambda: self.rake_extractTopics(self.sections[section],limit=max_topics)).repeat(1, 3)))
            # print("extracting topics in section \"{}\": {}".format(section,timeit.Timer(lambda: self.stanford_extractTopics(self.sections[section],parser,limit=max_topics)).repeat(1, 3)))
            # print("extracting topics in section \"{}\": {}".format(section,timeit.Timer(lambda: self.spacy_extractTopics(self.sections[section])).repeat(1, 3)))
            
            # self.stanford_extractTopics(self.sections[section],parser,limit=max_topics)
            # self.spacy_extractTopics(self.sections[section])
        
        self.frequencies = self.computeFrequencies()
        if(max_topics): 
            t_list = sorted(list(self.topics), key= lambda x: self.frequencies.get(x,0),reverse=True)
            self.topics = set(t_list[:max_topics])

    @staticmethod
    def from_raw(rawPaper,max_topics=None,parser=None):
        return StructuredPaper(rawPaper.sections_dict,rawPaper.full_text,info={},max_topics=max_topics,parser=parser)

    @staticmethod
    def from_json(json_path,max_topics=None,parser=None):
        with open(json_path,"r",encoding="utf-8") as json_fd:
            json_string = json_fd.read()
            p = json.loads(json_string)
        
        info = {
            "filename": str(p["name"]),
            "title": str(p["metadata"]["title"]),
            "authors": str(p["metadata"]["authors"]),
            "year": str(p["metadata"]["year"]),
            "references": str(p["metadata"]["references"]),

        }

        if(p["metadata"]["sections"] is None): return None

        sections = {
            PaperSections.PAPER_ABSTRACT: str(p["metadata"]["abstractText"])
        }

        full_text = str(p["metadata"]["abstractText"])+"\n"

        for section in p["metadata"]["sections"]:
            sections.update({str(section["heading"]):str(section["text"])})
            full_text+=str(section["heading"])+"\n"+str(section["text"])+"\n"

        return StructuredPaper(sections,full_text,info=info,max_topics=max_topics,parser=parser)



    def createTree(self,text,parser):
        return parser.raw_parse_sents(text.split(". "))

    def stanford_extractTopics(self,text,parser,limit=None):

        sentences_trees = self.createTree(text,parser)

        for sentenceTree in sentences_trees:
            for t in sentenceTree: root = t

            partial_topics = set()
        
            for child in root.subtrees(filter= lambda t: isSmallestNP(t)):
                partial_topics.add(" ".join(child.leaves()).replace(" -RRB-",")").replace("-LRB- ","("))
                self.topics.add(" ".join(child.leaves()).replace(" -RRB-",")").replace("-LRB- ","("))
                self.transformedTopics.add("#".join(child.leaves()).replace(" -RRB-",")").replace("-LRB- ","("))
                self.contexts["#".join(child.leaves()).replace(" -RRB-",")").replace("-LRB- ","(")] = set()
                
            # print(partial_topics)
        self.cleanRedoundantTopics()
    
    def spacy_extractTopics(self,text):
        doc = nlp(text)
        for token in doc.noun_chunks:
            token = str(token)
            self.topics.add(token)
            self.transformedTopics.add("#".join(token.split()))
            self.contexts["#".join(token.split())] = set()
        self.cleanRedoundantTopics()

    def rake_extractTopics(self,text,limit=None,threshold=5.0):
        keywords = rake.apply(text)
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
        self.transformedFullText = removePunctualization(removeEOL(removeWordWrap(self.fulltext.lower())))
        for topic in self.topics.union(extraTopics):
            t_topic = '#'.join(topic.split())
            self.transformedFullText = self.transformedFullText.replace(topic,t_topic)

    def computeContextAndFrequencies(self,window=5,extraTransformedTopics = set()):
        # create context and compute frequency with window size (only for current sentence)
        allTopics = self.transformedTopics.union(extraTransformedTopics)

        for t_sentence in self.transformedFullText.split(". "):
        
            sentenceWords=t_sentence.split()
           
            for i,word in enumerate(sentenceWords):
                if(word in allTopics):
                    contextWords = sentenceWords[max(0,i-window):i]+sentenceWords[i+1:min(i+window+1,len(sentenceWords))]

                    self.contexts.setdefault(word,set()).update(contextWords)

                    for j in range(i+1,min((i+window+1),len(sentenceWords)-1)):
                        if(sentenceWords[j]!=word and sentenceWords[j] in allTopics):
                            new_val = self.proximityFrequences.get((word,sentenceWords[j]),0)+1
                            self.proximityFrequences.update({(word,sentenceWords[j]): new_val})

    def computeFrequencies(self):
        frequencies = {}
        for topic in self.topics:
            frequency = self.fulltext.count(topic)
            frequencies.update({topic:frequency})
        return frequencies
    
    def getSubstituteConcept(self,focus_topic,alternatives):
        best_candidate = (None, None, -1)
        topic_set = alternatives-self.topics    

        # cosine similaity
        alternative, score = order_by_cosine_similarity(focus_topic,topic_set)[0]
        best_candidate = (focus_topic, alternative, score)

        # # wordnet similarity
        # for topic in topic_set:
        #     t = (focus_topic,topic,sentence_similarity(focus_topic,topic))
        #     if(best_candidate[2]<t[2]): best_candidate = t

        return best_candidate

        
    def getCandicateConcepts(self,focus_topic,alternatives,limit=None):
        distances = []
        topic_set = alternatives-self.topics

        distances = [(alt,score,self.frequencies.get(alt,None)) for alt, score in order_by_cosine_similarity(focus_topic,topic_set)]
        
        # # wordnet similarity
        # full_text = "\n".join([str(k)+"\n"+str(v) for k,v in self.sections.items()])
        # for topic in topic_set:
        #     topic_frequency = self.frequencies.get(focus_topic,None)
        #     distances.append((topic,sentence_similarity(focus_topic,topic),topic_frequency))
        #     # distances.append((topic,ic_sentence_similarity(focus_topic,topic,full_text),topic_frequency))
        # distances.sort(reverse=True,key= operator.itemgetter(2,1))
        if(limit): distances = distances[0:limit]
        return (focus_topic,distances)

    def generatePdf(self,args={},layout=None):

        args = Args(args)

        creator = PDFCreator(args, Margins(
            args.margin_right,
            args.margin_left,
            args.margin_top,
            args.margin_bottom))

        creator.generate()


class Repository():
    def __init__(self,paper_list = []):
        self.papers = [paper for paper in paper_list if paper] 

        self.generalContexts = dict()
        self.generalTopics = set()
        self.generalTransformedTopics = set()
        self.jcMatrix = dict()

        print("\tinitTopics...",end="")
        self.initTopics()
        print("\t[done]")
        
        # print("\tprepare for JC...",end="")
        # self.prepareForJC()       
        # print("\t[done]")

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
                self.generalContexts.setdefault(k,paper.contexts[k]).update(paper.contexts[k])

    def replacePaperTopics(self,paper):
        paper.replaceTopicsInText(extraTopics=self.generalTopics)
        
    def computePaperContext(self,paper):
        paper.computeContextAndFrequencies(extraTransformedTopics=self.generalTransformedTopics)

    def replacePapersTopics(self):

        # with mp.Pool(processes=None) as pool:
        #         results = pool.map(self.replacePaperTopics,self.papers)

        for paper in self.papers:
            paper.replaceTopicsInText(extraTopics=self.generalTopics)

    def computePapersContext(self):

        # with mp.Pool(processes=None) as pool:
        #         results = pool.map(self.computePaperContext,self.papers)

        for paper in self.papers:
            paper.computeContextAndFrequencies(extraTransformedTopics=self.generalTransformedTopics)

        # for paper in self.papers:
        #     paper.computeContextAndFrequencies(extraTransformedTopics=self.generalTransformedTopics)
        #     print("====FULLTEXT====\n",)
        #     print(removePunctualization(removeEOL(removeWordWrap(paper.fulltext.lower()))))
        #     print("====FULLTEXT TRANS====\n",)
        #     print(paper.transformedFullText)
        #     print("====CONTEXTS TRANS====\n",)
        #     for k,v in paper.contexts.items():
        #         print(k,"->",v)
        #     print("====END====\n",)
            

    def prepareForJC(self):
        print("\treplacePapersTopics...",end="")
        self.replacePapersTopics()
        print("\t[done]")
        print("\tcomputePapersContext...",end="")
        self.computePapersContext()
        print("\t[done]")
        print("\tinitGeneralContexts...",end="")
        self.initGeneralContexts()
        print("\t[done]")

    # compute repo jaccard [slow]
    def computeJC(self):
        for tp in self.generalTransformedTopics:
            for tpp in self.generalTransformedTopics:
                if(tp != tpp):
                    jc = forge_jaccard_distance(self.generalContexts[tp],self.generalContexts[tpp])
                    if(jc != 0 and jc > (self.jcMatrix.get(tp,(tp,0)))[1]): self.jcMatrix[tp] = (tpp,jc)

    def computeJCforPaper(self,paper):
        jcMatrix = dict()

        paper.replaceTopicsInText(extraTopics=self.generalTopics)
        paper.computeContextAndFrequencies(extraTransformedTopics=self.generalTransformedTopics)

        for tp in paper.transformedTopics:
            for tpp in self.generalTransformedTopics:
                if(tp != tpp):
                    jc = forge_jaccard_distance(paper.contexts[tp],self.generalContexts[tpp])
                    if(jc != 0 and jc > (paper.jcMatrix.get(tp,(tp,0)))[1]):
                        paper.jcMatrix[tp] = (tpp,jc)
        return paper.jcMatrix
        
def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = f.read().split("\f\n")
    return rows

global_id = [-1,]

def createPaper(idx,path,max_topics,parser):
    p = StructuredPaper.from_json(path,max_topics=max_topics,parser=parser)
    if(psutil.Process().cpu_num() == 0):
        print('\rcreating paper: {}                       '.format(idx),end="")
    return p

def findSubstitutions(idx,paper,focus_topic,repo,max_candidates):
    if(psutil.Process().cpu_num() == 0):
        print('\rfinding substitutions: {}                       '.format(idx),end="")
    topic, candidates = paper.getCandicateConcepts(focus_topic,repo.generalTopics)
    
    return (topic,[c[0] for c in candidates[:max_candidates]])

def main(args):
    
    with open(args.inFile,mode="r") as in_fd:

        file_text =  in_fd.read()

    sections = {
        PaperSections.PAPER_ABSTRACT: "",
        PaperSections.PAPER_INTRO : file_text, 
        PaperSections.PAPER_CORPUS: "",
        PaperSections.PAPER_CONCLUSION: "",
    }

    raw = RawPaper.fromSections(sections)

    paper = StructuredPaper(raw.sections_dict,raw.full_text,parser=stan_parser)

    print(paper.topics)
    
    return    
    

if __name__ == "__main__":
    import argparse
    import timeit

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "inFile",
        help="text file to be processed",
        default=None,
        type=str,
    )

    args = arg_parser.parse_args()

    print(timeit.Timer(lambda: main(args)).repeat(1, 1))