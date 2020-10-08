# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:53:30 2020

@author: GIORGIO-DESKTOP
"""

import sys
import nltk #requiment
import pandas #requirement
from multi_rake import Rake as mRake #requirement
import spacy #requirement
import re
import os

import BigHugeThesaurus.BigHugeThesaurus as bgThesaurus

from nltk.tree import Tree
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from nltk.parse import stanford
from ontology_analysis import rdf_get_sibligs, isInOntology, getWnTerm, showTree, sentence_similarity, ic_sentence_similarity, order_by_cosine_similarity
from pdf_parsing.paper_to_txt import RepositoryExtractor, RawPaper, PaperSections, removeEOL, removeWordWrap, escape_semicolon
from pdf_parsing.txt2pdf import PDFCreator, Args, Margins

import operator
import json
import itertools

import multiprocessing as mp
import psutil

# from rake_nltk import Rake as nltkRake
from multi_rake.stopwords import STOPWORDS

from nltk.stem import PorterStemmer

stopwords_en_set=set([*STOPWORDS.get("en"),"•"])

global_jc_matrix = dict()

t_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# nlp = spacy.load("en_core_web_sm")
m_rake = mRake(max_words=5,min_freq=2,language_code="en")
# nltk_rake = nltkRake(min_length=2, max_length=5,stopwords=stopwords_en_set)

stemmer = PorterStemmer()

THESAURUS_API_CONFIG = bgThesaurus.ApiConfig(
        "https://words.bighugelabs.com",
        "api/2",
        "f5acf68d71dad138a4374ec8e7c3522a", #proavston
        # "461035564047c36ddca0468beda8a0a4", #sap
    )

THESAURUS_WEBSITE_CONFIG = bgThesaurus.ApiConfig(
        "https://tuna.thesaurus.com/relatedWords/",
        "",
        "",
    )

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

def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def replace_multi_regex(text,rep_dict):
    rep_dict = dict((re.escape(k), v) for k, v in rep_dict.items()) 
    pattern = re.compile("|".join(rep_dict.keys()))
    text = pattern.sub(lambda m: rep_dict[re.escape(m.group(0))], text)
    return text

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
    syn_items = synsets_dict.items()
    added_synonyms = []
    for t in topics:
        for w,lst in syn_items: 
            if w in topics or w in added_synonyms: 
                continue
            topic_set = set(wn.synsets("_".join(t.split())))
            word_set = set(lst)
            intersec =  topic_set.intersection(word_set)

            if topic_set and word_set and not topic_set.isdisjoint(word_set) :
                int_len = len(intersec)/min(len(topic_set),len(word_set))
                item = res_dict.setdefault(t,[[],0])
                item[0].append(w)
                added_synonyms.append(w)
                item[1] += paper.fulltext.lower().count(" "+w.lower()+" ")*(int_len)
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

        self.scores = {}

        # self.spacy_extractTopics()
        for section in self.sections.keys() :
            # self.sections[section] = re.sub("ï¬�|ï¬","fi",self.sections[section])
            self.sections[section] = replace_multi_regex(self.sections[section],str_rep)
            if(fulltext == "" or fulltext is None): self.fulltext += section+"\n"+self.sections[section]+"\n"
            section_sents = self.sections[section].split(". ")
            section_sents_count = len(section_sents)
            offset = 3
            self.multi_rake_extractTopics(self.sections[section],limit=max_topics)
            # self.nltk_rake_extractTopics(self.sections[section],limit=max_topics)
            # self.stanford_extractTopics(self.sections[section],parser,limit=max_topics)
            # self.spacy_extractTopics(self.sections[section])
        
        # self.frequencies = self.computeFrequencies()
        if(max_topics): 
            t_list = sorted(list(self.topics), key= lambda x: self.scores.get(x,0),reverse=True)
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

    def multi_rake_extractTopics(self,text,limit=None,threshold=0.0):
        keywords = m_rake.apply(text)
        for k_text,k_rating in keywords:
            if(k_rating > threshold):
                self.topics.add(k_text)
                self.scores[k_text] = k_rating
                self.transformedTopics.add("#".join(k_text.split()))
                self.contexts["#".join(k_text.split())] = set()

    def nltk_rake_extractTopics(self,text,limit=None,threshold=0.0):
        nltk_rake.extract_keywords_from_text(text)

        extracted = nltk_rake.get_ranked_phrases_with_scores()

        for k_rating,k_text in extracted:
            if(k_rating > threshold):
                self.topics.add(k_text)
                self.scores[k_text] = k_rating
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
                    contextWords = [stemmer.stem(w.lower()) for w in contextWords]
                    contextWords = set(contextWords) - stopwords_en_set
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

        return best_candidate

        
    def getCandidateConcepts(self,focus_topic,alternatives,limit=None):
        distances = []
        topic_set = alternatives-self.topics

        distances = [(alt,score,self.scores.get(alt,0)) for alt, score in order_by_cosine_similarity(focus_topic,topic_set)]
                
        if(limit): distances = distances[0:limit]
        return (focus_topic,distances)

    def alteredCandidateConcepts(self,thesaurus,focus_topic):
        assert isinstance(thesaurus,bgThesaurus.Thesaurus)

        words = focus_topic.split(" ")

        res = []
        for idx,syns in enumerate(map(lambda x: thesaurus.synonyms(x),words)):
            for syn in syns:
                res.append(" ".join([" ".join(words[:idx]),syn," ".join(words[idx+1:])]))
        
        return res



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

        print("\tinitTopics...",flush=True,end="")
        self.initTopics()
        print("\t[done]")
        
        # print("\tprepare for JC...",flush=True,end="")
        # self.prepareForJC()       
        # print("\t[done]")

        ## do not use this low function
        # print("\tcomputeJC...",flush=True,end="")
        # self.computeJC()
        # print("\t[done]")
        
    def initTopics(self):
        for paper in self.papers:
            self.generalTopics = self.generalTopics.union(paper.topics)
            self.generalTransformedTopics = self.generalTransformedTopics.union(paper.transformedTopics)
        
    def initGeneralContexts(self):
        for paper in self.papers:
            for k in paper.contexts.keys():
                self.generalContexts.setdefault(k,list()).append(paper.contexts[k])

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
            

    def prepareForJC(self):
        print("\treplacePapersTopics...",flush=True,end="")
        self.replacePapersTopics()
        print("\t[done]")
        print("\tcomputePapersContext...",flush=True,end="")
        self.computePapersContext()
        print("\t[done]")
        print("\tinitGeneralContexts...",flush=True,end="")
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

        t_topics_count = len(paper.transformedTopics)

        printProgressBar(0,t_topics_count,prefix="Computing JC for paper [{},{}]".format(0,t_topics_count),suffix="",length=50)
        i = 0
        for tp in paper.transformedTopics:
            i+=1
            ctx = paper.contexts[tp]
            for tpp in self.generalTransformedTopics:
                if(tp != tpp):
                    best = 0
                    for ctx_p in self.generalContexts[tpp]:
                        jc = forge_jaccard_distance(ctx,ctx_p)
                        if(best<jc): best = jc
                    if(best > (paper.jcMatrix.get(tp,(tp,0))[1])): paper.jcMatrix[tp] = (tpp,best)
            printProgressBar(i,t_topics_count,prefix="Computing JC for paper [{},{}]".format(i,t_topics_count),suffix="",length=50)
        return paper.jcMatrix
        
def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = f.read().split("\f\n")
    return rows

global_id = [-1,]

def createPaper(idx,path,max_topics,parser):
    p = StructuredPaper.from_json(path,max_topics=max_topics,parser=parser)
    if(psutil.Process().cpu_num() == 0):
        print('\rcreating paper: {}                       '.format(idx),flush=True,end="")
    return p

def findSubstitutions(idx,paper,focus_topic,repo,max_candidates):
    cpu_num = psutil.Process().cpu_num()
    if(cpu_num == 0):
        str_list = "\rStart subtitutuon search... ["+str(max(t_list))+"]    "
    t_list[cpu_num] = idx
    topic, candidates = paper.getCandidateConcepts(focus_topic,repo.generalTopics)
    
    return (topic,[c[0] for c in candidates[:max_candidates]])

def main(args):


    os.environ['STANFORD_PARSER'] = '/home/user/gbelli/FDG_Data/models/corenlp400/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = '/home/user/gbelli/FDG_Data/models/corenlp400/stanford-parser-4.0.0-models.jar'
    os.environ['CLASSPATH'] = '/home/user/gbelli/FDG_Data/models/corenlp400/*'
    
    java_path = "/usr/bin/java"
    os.environ['JAVAHOME'] = java_path

    stan_parser = stanford.StanfordParser(model_path="/home/user/gbelli/FDG_Data/models/corenlp400/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

    # parser = None

    max_topics = int(args.extracted_topics_limit)
    csv_datasets_dir = args.csv_datasets_dir
    json_papers_dir = args.json_papers_dir
    pdf_papers_dir = args.pdf_papers_dir

    if csv_datasets_dir:
        abstract_csv = load_csv(os.path.join(csv_datasets_dir,"abstract.csv"))
        introduction_csv = load_csv(os.path.join(csv_datasets_dir,"introduction.csv"))
        corpus_csv = load_csv(os.path.join(csv_datasets_dir,"corpus.csv"))
        conclusion_csv = load_csv(os.path.join(csv_datasets_dir,"conclusion.csv"))

        raw_papers = []

        limit = int(args.limit) if args.limit else 0
        dataset_len = min(limit,min(len(abstract_csv), len(introduction_csv), len(corpus_csv), len(conclusion_csv)))
        
        for idx in range(dataset_len):
            sections = {
                PaperSections.PAPER_ABSTRACT : abstract_csv[idx],
                PaperSections.PAPER_INTRO : introduction_csv[idx],
                PaperSections.PAPER_CORPUS : corpus_csv[idx],
                PaperSections.PAPER_CONCLUSION : conclusion_csv[idx]
            }
            raw_papers.append(RawPaper.fromSections(sections))
    elif pdf_papers_dir: # read pdf repo and generates papers

        csv_dir = args.csv_dir

        repo_extr = RepositoryExtractor(pdf_papers_dir)

        limit = int(args.limit) if args.limit else None

        if(args.mp):
            print("starting extraction... ",flush=True,end="")
            workers = None if args.mp == "all" else int(args.mp)
            failed = repo_extr.extractMP(limit=limit,processes=workers)
            print("\t[done] [{} fails]".format(failed))
        else: 
            repo_extr.extract(limit=limit)


        if(csv_dir):
            print("starting csv creation... ",flush=True,end="")
            repo_extr.exportSections(csv_dir,sections=PaperSections.as_list())
            print("\t[done]")


        for i,p in enumerate(repo_extr.papers):
            p.extract_sections()
            if(args.dump_raw_papers):
                p.dump(outPath=os.path.join(args.dump_raw_papers,"paper-rawtest-%s.txt"%i))

        raw_papers = repo_extr.papers
        

    paper_list = []
    
    if json_papers_dir: #create directly structured papers from json
        limit = int(args.limit) if args.limit else None
        files = os.listdir(json_papers_dir)[:limit]
        paper_count = len(files)

        inputs = [(idx,os.path.join(json_papers_dir,json_path),max_topics,stan_parser) for idx,json_path in enumerate(files)]
        
        if(args.mp):
            workers = None if args.mp == "all" else int(args.mp)
            with mp.Pool(processes=workers) as pool:
                results = pool.starmap(createPaper,inputs)
            print('\rcreating paper: {}\t[done]             '.format(len(results)))
            for paper in results:
                if paper: paper_list.append(paper)
        else:
            printProgressBar(0,paper_count,prefix="Creating papers [{},{}]".format(0,paper_count),suffix="",length=50)
            for i,json_path in enumerate(files):
                # if(i==250): break
                
                paper = StructuredPaper.from_json(os.path.join(json_papers_dir,json_path),max_topics=max_topics,parser=stan_parser)
                if paper: paper_list.append(paper)

                printProgressBar(i+1,paper_count,prefix="Creating papers [{},{}]".format(i+1,paper_count),suffix="",length=50)

        
    else: # create structured papers on rawpapers
        paper_count = len(raw_papers)
        printProgressBar(0,paper_count,prefix="Creating papers [{},{}]".format(0,paper_count),suffix="",length=50)
        for i,raw_paper in enumerate(raw_papers):
            # if(i==100): break
            paper_list.append(StructuredPaper(raw_paper.sections_dict,raw_paper.full_text,max_topics=max_topics,parser=stan_parser))
            printProgressBar(i+1,paper_count,prefix="Creating papers [{},{}]".format(i+1,paper_count),suffix="",length=50)
        print()

    ## concepts extraction
    print("\ncreating repo...")
    repo = Repository(paper_list=paper_list)

    print()

    fake_paper_dump_file = args.fake_paper_dump

    if(args.in_file_path.endswith(".json")):
        p_test = StructuredPaper.from_json(args.in_file_path,max_topics=max_topics,parser=stan_parser)
        if not p_test: raise Exception("cannot extract sections from paper")
    elif(args.in_file_path.endswith(".pdf")):
        rp = RawPaper.fromPdf(path=args.in_pdf_path)
        p_test = StructuredPaper.from_raw(rp,max_topics=max_topics,parser=stan_parser)
    else:
        raise Exception("input paper can be only pdf or json (.json, .pdf)")


    substitutions = []
    num_concepts = len(p_test.topics)
    if (args.alter_concepts):
        ts = bgThesaurus.Thesaurus(THESAURUS_API_CONFIG)

        substitutions = [(t,[(at,0,0) for at in p_test.alteredCandidateConcepts(ts,t)]) for t in p_test.topics]

        inputs = [(idx,p_test,focus_topic,repo,max_topics) for idx,focus_topic in enumerate([t for t,lst in substitutions if lst == []])]
        
    else:
        # workers = None if args.mp == "all" else int(args.mp)
        inputs = [(idx,p_test,focus_topic,repo,max_topics) for idx,focus_topic in enumerate(p_test.topics)]

        print("start subtitution search...",flush=True,end="")
        with mp.Pool(processes=None) as pool:
            substitutions = pool.starmap(findSubstitutions,inputs)
        print("\t[done]")
    

    if(args.synonyms):
        synset_dict = {}
        print("finding synonyms",flush=True,end="")
        for word in set(group(p_test.fulltext.split(),2)):
            key = " ".join(word)
            synset_dict[key] = wn.synsets("_".join(word))
        
        syns_dict = computeSynonymsDict(p_test,p_test.topics,synset_dict)
        print("\t[done]")
    
    with open(fake_paper_dump_file,"wb") as result:
        print("writing results...",flush=True,end="")

        text = ""

        text += "="*20+"KEYWORDS"+"="*20+"\n"
        text += "\n"+str(p_test.topics)
        text += "\n"+"="*40+"\n"

        text += "="*20+"REPLACEMENTS"+"="*20+"\n"
        l = [(s[0],p_test.scores.get(s[0]),s[1]) for s in substitutions]
        #add synonyms
        subs_keys = [s[0] for s in substitutions]

        if(args.synonyms):
            for s in substitutions: 
                item = syns_dict.get(s[0],False)
                if item:
                    for synonym in item[0]:
                        if synonym not in subs_keys: l.append((synonym,item[1],s[1]))
        
        if (args.alter_concepts):
            l = [(s[0],[x[0] for x in s[1]]) for s in substitutions]
        else:
            l.sort(key=lambda s: s[1],reverse=True)
        text += "\n"+",\n".join([str(lp) for lp in l])
        text += "\n"+"="*40+"\n"

        # matrix = repo.computeJCforPaper(p_test)
        # # print(matrix)
        # l = [(k,v[0],v[1])for k, v in matrix.items()]
        # l.sort(reverse=True,key=(lambda x: x[2]))
        # text += "="*20+"JC"+"="*20+"\n"
        # text += "\n"+str(l)
        # text += "\n"+"="*40+"\n"


        for name,content in p_test.sections.items():
            text += "="*20+"ORIGINAL "+name+"="*20+"\n"
            text += content
            text += "\n"+"="*40+"\n"

        text += "="*20+"COMPLETE TEXT"+"="*20+"\n"
        text += p_test.fulltext
        text += "\n"+"="*40+"\n"

        result.write(text.encode("utf-8",'surrogatepass'))
        print("\t[done]")

    # p_test.generatePdf(args={
    #     "filename":fake_paper_dump_file,
    #     "output": "../Results/result.pdf",
    #     })
    
    return    
    

if __name__ == "__main__":
    import argparse
    import timeit

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--concept-list-size",
        help="maximum replacement elements for each concept",
        default=50,
    )

    arg_parser.add_argument(
        "--extracted-topics-limit",
        help="maximum topics extracted for each paper",
        default=50,
    )

    arg_parser.add_argument(
        "-o",
        "--output",
        help="file where store result output",
        default=50,
        
    )

    arg_parser.add_argument(
        "--fake-paper-dump",
        help="result of keyword extraction and substitutions",
        default="../Results/result.out",
    )

    arg_parser.add_argument(
        "--dump-raw-papers",
        help="if provided, each raw-paper is dumped",
        default=None,
            
    )

    arg_parser.add_argument(
        "--pdf-papers-dir",
        help="specify repository where pick pdfs",
        default=None,
            
    )

    arg_parser.add_argument(
        "--csv-dir",
        help="specify repository where export extracted csv",
        default=None,
            
    )

    arg_parser.add_argument(
        "--mp",
        help="start process on NUM multiple cpus",
        default=None,
            
    )

    arg_parser.add_argument(
        "--limit",
        help="specify max paper from repo",
        default=None,
            
    )

    arg_parser.add_argument(
        "--csv-datasets-dir",
        help="specify csv folder where are stored sections",
        default=None,
            
    )

    arg_parser.add_argument(
        "--in-file-path",
        help="file to be faked, must end in .pdf or .json",
        required=True,
    )

    arg_parser.add_argument(
        "--json-papers-dir",
        help="json directory",
        default=None
    )

    arg_parser.add_argument(
        "--synonyms",
        help="if set, synonyms will be included in topic list",
        type=str2bool,
        nargs='?',
        default=False,
    )

    arg_parser.add_argument(
        "--alter-concepts",
        help="if set, synonyms are computed with thesaurus",
        type=str2bool,
        nargs='?',
        default=False,
    )


    args = arg_parser.parse_args()

    print(timeit.Timer(lambda: main(args)).repeat(1, 1))