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
from pdf_parsing.paper_to_txt import RepositoryExtractor, RawPaper, PaperSections
from pdf_parsing.txt2pdf import PDFCreator, Args, Margins

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

def replace_multi_regex(text,rep_dict):
    rep_dict = dict((re.escape(k), v) for k, v in rep_dict.items()) 
    pattern = re.compile("|".join(rep_dict.keys()))
    text = pattern.sub(lambda m: rep_dict[re.escape(m.group(0))], text)
    return text

with open("../datasets/AI_glossary.txt","r") as ai_glossary_fd:
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

class StructuredPaper():
    def __init__(self,sections,fulltext,parser):
        str_rep = {"ï¬�":"fi","ï¬":"fi","ï¬‚":"fl","ﬁ":"fi","ﬀ":"ff","ﬂ":"fl"}

        self.sections = sections
        # self.fulltext = re.sub("ï¬�|ï¬","fi",fulltext)
        self.fulltext = replace_multi_regex(fulltext,str_rep)
        self.topics = set()
        self.contexts = dict()
        self.proximityFrequences = {}
        self.jcMatrix = dict()
        # self.sentencesTrees = self.createTree(parser)

        self.transformedFullText = ""
        self.transformedTopics = set()

        # self.spacy_extractTopics()
        for section in self.sections.keys() :
            # self.sections[section] = re.sub("ï¬�|ï¬","fi",self.sections[section])
            self.sections[section] = replace_multi_regex(self.sections[section],str_rep)
            self.rake_extractTopics(self.sections[section])

    @staticmethod
    def from_raw(rawPaper,parser=None):
        return StructuredPaper(rawPaper.sections_dict,rawPaper.full_text,parser)


    def createTree(self,text,parser):
        return parser.raw_parse_sents(text.split(". "))

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
    
    def spacy_extractTopics(self,text):
        doc = nlp(text)
        for token in doc.noun_chunks:
            token = str(token)
            self.topics.add(token)
            self.transformedTopics.add("#".join(token.split()))
            self.contexts["#".join(token.split())] = set()
        self.cleanRedoundantTopics()

    def rake_extractTopics(self,text,threshold=5.0):
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
            distances.append((topic,sentence_similarity(focus_topic,topic)))
        distances.sort(reverse=True,key=lambda x: x[1])
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
        self.papers = paper_list

        self.generalContexts = dict()
        self.generalTopics = set()
        self.generalTransformedTopics = set()
        self.jcMatrix = dict()

        print("\tinitTopics...",end="")
        self.initTopics()
        print("\t[done]")
        

        # self.prepareForJC()
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

    def prepareForJC():
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
        for tp in paper.transformedTopics:
            for tpp in self.generalTransformedTopics:
                if(tp != tpp):
                    jc = forge_jaccard_distance(paper.contexts[tp],self.generalContexts[tpp])
                    if(jc != 0 and jc > (paper.jcMatrix.get(tp,(tp,0)))[1]): 
                        paper.jcMatrix[tp] = (tpp,jc)
        return paper.jcMatrix
        
def load_csv(path):
    with open(path, "r") as f:
        rows = f.read().split("\f\n")
    return rows

def main(args):

    # os.environ['STANFORD_PARSER'] = 'C:\\Users\\GIORGIO-DESKTOP\\Documents\\Universita\\FakeDocumentGenerator\\models\\stanford-parser-full-2018-10-17\\stanford-parser.jar'
    # os.environ['STANFORD_MODELS'] = 'C:\\Users\\GIORGIO-DESKTOP\\Documents\\Universita\\FakeDocumentGenerator\\models\\stanford-parser-full-2018-10-17\\stanford-parser-3.9.2-models.jar'
    
    # java_path = "C:\\Program Files\\Java\\jre-10.0.1\\bin\\java.exe"
    # os.environ['JAVAHOME'] = java_path

    # parser = stanford.StanfordParser(model_path="C:\\Users\\GIORGIO-DESKTOP\\Documents\\Universita\\FakeDocumentGenerator\\models\\stanford-parser-full-2018-10-17\\models\\englishPCFG.ser.gz")
    parser = None
    
    # csv_path = "C:\\Users\\GIORGIO-DESKTOP\\Documents\\Universita\\FakeDocumentGenerator\\datasets\\arxiv\\4500_summaries_trainingSet.csv"
    # csv_path = "C:\\Users\\GIORGIO-DESKTOP\\Desktop\\intros.csv"
    csv_datasets_dir = args.csv_datasets_dir

    # abstract_csv = pandas.read_csv(os.path.join(csv_datasets_dir,"abstract.csv"), delimiter = '\f\n', engine="python")
    # intro_csv = pandas.read_csv(os.path.join(csv_datasets_dir,"intro.csv"), delimiter = '\f\n', engine="python")
    # corpus_csv = pandas.read_csv(os.path.join(csv_datasets_dir,"corpus.csv"), delimiter = '\f\n', engine="python")
    # conclusion_csv = pandas.read_csv(os.path.join(csv_datasets_dir,"conclusion.csv"), delimiter = '\f\n', engine="python")

    if csv_datasets_dir:
        abstract_csv = load_csv(os.path.join(csv_datasets_dir,"abstract.csv"))
        introduction_csv = load_csv(os.path.join(csv_datasets_dir,"introduction.csv"))
        corpus_csv = load_csv(os.path.join(csv_datasets_dir,"corpus.csv"))
        conclusion_csv = load_csv(os.path.join(csv_datasets_dir,"conclusion.csv"))

        # print(len(abstract_csv))
        # print(len(introduction_csv))
        # print(len(corpus_csv))
        # print(len(conclusion_csv))

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

    else: # read pdf repo and generates papers
        path = args.pdf_repo

        csv_dir = args.csv_dir

        repo_extr = RepositoryExtractor(path)

        limit = int(args.limit) if args.limit else None

        if(args.mp):
            print("starting extraction... ",end="")
            workers = None if args.mp == "all" else int(args.mp)
            failed = repo_extr.extractMP(limit=limit,processes=workers)
            print("\t[done] [{} fails]".format(failed))
        else: 
            repo_extr.extract(limit=limit)


        if(csv_dir):
            print("starting csv creation... ",end="")
            repo_extr.exportSections(csv_dir,sections=PaperSections.as_list())
            print("\t[done]")


        for i,p in enumerate(repo_extr.papers):
            p.extract_sections()
            if(args.dump_raw_papers):
                p.dump(outPath=os.path.join(args.dump_raw_papers,"paper-rawtest-%s.txt"%i))

        raw_papers = repo_extr.papers

    ## concepts extraction

    paper_list = []
    paper_count = len(raw_papers)
    printProgressBar(0,paper_count,prefix="Creating papers [{},{}]".format(0,paper_count),suffix="",length=50)
    for i,raw_paper in enumerate(raw_papers):
        # if(i==100): break
        paper_list.append(StructuredPaper(raw_paper.sections_dict,raw_paper.full_text,parser))
        printProgressBar(i+1,paper_count,prefix="Creating papers [{},{}]".format(i+1,paper_count),suffix="",length=50)
    print()
    print("\ncreating repo...")
    repo = Repository(paper_list=paper_list)

    print()

    fake_paper_dump_file = args.fake_paper_dump

    rp = RawPaper.fromPdf(path=args.in_pdf_path)

    # text_abstract = args.text_abstract
    # text_intro = args.text_intro
    # text_body = args.text_body
    # text_conclusion = args.text_conclusion

    # if(args.text_abstract is None ): text_abstract = input("text-abstract to be faked: ")
    # if(args.text_intro is None ): text_intro = input("text-intro to be faked: ")
    # if(args.text_body is None ): text_body = input("text-body to be faked: ")
    # if(args.text_conclusion is None ): text_conclusion = input("text-conclusion to be faked: ")

    # fulltext = text_abstract+" "+text_intro+" "+text_body+" "+text_conclusion
    # sections = {
    #         PaperSections.PAPER_ABSTRACT : text_abstract,
    #         PaperSections.PAPER_INTRO : text_intro,
    #         PaperSections.PAPER_CORPUS : text_body,
    #         PaperSections.PAPER_CONCLUSION : text_conclusion
    #     }

    p_test = StructuredPaper.from_raw(rp)


    # found = []
    # for t in p_test.transformedTopics:
    #     isInOnt = isInOntology(t)
    #     if(isInOnt):
    #         for term in t.split("#"):
    #             showTree(getWnTerm(term))
    

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
    if(num_concepts>0): printProgressBar(0,num_concepts,prefix="Finding substitution [{}/{}]".format(0,num_concepts),suffix="",length=50)

    for i,focus_topic in enumerate(list(p_test.topics)[1:4]):
        printProgressBar(i+1,num_concepts,prefix="Finding substitutions [{}/{}]".format(i+1,num_concepts),suffix="computing: "+focus_topic,length=50)
        topic, candidates = p_test.getCandicateConcepts(focus_topic,repo.generalTopics)
        substitutions.append((topic,[c[0] for c in candidates[:50]]))
    print()

    with open(fake_paper_dump_file,"wb") as result:
        print("writing results...",end="")

        text = ""

        text += "="*20+"KEYWORDS"+"="*20+"\n"
        text += "\n"+str(p_test.topics)
        text += "\n"+"="*40+"\n"

        text += "="*20+"REPLACEMENTS"+"="*20+"\n"
        text += ",\n"+"\n".join([str(s) for s in substitutions])
        text += "\n"
        text += "\n"+str(treplace)
        text += "\n"+"="*40+"\n"

        # matrix = repo.computeJCforPaper(p_test)
        # l = [(k,matrix[k][0],matrix[k][1])for k in matrix.keys()]
        # l.sort(reverse=True,key=(lambda x: x[2]))
        # text += "="*20+"JC"+"="*20+"\n"
        # text += "\n"+str(l)
        # text += "\n"+"="*40+"\n"

        text += "="*20+"ORIGINAL ABSTRACT"+"="*20+"\n"
        text += p_test.sections[PaperSections.PAPER_ABSTRACT]
        text += "\n"+"="*40+"\n"
        text += "="*20+"ORIGINAL INTRO"+"="*20+"\n"
        text += p_test.sections[PaperSections.PAPER_INTRO]
        text += "\n"+"="*40+"\n"
        text += "="*20+"ORIGINAL BODY"+"="*20+"\n"
        text += p_test.sections[PaperSections.PAPER_CORPUS]
        text += "\n"+"="*40+"\n"
        text += "="*20+"ORIGINAL CONCLUSIONS"+"="*20+"\n"
        text += p_test.sections[PaperSections.PAPER_CONCLUSION]
        text += "\n"+"="*40+"\n"
        text += "="*20+"COMPLETE TEXT"+"="*20+"\n"
        text += p_test.fulltext
        text += "\n"+"="*40+"\n"

        result.write(text.encode("utf-8"))
        print("\t[done]")

    p_test.generatePdf(args={
        "filename":fake_paper_dump_file,
        "output": "../Results/result.pdf",
        })
    
    return    
    

if __name__ == "__main__":
    import argparse
    import timeit

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--concept-list-size",
        help="maximum replacement elements for each concept",
        default=50,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="file where store result output",
        default=50,
        
    )

    # parser.add_argument(
    #     "--text-abstract",
    #     help="text-abstract to be faked",
    #     default=None,
    #     
    # )

    # parser.add_argument(
    #     "--text-intro",
    #     help="text-intro to be faked",
    #     default=None,
    #     
    # )

    # parser.add_argument(
    #     "--text-body",
    #     help="text-body to be faked",
    #     default=None,
    #     
    # )

    # parser.add_argument(
    #     "--text-conclusion",
    #     help="text-conclusion to be faked",
    #     default=None,
    #     
    # )

    parser.add_argument(
        "--fake-paper-dump",
        help="result of keyword extraction and substitutions",
        default="../Results/result.out",
    )

    parser.add_argument(
        "--dump-raw-papers",
        help="if provided, each raw-paper is dumped",
        default=None,
            
    )

    parser.add_argument(
        "--pdf-repo",
        help="specify repository where pick pdfs",
        default=None,
            
    )

    parser.add_argument(
        "--csv-dir",
        help="specify repository where export extracted csv",
        default=None,
            
    )

    parser.add_argument(
        "--mp",
        help="start process on NUM multiple cpus",
        default=None,
            
    )

    parser.add_argument(
        "--limit",
        help="specify max paper from repo",
        default=None,
            
    )

    parser.add_argument(
        "--csv-datasets-dir",
        help="specify csv folder where are stored sections",
        default=None,
            
    )

    parser.add_argument(
        "--in-pdf-path",
        help="pdf file to be faked",
        required=True,
    )

    # csv path (valutare i csv cper intro, abstract....)





    args = parser.parse_args()

    print(timeit.Timer(lambda: main(args)).repeat(1, 1))