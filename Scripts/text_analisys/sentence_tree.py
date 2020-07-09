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
    # use these three lines to do the replacement
    rep_dict = dict((re.escape(k), v) for k, v in rep_dict.items()) 
    #Python 3 renamed dict.iteritems to dict.items so use rep.items() for latest versions
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
        str_rep = {"ï¬�":"fi","ï¬":"fi","ï¬‚":"fl"}

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
    def from_raw(rawPaper,parser):
        return StructuredPaper(None,rawPaper.sections,rawPaper.fulltext,parser)


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
        # print("\treplacePapersTopics...",end="")
        # self.replacePapersTopics()
        # print("\t[done]")
        # print("\tcomputePapersContext...",end="")
        # self.computePapersContext()
        # print("\t[done]")
        # print("\tinitGeneralContexts...",end="")
        # self.initGeneralContexts()
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
    csv_datasets_dir = "../Results/extractionMP/"
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
        for idx in range(min(len(abstract_csv), len(introduction_csv), len(corpus_csv), len(conclusion_csv))):
            sections = {
                PaperSections.PAPER_ABSTRACT : abstract_csv[idx],
                PaperSections.PAPER_INTRO : introduction_csv[idx],
                PaperSections.PAPER_CORPUS : corpus_csv[idx],
                PaperSections.PAPER_CONCLUSION : conclusion_csv[idx]
            }
            raw_papers.append(RawPaper.fromSections(sections))

    else: # read pdf repo and generate papers
        path = args.pdf_repo

        csv_dir = args.csv_dir

        repo_extr = RepositoryExtractor(path)

        limit = int(args.repo_limit) if args.repo_limit else None

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

    # text = '''1 Introduction  Despite their proven ability to tackle a large class of complex problems [1], neural networks are still poorly understood from a theoretical point of view. While general theorems prove them to be universal approximators [2], their ability to obtain generalizing solutions given a finite set of examples remains largely unexplained. This behavior has been observed in multiple settings. The huge number of parameters and the optimization algorithms employed to optimize them (gradient descent and its variations) are thought to play key roles in it [3–5]. In consequence, a large research effort has been devoted in recent years to understanding the training dynamics of neural networks with a very large number of nodes [6–8]. Much theoretical insight has been gained in the training dynamics of linear [9, 10] and nonlinear networks for regression problems, often with quadratic loss and in a teacher-student setting [11–14], highlighting the evolution of correlations between data and network outputs. More generally, the input-output correlation and its effect on the landscape has been used to show the effectiveness of gradient descent [15, 16]. Other approaches have focused on infinitely wide networks to perform a mean-field analysis of the weights dynamics [17–22], or study its neural tangent kernel (NTK, or “lazy”) limit [23–26]. In this work, we investigate the learning dynamics for binary classification problems, by considering one of the most common cost functions employed in this setting: the linear hinge loss. The idea behind the hinge loss is that examples should contribute to the cost function if misclassified, but also if classified with a certainty lower than a given threshold. In our case this cost is linear in the distance from the threshold, and zero for examples classified above threshold, that we shall call satisfied henceforth. This specific choice leads to an interesting consequence: the instantaneous gradient for each node due to unsatisfied examples depends on the activation of the other nodes only through their population, while that due to satisfied examples is just zero. Describing the learning dynamics in the mean-field limit amounts to computing the effective example distribution for a given distribution of parameters: each node then evolves “independently” with a time-dependent dataset determined self-consistently from the average nodes population. Contribution. We provide an analytical theory for the dynamics of a single hidden layer neural network trained for binary classification with linear hinge loss. In Sec. 2 we obtain the mean-field theory equations for the training dynamics. Those equations are a generalizations of the ones obtained for mean-square loss in [17–22]. In Sec. 3 we focus on linearly separable data with spherical symmetry and present an explicit analytical solution of the dynamics of the nodes parameters. In this setting we provide a detailed study of the cross-over between the lazy [23] and rich [27] learning regimes (Sec. 3.2). Finally, we asses the limitations of mean-field theory by studying the case of large but finite number of nodes and finite number of training samples (Sec. 3.3). The most important new effect is overfitting, which we are able to describe by analyzing corrections to mean-field theory. In Sec. 3.4 we show that introducing a small fraction of mislabeled examples induces a slowing down of the dynamics and hastens the onset of the overfitting phase. Finally in Sec. 4 we present numerical experiments on a realistic case, and show that the associated nodes dynamics in the first stage of training is in good agreement with our results. The merit of the model we focused on is that, thanks to its simplicity, several effects happening in real networks can be studied analytically. Our analytical theory is derived using reasoning common in theoretical physics, which we expect can be made rigorous following the lines of [17–22]. All our results are tested throughout the paper by numerical simulations which confirm their validity.'''
    # text = '''Consider the task of finding your way to the bathroom while at a new restaurant. As humans, we can efficiently solve such tasks in novel environments in a zero-shot manner. We leverage common sense patterns in the layout of environments, which we have built from our past experience of similar environments. For finding a bathroom, such cues will be that they are typically towards the back of the restaurant, away from the main seating area, behind a corner, and might have signs pointing to their locations (see Figure 1). Building computational systems that can similarly leverage such semantic regularities for navigation has been a long-standing goal. Hand-specifying what these semantic cues are, and how they should be used by a navigation policy is challenging. Thus, the dominant paradigm is to directly learn what these cues are, and how to use them for navigation tasks, in an end-to-end manner via reinforcement learning. While this is a promising approach to this problem, it is sample inefficient, and requires many million interaction samples with dense reward signals to learn reasonable policies. But, is this the most direct and efficient way of learning about such semantic cues? At the end of the day, these semantic cues are just based upon spatial consistency in co-occurrence of visual patterns next to one another. That is, if there is always a bathroom around the corner towards the back of the restaurant, then we can learn to find this bathroom, by simply finding corners towards the back of the restaurant. This observation motivates our work, where we pursue an alternate paradigm to learn semantic cues for navigation: learning about this spatial co-occurrence in indoor environments through video tours of indoor spaces. People upload such videos to YouTube (see project video) to showcase real estate for renting and selling. We develop techniques that leverage such YouTube videos to learn semantic cues for effective navigation to semantic targets in indoor home environments (such as finding a bed or a toilet). Such use of videos presents three unique and novel challenges, that don’t arise in standard learning from demonstration. Unlike robotic demonstrations, videos on the Internet don’t come with any action labels. This precludes learning from demonstration or imitation learning. Furthermore, goals and intents depicted in videos are not known, i.e., we don’t apriori know what each trajectory is a demonstration for. Even if we were to label this somehow, the depicted trajectories may not be optimal, a critical assumption in learning from demonstration [49] or inverse reinforcement learning [41]. Our formulation, Value Learning from Videos or VLV, tackles these problems by a) using pseudo action labels obtained by running an inverse model, and b) employing Q-learning to learn from video sequences that have been pseudo-labeled with actions. We follow work from Kumar et al. [36] and use a small number of interaction samples (40K) to acquire an inverse model. This inverse model is used to pseudo-label consecutive video frames with the action the robot would have taken to induce a similar view change. This tackles the problem of missing actions. Next, we obtain goal labels by classifying video frames based on whether or not they contain the desired target objects. Such labeling can be done using off-the shelf object detectors. Use of Q-learning [58] with consecutive frames, intervening actions (from inverse model), and rewards (from object category labels), leads to learning optimal Q-functions for reaching goals [53, 58]. We take the maximum Q-value over all actions, to obtain value functions. These value functions are exactly γs, where s is the number of steps to the nearest view location of the object of interest (γ is the Q-learning discount factor). These value functions implicitly learn semantic cues. An image looking at the corner towards the back of the restaurant will have a higher value (for bathroom as the semantic target) than an image looking at the entrance of the restaurant. These learned value functions when used with a hierarchical navigation policy, efficiently guide locomotion controllers to desired semantic targets in the environment. Learning from such videos can have many advantages, some of which address limitations of learning from direct interaction (such as via RL). Learning from direct interaction suffers from impractical sample complexity (the policy needs to discover high-reward trajectories which may be hard to find in sparse reward scenarios) and poor generalization (limited number of instrumented physical environments available for reward-based learning, or sim2real gap). Learning from videos side-steps both these issues. Our experiments in visually realistic simulations show 66% better performance than RL methods, while at the same time requiring 250× fewer active interaction samples for training.'''
    # text = '''1 Introduction  Active learning is an important machine learning paradigm with a rich class of problems and mature literature [Prince, 2004, Settles, 2012, Hanneke et al., 2014]. Oftentimes, users have access to a large pool of unlabeled data and an oracle that can provide a label to a data point that is queried. Querying the oracle for the label comes at a cost, computational and/or monetary. Hence, a key objective for the algorithm is to â€œwiselyâ€� choose the set of points from the unlabelled pool that can provide better generalization. In this paper, we propose a probabilistic querying procedure to choose the points to be labeled by the oracle motivated from importance sampling literature [Tokdar and Kass, 2010]. Importance sampling is a popular statistical technique widely used for fast convergence in Monte Carlo based methods [Doucet et al., 2001] and stochastic optimization [Zhao and Zhang, 2015].  The main contributions of this paper are as follows. (a) We propose an importance sampling based algorithm for active learning, which we call Active Learning with Importance Sampling (ALIS). (b) We derive a high probability upper bound on the true loss and design the ALIS algorithm to directly minimize the bound. (c) We determine an optimal sampling probability distribution for the algorithm. (d) We demonstrate that the optimal sampling distribution gives a tighter bound on the true loss compared to the baseline uniform sampling procedure.'''

    text_abstract = args.text_abstract
    text_intro = args.text_intro
    text_body = args.text_body
    text_conclusion = args.text_conclusion

    if(args.text_abstract is None ): text_abstract = input("text-abstract to be faked: ")
    if(args.text_intro is None ): text_intro = input("text-intro to be faked: ")
    if(args.text_body is None ): text_body = input("text-body to be faked: ")
    if(args.text_conclusion is None ): text_conclusion = input("text-conclusion to be faked: ")

    # TODO se file singolo o testo completo creo un rawPaper e faccio extrat sections 
    
    fulltext = text_abstract+" "+text_intro+" "+text_body+" "+text_conclusion
    sections = {
            PaperSections.PAPER_ABSTRACT : text_abstract,
            PaperSections.PAPER_INTRO : text_intro,
            PaperSections.PAPER_CORPUS : text_body,
            PaperSections.PAPER_CONCLUSION : text_conclusion
        }

    p_test = StructuredPaper(sections,fulltext,parser)


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

    for i,focus_topic in enumerate(p_test.topics):
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
        text += "\n"+"\n".join([str(s) for s in substitutions])
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

        result.write(text.encode("utf-8"))
        print("\t[done]")
    
    return    
    

if __name__ == "__main__":
    import argparse
    import timeit

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--conc-list-size",
        help="maximum replacement elements for each concept",
        default=50,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="file where store result output",
        default=50,
    )

    parser.add_argument(
        "--text-abstract",
        help="text-abstract to be faked",
        default=None,
    )

    parser.add_argument(
        "--text-intro",
        help="text-intro to be faked",
        default=None,
    )

    parser.add_argument(
        "--text-body",
        help="text-body to be faked",
        default=None,
    )

    parser.add_argument(
        "--text-conclusion",
        help="text-conclusion to be faked",
        default=None,
    )

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
        "--repo-limit",
        help="specify max paper from repo",
        default=None,
    )
    

    # csv path (valutare i csv cper intro, abstract....)





    args = parser.parse_args()

    print(timeit.Timer(lambda: main(args)).repeat(1, 1))