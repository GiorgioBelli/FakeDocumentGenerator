
import rdflib #requirement
from rdflib import plugin, URIRef
from rdflib.namespace import Namespace
from rdflib.graph import Graph
from rdflib.paths import evalPath, MulPath, SequencePath, OneOrMore, ZeroOrMore, ZeroOrOne
import urllib

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import stopwords 

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

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

def ic_sentence_similarity(sentence1, sentence2, context):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    ic = wordnet_ic.ic("ic-brown.dat")
 
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
        score_list = list(filter(lambda ps: ps, [synset.shortest_path_distance(ss) for ss in synsets2]))
        best_score = max(score_list) if score_list else 0
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
 
    # Average the values
    if(count==0): return 0
    score /= count
    return score


def order_by_cosine_similarity(focus_topic,alternatives):

    # tokenization 
    X_list = word_tokenize(focus_topic)  

    # sw contains the list of stopwords 
    sw = stopwords.words('english')  

    # remove stop words from the string 
    X_set = {w for w in X_list if not w in sw}  

    ret = []

    for alt in alternatives:
        l1 =[];l2 =[] 

        Y_list = word_tokenize(alt) 

        # remove stop words from the string 
        Y_set = {w for w in Y_list if not w in sw} 

        # form a set containing keywords of both strings  
        rvector = X_set.union(Y_set)  
        for w in rvector: 
            if w in X_set: l1.append(1) # create a vector 
            else: l1.append(0) 
            if w in Y_set: l2.append(1) 
            else: l2.append(0) 
        c = 0
        
        # cosine formula  
        for i in range(len(rvector)): 
                c+= l1[i]*l2[i] 
        cosine = c / float((sum(l1)*sum(l2))**0.5) 

        ret.append((alt,cosine))
    ret.sort(key=lambda x: x[1],reverse=True)
    return ret

def tf_idf_similarity_matrix(focus_topic, candidates):
    
    complete_set = [focus_topic,*candidates]

    vect = TfidfVectorizer(min_df=1, stop_words="english")
    tfidf = vect.fit_transform(complete_set)
    pairwise_similarity = tfidf * tfidf.T

    arr = pairwise_similarity.toarray()

    np.fill_diagonal(arr, np.nan)

    focus_topic_idx = 0

    most_similar_idx = np.nanargmax(arr[focus_topic_idx])

    return complete_set[most_similar_idx]

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


    
if __name__ == "__main__":
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

    repo_topics = {'boston housing data', 'optimal linear hyper plane', 'facilitates user custom extension', 'jm‖wut − wt‖ ≤', 'successfully detect distinctions', 'x⊤i xj', 'control sequence serves', 'established research field', 'dinh phan caonguyen', 'q∗i ∝ exp', 'batch size set', 'completely abandon reporting', 'large simulation error', 'previous metafeatures w̃j', 'desired business contexts', 'conventional methods propose', 'sequence uk ∈ `p', 'polynomial time precisely', 'nonenumerable universal measures computable', 'manually designed normalized indices', 'word embedding vectors', 'transductive boltzmann machine', 'optimal global solution', 'yt eŷt ∼qt', 'fine explanation structure', 'adding noises sampled', 'minimize false positives', 'a∗ ∈ ca', 'best-performing tabular model', 'shortest path routing', 'coherent latent space', 'τ̂i = −τi', 'entire input sequence', 'κλ √', 'chapter 5 learning continually', 'infrared gait recognition based', 'fast preliminary check', '|p ≤', 'layer groups specific operations', 'real concept drift', '+ γmax a′ q∗', 'covariate shift boils', 'full scripts provided', 'edge2 = net', 'providing meaningful end-to-end protection', 'passive learning algorithm', '∈ arg min ~λ∈λ π', 'highly strategic game', 'reducing computational overhead', 'outcome variable iserror', 'de sitter space', 'approximate solution ∆α̂`', 'interactive plotting api', 'e1 = b1 + b2', 'assigns scores α =', 'provide light-weight security protection', '• abalone dataset10', '288 nε2 ln 8c', 'independently sequenced samples', '23rd acm conference', 'fully connected', 'editing experiment association analysis', 'interaction capabilities', 'laborious human involvement', 'persistent data store', 'dna variant sites', 'solve non-linearly separable problems', 'fi ln pi', 'machine predictions increases', 'test audio-based deep learning systems', 'content image i′', 'attracted ample interest', 'label requests', 'information resources', 'common pool resource appropriation problem', '= |v |', 'newly emerged interpretable models', 'network expects structures similar', 'adaptively solicits feedback', 'reduce label complexity', 'weak learner accidentally biases', 'embedding coordinates', 'k∑ k=1 nk∑ i=1 zki', 'exert high forces', 'knudsen number increases', 'linear system u>x =', 'numpy array format', 'solving trifactor mtl', 'benchmark real datasets', 'final analysis show', '∑ t=0 αn−t ∂h ∂', 'convolutional networks', 'selective adversarial attack', 'utilizing semantic relations', 'resulting clusters increases', 'compute hopping probabilities', 'ip + 1', 'typically considered', 'grafting energy-harvesting leaves', 'affect-expressive motion graphs', 'accurately learn noise', 'diversity measurement based', 'original black box', 'steepest descent algorithm—also referred', 'find predicted values', 'training complex neural network architectures', 'original gan model goodfellow', 'powerful methods', 'attacker’s planned scenario', 'gt − d̂t', 'σ ◦ ~x', 'compiler decision', '= inf h∈c', 'auxiliary averaged sequence', 'standard evaluation schemes', 'library kernel matmul', 'input-output learning problems', 'fundamentally flawed idea', 'perform parameter updates', 'task-specific policy parameter', 'accompanying python code', 'learning bias', 'received local models', 'lower bounded', 'recognizing epileptic seizure', 'iteration remains cheap', 'classification based approach', 'future working conditions', 'train unsupervised clustering algorithms', 'feature continuous state spaces', 'a∗ search conducts exploitation', 'malware detection decisions', 'common machine learning task', 'greedy layer-wise training', 'η ← η + ∑', 'generated noisy labels', 'robotic applications', 'interesting question arises', '1 + γ · 1wjk≥0', '∈ ck', 'aligned source word', 'proofs and/or references', '+ log 2λ 2', 'algorithm 𝐴𝜆 learns', 'pac model makes', 'process large-scale data', 'ma hines', 'kurtosis question2 vector', 'extreme output regions', 'method called gradient boosted trees', 'filtering attacker’s updates', 'essentially parameterized computational procedures', 'network gradient', 'average score chapter 4', 'fewer training examples', 'natural image sequence data', 'normal computer vision algorithms', 'true governing parameters', 'enforces m̌d̃f ≤ max', 'choose λ = 104', 'valid x86 instruction', 'moderately effective independently', 'conventional comparison sorting algorithm', 'accept combination goodness', 'informative alarms show', 'mps tensor networks', 'outer cross validation step', 'size |d| = 10', 'joint extraction', 'fugro roames engineers', 'compare f1 performance results', 'source code produced', 'factorized natural gradient', 'training file trainfile', 'double stepsize = 1e-5', 'pretraining-then-transferring learned representations', 'deep learning model tested', '6 machine learning classifiers', 'development process models specific', 'dynamic control flows', 'discrete atari games', 'attack surface guidelines', 'propose opponent-guided tactic learning', 'information retrieval research projects', 'recently interpretable deep neural networks', 'pharmacy insurance coverage', 'gurobi’s cumulative time-to-targets exceeds', 'concretization operation ρ', 'concept exemplar set', 'intermittent learning', 'subsampling step retains', 'large compositional envelop accessible', '√ min h∈h ∑', 'k2 = σ', 'typically performed separately', 'obtained edit similarity', 'prior tasks tj ∈', 'artificial variable created', 'guide optimization techniques', 'sections 2 briefly introduces', 'vanilla recurrent neural network', 'electricity context means', 'observed pattern modes', 'oov rate', 'regularized erm problem', 'helix’s cumulative run time', 'inducing values uf', 'sophisticated episodic memories', 'closely related problems', 'time-varying wireless environment', 'exceedingly large', 'acm computing classification system', '+ η satisfies', 'data volume requirements presents', 'recidivism prediction', 'θ− ← θ', 'semi-supervised learning wishes', 'additional information source', 'budget size based', '56 spectral analysis', 'discuss related work', 'major machine learning algorithms', 'accuracy %100 fnfptntp tntp', 'higher risk users make', 'global range effect due', 'consensus objective', 'iid bernoulli trials', 'recently shown promising results', 'classical bayesian decision theory', 'implies ŷ = argmax y∈', 'original mtat tags', 'potentially model language', 'interactive question answering', 'actual predictions ŷ1', 'specific geographic regions', '145 made volatility predictions', 'language prediction model', 'specific application areas', 'ml algorithm', 'generative linear-gaussian models', 'detecting gravitational lenses', 'stochastic dynamic programming approaches', 'arrest case id', 'xti xj m∑ r=1 a2r', '• sequence generation', 'symmetric arguments show', 'machine learning community clash', 'lower dimensional space', 'popular research trend', 'image analysis method', 'present preliminary results achieved', 'rl ŵ', 'exhibit desired property profiles', 'required claim bq ≤', 'tweaking hyper parameters', 'unique global minimizer', 'neural sequence-to-sequence learning problems', 'design approach', 'kernel machines minimize', 'proposed neural programmer', '∈ ft ∥ ∥f', 'describe preliminary defenses', 'user securely quit', 'high reuse rate', 'input file formats', 'sun goodness measures', 'supports fuzz test input generation', '= ŷkj 8 end', 'identify systematic noise', 'set si ⊂ rp', 'support = 1%', 'defined quantum agent', 'gaussian measure', 'drop rate set', 'high-fidelity imitation model metamimic proposed', 'ρ = 0', 'additional thread pool', 't+l | ft', 'stochastic proximal gradient methods', 't̂k = ∑m̂k m=m̂k−1+1 i⋆mk', '= q∗', 'statistical mechanics', 'encrypted communication channel', 'original neural architecture remains', 'θ′ =', 'infinite neural network', 'local models received', 'raw images make', 'wij = wji', 'model sharing', 'subset st ⊂', 'parties updates unit', 'uniformly sampling data points', 'probabilistic density function', '• requirements traceability', 'non-oracularized quantum extension eq', 'software development life cycle', 'signal data processing', 'subject receives relevant', 'nlp tasks involve learning', 'metamorphic testing approach', 'ge ne r-', 'strong geometric requirement', 'categorical fake/not fake error', '• adding residual connections', 'don’t explore function approximation', 'leaf nodes tf =', 'represent valid solutions', 'linear kernel equation', '= rκ−1 + 2δ', 'hyperplane x>1 = 1', 'higher degree', 'explanation facilitates humans', 'advanced prostatic cancer', '‘hard’ selection operator', 'final equalities follow', 'elixir-czech republic marco antonio tangaro', 'credit card dataset', 'foster code understanding', 'em algorithms coincide', 'learn mutual defection', 'ut = ∇st+1f', 'simulator server-client interface', 'finding sufficient labeled examples', 'observed d-dimensional binary-valued vectors', 'label queries', 'multi-head gated graph attention aggregator', 'exponential-family approximating distributions', '3-layers fully-connected network', 'real-world supervised learning problems perfectly', 'probabilistic svm', 'traditional rule based'}

    l = {"An apple a day keeps the doctor away", 
           "Never compare an apple to an orange", 
           "I prefer scikit-learn to Orange", 
           "The scikit-learn docs are Orange and Blue"}

    print(tf_idf_similarity_matrix("human cognitive abilities",list(repo_topics)))
