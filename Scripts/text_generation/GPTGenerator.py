import re

class Span():
    def __init__(self,start,end):
        self.start = start
        self.end = end

    def __str__(self):
        return "Span({},{})".format(self.start,self.end)

class ReplacementConcept():
    def __init__(self,concept,frequency,alternatives):
        self.alternatives = alternatives
        self.concept = concept
        self.frequency = frequency

class ConceptsUtils():

    @staticmethod
    def get_concept_names(concepts):
        return [concept.concept for concept in concepts]

    @staticmethod
    def get_concept_dict(concepts):
        conc_dict = {}
        for concept in concepts:
            conc_dict[concept.concept] = concept
        return conc_dict


    @staticmethod
    def create_freq_dict(concepts):
        freq_dict = {}
        for concept in concepts:
            freq_dict[concept.concept] = concept.frequency
        return freq_dict

    @staticmethod
    def discard_by_frequency(concepts,threshold=2):
        concepts = [concept for concept in concepts if concept.frequency>=threshold]
        concepts.sort(key=lambda concept: len(concept.concept))
        return concepts
    

class GPTGenerator():  
    def __init__(self,gpt_args,sentence_separator=r"(\. [a-zA-Z])",context_threshold=500):
        self.sentence_separator = sentence_separator
        self.context_threshold = context_threshold
        
        self.session = gpt2.start_tf_sess()
        self.run_name = gpt_args.get("run_name","checkpoint/run1")
        gpt2.load_gpt2(self.session, **gpt_args)

    def generate_eq_sentence(self,sentence,context, span, concept_string):
        # sentence = sentence[:span.start] + concept_string
        
        contexted_text = context+concept_string
        context_len = len(context)

        print("\nCONTEXTED-TEXT:",contexted_text)
        # text = sentence[:span.end]+"("+concept_string+")"+sentence[span.end:]+". A" ## TODO gen with gpt model
        res = gpt2.generate(self.session,seed=0,run_name=self.run_name,return_as_list=True,prefix=contexted_text,truncate="<|endoftext|>",nsamples=1,temperature=0.75,top_k=40,top_p=0.9)

        text = res[0]
        print("\nGENERATED-TEXT:",res[0])

        try:
            first_match = next(re.finditer(self.sentence_separator,text[context_len:]))
            
            new_sentence = sentence[:span.start]+text[context_len:context_len+first_match.start()+1]
        except StopIteration:
            return None

        return new_sentence

    def get_context(self,whole_text,concept_span):
        
        min_lookback = concept_span.end-self.context_threshold
        
        if min_lookback > 0:
        
            matches = list(re.finditer(self.sentence_separator,whole_text[min_lookback:concept_span.start+1]))
            
            try:
                first_match = next(re.finditer(self.sentence_separator,whole_text[min_lookback:]))
                context = whole_text[min_lookback+first_match.end()-1:concept_span.start]
            except StopIteration:
                return None
        else:
            context = "<|startoftext|>"+whole_text[:concept_span.start]

        return context

    def __split_sentences(self,whole_text):
        sentences = re.split(self.sentence_separator,text)
        if(len(sentences)>1):
            for idx in range(2,len(sentences),2):
                first_char = sentences[idx-1][-1]
                sentences[idx] = first_char+sentences[idx]
                sentences[idx-2] += sentences[idx-1][0]
            sentences = sentences[::2]
        else: sentences = [whole_text]

        return sentences

    def printDiff(self,original,faked):
        print(
          '''
            <html>

              <head>
                  <link href="https://fonts.googleapis.com/css2?family=Baskervville&family=Libre+Baskerville&display=swap" rel="stylesheet">
                  <style>
                      body{{
                        font-family: 'Baskervville', serif;font-family: 'Libre Baskerville', serif;
                        width: 75%;
                        margin: auto;
                      }}

                      h1{{
                          text-align: center;
                      }}

                      p{{
                          line-height: 25px;
                      }}

                      .highlight>a{{
                        text-decoration: none;
                        outline: none;                        
                      }}

                      .highlight{{
                          background-color: black;
                          font-style: bold;
                          color: white;
                      }}

                      a{{
                        background-color: black;
                        color: white;
                        font-style: bold;
                      }}

                      .not_gen{{
                        color: red;
                      }}

                      .k1>a{{background-color: #c0cde4;color: black;font-style: bold;}}
                      .k2>a{{background-color: #5191f9;color: white;font-style: bold;}}
                      .k3>a{{background-color: #78e620;color: white;font-style: bold;}}
                      .k4>a{{background-color: #5c5376;color: white;font-style: bold;}}
                      .k5>a{{background-color: #f7b9e8;color: black;font-style: bold;}}
                      .k6>a{{background-color: #ccf1c0;color: black;font-style: bold;}}
                      .k7>a{{background-color: #af98e9;color: white;font-style: bold;}}
                      .k8>a{{background-color: #2a52a8;color: white;font-style: bold;}}
                  </style>
              </head>

              <body>
                <h1>Original Text</h1><br>
                  <p>{0}</p>
                <hr>
                <h1>Fake Text</h1><br>
                  <p>{1}</p>
              </body>
            </html>
          '''.format(". ".join(original),". ".join(faked))
        )

    def fake_text(self,text,concepts,concept_discard_func=(ConceptsUtils.discard_by_frequency,{"threshold":0}),):
        
        text = text.replace("e.g.","e.g.,")

        concepts = concept_discard_func[0](concepts,**concept_discard_func[1])

        sentences = self.__split_sentences(text)

        concepts_regex = "|".join(ConceptsUtils.get_concept_names(concepts))

        concepts_freq_dict = ConceptsUtils.create_freq_dict(concepts)
        concepts_objects = ConceptsUtils.get_concept_dict(concepts)

        html_new_sentences = []
        html_old_sentences = []

        new_sentences = sentences

        html_old_sentences += sentences
        html_new_sentences += sentences

        print(len(html_old_sentences),len(html_new_sentences))

        print(text)

        inc_len = 0
        replace = 0

        for sent_idx,sentence in enumerate(sentences):

            sentence_span = Span(inc_len,inc_len+len(sentence))

            matches = list(re.finditer(concepts_regex,sentence))

            if not matches:
                inc_len+=len(sentence)+2
                continue

            matches.sort(key= lambda c: concepts_freq_dict.get(c.string,0),reverse=True)

            relevant_match = matches[0] 

            concept = concepts_objects[relevant_match.group(0)]

            concept_span = Span(relevant_match.start(),relevant_match.end())
            
            new_concept = concept.alternatives[0]

            new_concept = concept.concept

            print("\nREPLACING:",concept.concept,"({},{})".format(concept_span.start,concept_span.end),"-->",new_concept)
            print("\nSENTENCE:",sentence)
            sentence_context = self.get_context(". ".join(new_sentences),
                                                    Span(sentence_span.start+relevant_match.start(),sentence_span.start+relevant_match.end()),
                                                )
            print("\nCONTEXT:",sentence_context)

            
            
            new_sentence = self.generate_eq_sentence(sentence,
                                                        sentence_context,
                                                        concept_span,
                                                        new_concept,)
            replace += 1
            
            if new_sentence: 
                new_sentences[sent_idx] = new_sentence
                
                html_old_sentences[sent_idx] = sentence[:concept_span.start]+"<font id='orig-{}' class='highlight k{}'><a href='#fake-{}'>".format(replace,replace,replace)+concept.concept+"</a></font>"+sentence[concept_span.end:]
                html_new_sentences[sent_idx] = new_sentence[:concept_span.start]+"<font id='fake-{}' class='highlight k{}'><a href='#orig-{}'>".format(replace,replace,replace)+new_concept+"</a></font>"+new_sentence[concept_span.start+len(new_concept):]

            else:
                new_sentences[sent_idx] = "non generata"

                html_old_sentences[sent_idx] = sentence[:concept_span.start]+"<font id='orig-{}' class='highlight k{}'><a href='#fake-{}'>".format(replace,replace,replace)+concept.concept+"</a></font>"+sentence[concept_span.end:]
                html_new_sentences[sent_idx] = new_sentence[:concept_span.start]+"<font id='fake-{}' class='highlight k{}'><a href='#orig-{}'>".format(replace,replace,replace)+new_concept+"</a></font>"+"<font class='not_gen'> NOT GENERATED</font>"


            print("\nNEW_SENTENCE:",new_sentence)

            

            inc_len+=len(new_sentence)+2

            
            
            
            print("\n\n\n")
        print(len(html_old_sentences),len(html_new_sentences))
        self.printDiff(html_old_sentences,html_new_sentences)

        return ". ".join(new_sentences)


if __name__ == "__main__":

    gpt_args = {"run_name":'1000_intro_csv_[simple]',"model_name":"345M","checkpoint_dir":"/content/checkpoint/1000_intro_csv_[simple]/"}
    tf.reset_default_graph()
    generator = GPTGenerator(gpt_args, sentence_separator=r"(([^e.g]\.) [a-zA-Z?-])", context_threshold=4000)

    text = '''The year of 2006 was exceptionally cruel to me – almost all of my papers submitted for that year conferences have been rejected. Not “just rejected” – unduly strong rejected. Reviewers of the ECCV (European Conference on Computer Vision) have been especially harsh: "This is a philosophical paper... However, ECCV neither has the tradition nor the forum to present such papers. Sorry..." O my Lord, how such an injustice can be tolerated in this world? However, on the other hand, it can be easily understood why those people hold their grudges against me: Yes, indeed, I always try to take a philosophical stand in all my doings: in thinking, paper writing, problem solving, and so no. In my view, philosophy is not a swear-word. Philosophy is a keen attempt to approach the problem from a more general standpoint, to see the problem from a wider perspective, and to yield, in such a way, a better comprehansion of the problem’s specificity and its interaction with other world realities. Otherwise we are doomed to plunge into the chasm of modern alchemy – to sink in partial, task-oriented determinations and restricted solution-space explorations prone to dead-ends and local traps. It is for this reason that when I started to write about “Machine Learning“, I first went to the Wikipedia to inquire: What is the best definition of the subject? “Machine Learning is a subfield of Artificial Intelligence“ – was the Wikipedia’s prompt answer. Okay. If so, then: “What is Artificial Intelligence?“ – “Artificial Intelligence is the intelligence of machines and the branch of computer science which aims to create it“ – was the response. Very well. Now, the next natural question is: “What is Machine Intelligence?“ At this point, the kindness of Wikipedia has been exhausted and I was thrown back, again to the Artificial Intelligence definition. It was embarrassing how quickly my quest had entered into a loop – a little bit confusing situation for a stubborn philosopher. Attempts to capitalize on other trustworthy sources were not much more productive (Wang, 2006; Legg & Hutter, 2007). For example, Hutter in his manuscript (Legg & Hutter, 2007) provides a list of 70-odd “Machine Intelligence“ definitons. There is no consensus among the items on the list – everyone (and the citations were chosen from the works of the most prominent scholars currently active in the field), everyone has his own particular view on the subject. Such inconsistency and multiplicity of definitions is an unmistakable sign of
    philosophical immaturity and a lack of a will to keep the needed grade of universality and generalization. It goes without saying, that the stumbling-block of the Hutter’s list of definitions (Legg & Hutter, 2007) are not the adjectives that was used – after all the terms “Artificial“ and “Machine“ are consensually close in their meaning and therefore are commonly used interchangeably. The real problem – is the elusive and indefinable term „Intelligence“. I will not try the readers’ patience and will not tediously explain how and why I had arrived at my own definition of the matters that I intend to scrutinize in this paper. I hope that my philosophical leanings will be generously excused and the benevolent readers will kindly accept the unusual (reverse) layout of the paper’s topics. For the reasons that would be explained in a little while, the main and the most general paper’s idea will be presented first while its compiling details and components will be exposed (in a discending order) afterwards. And that is how the proposed paper’s layout should look like:
    - Intelligence is the system’s ability to process information. This statement is true both for all biological natural systems as for artificial, human-made systems. By “information processing“ we do not mean its simplest forms like information storage and retrieval, information exchange and communication. What we have in mind are the high-level information processing abilities like information analysis and interpretation, structure patterns recognition and the system’s capacity to make decisions and to plan its own behavior. - Information in this case should be defined as a description – A language and/or an alphabet-based description, which results in a reliable reconstruction of an original object (or an event) when such a description is carried out, like an execution of a computer program. - Generally, two kinds of information must be distinguished: Objective (physical) information and subjective (semantic) information. By physical information we mean the description of data structures that are discernable in a data set. By semantic information we mean the description of the relationships that may exist between the physical structures of a given data set. - Machine Learning is defined as the best means for appropriate information retrieval. Its usage is endorsed by the following fundamental assumptions: 1) Structures can be revealed by their characteristic features, 2) Feature aggregation and generalization can be achieved in a bottom-up manner where final results are compiled from the component details, 3) Rules, guiding the process of such compilation, could be learned from the data itself. - All these assumptions validating Machine Learning applications are false. (Further elaboration of the theme will be given later in the text). Meanwhile the following considerations may suffice: - Physical information, being a natural property of the data, can be extracted instantly from the data, and any special rules for such task accomplishment are not needed. Therefore, Machine Learning techniques are irrelevant for the purposes of physical information retrieval. - Unlike physical information, semantics is not a property of the data. Semantics is a property of an external observer that watches and scrutinizes the data. Semantics is assigned to phisical data structures, and therefore it can not be learned
    straightforwardly from the data. For this reason, Machine Learning techniques are useless and not applicable for the purposes of smantic information extraction. Semantics is a shared convention, a mutual agreement between the members of a particular group of viewers or users. Its assignment has to be done on the basis of a consensus knowledge that is shared among the group members, and which an artificial semantic-processing system has to possess at its disposal. Accomodation and fitting of this knowledge presumes availability of a different and usually overlooked special learning technique, which would be best defined as Machine Teaching – a technique that would facilitate externally-prepared-knowledge transfer to the system’s disposal .
    These are the topics that I am interested to discuss in this paper. Obviously, the reverse order proposed above, will never be reified – there are paper organization rules and requirements, which none never will be allowed to override. They must be, thus, reverently obeyed. And I earnestly promiss to do this (or at least to try to do this) in this paper.
    '''

    concepts = [
        ReplacementConcept('physical information', ['extract physical information', 'important physical quantity', 'prior physical knowledge', 'physical knowledge', 'physical laws', 'physical chemistry properties', 'represents related physical properties', 'physical creative expressiveness', 'physical properties', 'physical objects compares', 'established physical laws', 'incorporate obvious physical invariances', 'physical parameter estimation', 'maintaining physical plausibility', 'shared physical environment', 'ensuring physical plausibility', 'physical problems', 'physical science problems', 'physical problem anymore', 'solve complicated physical problem', 'physical processes governing', 'physical plans', 'physical modeling offers', 'capture simple physical systems', 'violates physical laws/engineering limits', 'physical time', 'capturing physical reality', 'optimized physical plan', 'physical plan', 'physical science problem', 'physical laws/engineering limits', 'irreversible physical transformations', 'physical devices', 'original physical interpretation', 'physical theories offers', 'physical distance', 'many-body physical problems', 'violate physical constraints', 'concrete physical implementation', 'physical calculations', 'physical model based technique', 'physical modeling', 'model physical response', 'encoding physical principles', 'fundamental physical principles', 'onedimensional physical system', 'actual physical systems', 'physical systems due', 'art physical models', 'physical system'],16),
        ReplacementConcept("information processing",['distinctive information processing abilities', 'neural information processing systems', 'quantum information processing', 'brain information processing', 'nonstationary signal processing data', 'hand-crafted signal processing', 'natural language processing', 'judiciously apply natural language processing', 'correspondencebased natural language processing', 'discrete signal processing', 'topological signal processing', 'natural language processing applications', 'natural language processing methods', 'graph signal processing', 'signal processing', 'stationary signal processing data', 'information extraction', 'natural language processing application', 'natural language processing tasks', 'task-specific information results', 'œ information results', 'problem-specific information results', 'language processing system', 'message processing', 'side information lookup table', 'popular natural language processing data-sets', 'entity type information', 'acoustic signal processing', 'message processing time', 'side information lookup table built', 'language processing', 'natural language processing systems', 'randomly discards information', 'supports individualized information inflow', 'medical information retrieval', 'gene location information', 'audio processing system capable', 'music information retrieval researchers', 'interactive information retrieval', 'information spreading mechanism', 'information retrieval method', 'unstructured information retrieval', 'image information retrieval', 'audio processing', 'graphics processing', 'information retrieval', 'music information retrieval', 'input processing circuit', 'input layer processing', 'discard taskirrelevant information'],13),
        ReplacementConcept("semantic information",['higher-level semantic information', 'semantic information retained', 'capturing semantic information', 'semantic meaning', 'semantic meaning assigned', 'sentence’s semantic meaning', 'semantic entities acting', 'silent semantic relationships', 'music-specific semantic relationship', 'pre-trained general word semantic space', 'single semantic space', 'domain semantic space', 'semantic space', 'learned semantic relationships', 'incorporate semantic knowledge', 'semantic parsing literature', 'intermediate semantic space', 'semantic relations', 'capture semantic properties', 'regional latent semantic dependencies', 'semantic segmentation networks', 'finding semantic relations', 'semantic level', 'disease semantic similarity', 'semantic web', 'encodes semantic similarities', 'semantic web community', 'semantic translation problem', 'semantic image observation', 'semantic labels requires time', 'semantic labels requires significant time', 'semantic image editing', 'semi-supervised semantic segmentation', 'baseline neural semantic parser', 'semantic taxonomy modeling', 'semantic segmentation', 'semantic parser', 'distributed semantic structure', 'deep semantic segmentation', 'naturally support semantic taxonomy', 'neural semantic parser', 'imposing semantic monotonicity constraints', 'semantic parsers', 'perceived semantic closeness', 'semantic labels', 'generalizes deep semantic segmentation', 'semantic taxonomy', 'probabilistic latent semantic analysis', '/ probabilistic latent semantic indexing', 'semantic restriction iff'],11),
        ReplacementConcept("information content",['model content information', 'information theoretic content', 'explicit information collection', 'exploit class label information', 'class discriminative information', 'preserve class information', 'parameter setting information set', 'preparation information set', 'information theoretic set', 'holdout information set', 'data carried information', 'organizational group information', 'information set', 'network structure information', 'entity type information', 'information study deep networks', 'heterogeneous information network', 'information engineering university', 'domain information matrix', 'side information lookup table', 'pairwise mutual information matrix', 'information entropy signals', 'side information lookup table built', 'text line information', 'f0 information', 'extract task-relevant information', 'learner receives information', 'aggregating children information', 'information access 2018 quantum theory', 'learned adversarial information', 'inputoutput mutual information', 'information theoretic criteria proposed', 'incorporate temporal information', 'minimum mutual information', 'high-order correlation information', 'limiting mutual information', 'adding categorical information greatly improves', 'backward information', 'capture geometric information', 'information criteria', 'divulge specific information', 'function received information', 'maximize information gain', 'robot action information enable', 'sending undesirable information', 'visual information', 'content representation', 'provide crucial information', 'information probabilistic predictions provide', 'extracting pertinent information'],10),
    ]

    print(generator.fake_text(text,concepts))