# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:41:12 2020

@author: GIORGIO-DESKTOP
"""

from io import StringIO
import os
import re

from pdfminer.converter import TextConverter #requiremnt
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfparser import PDFSyntaxError

import multiprocessing as mp
import psutil


knownTitles = ["ABSTRACT","Abstract","introduction","Introduction", "INTRODUCTION","Conclusion","conclusion", "contents","Contents", "CONTENTS","Preliminaries", "Related Works","Related Work", "References"]

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



def pdf_to_txt(path):
    if(not path): raise Exception("Empty path is not valid")
    output_string = StringIO()
    with open(path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams(all_texts=False,line_overlap=0,word_margin=0.1))
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
    return output_string.getvalue()
        

class PaperSections():
    PAPER_ABSTRACT = "abstract"
    PAPER_INTRO    = "introduction"
    PAPER_CORPUS   = "corpus"
    PAPER_CONCLUSION = "conclusion"

    @staticmethod
    def as_list(): return [PaperSections.PAPER_ABSTRACT, PaperSections.PAPER_INTRO, PaperSections.PAPER_CORPUS, PaperSections.PAPER_CONCLUSION]

class RawPaper():
    def __init__(self,text="",path=""):
        self.full_text = text #complete text
        self.path=path
        self.contents = [] #

        self.sections_indexes= {
            PaperSections.PAPER_ABSTRACT : (None,None),
            PaperSections.PAPER_INTRO : (None,None),
            PaperSections.PAPER_CORPUS : (None,None),
            PaperSections.PAPER_CONCLUSION : (None,None)
        }
        
        self.sections_dict = {
            PaperSections.PAPER_ABSTRACT : "",
            PaperSections.PAPER_INTRO : "",
            PaperSections.PAPER_CORPUS : "",
            PaperSections.PAPER_CONCLUSION : ""
        }

    def fromSections(sections):
        full_text = "\n".join([sections[k] for k in sections.keys()])
        rp = RawPaper(text=full_text)
        for section in PaperSections.as_list():
            rp.sections_dict[section] = sections[section]
        return rp
    
    def fromPdf(path=""):
        # print("processing",path.split("\\")[-1],end="")
        if(not path): raise Exception("Empty path is not valid")
        p = RawPaper(path=path)
        p.full_text = re.sub(r'\f',"\n", pdf_to_txt(p.path))
        p.full_text = removeWordWrap(p.full_text)
        p.extract_content_table()
        p.extract_sections()
        
        return p
    
    def dump(self, outPath="", includeTableOfContents=True, mode="wb"):
        if(not outPath): raise Exception("Empty path is not valid")
        text=""
        with open(outPath,mode) as fout:
            text+= "PAPER PATH: {}\n\n".format(self.path)

            if(includeTableOfContents): text += "\n"+"="*16+"TABLE OF CONTENTS"+"="*16+"\n"+"\n".join([str(x[1])+":\t"+x[0] for x in self.contents])+"\n"+"="*40+"\n"

            text +="="*20+"SECTION INDEXES"+"="*20+"\n"
            text += str(self.sections_indexes)+"\n"
            text+="="*40+"\n"
            
            for sec in self.sections_dict.keys():
                text +="="*20+sec.upper()+"="*20+"\n"
                text += self.sections_dict[sec]+"\n"
                text+="="*40+"\n"
            
            text +="="*20+"COMPLETE TEXT"+"="*20+"\n"
            text += self.full_text

            text = removeEOL(pdf_to_txt(self.path))

            fout.write((text).encode('utf-8'))

    def extract_content_table(self):
        lines = []
        for idx,line in enumerate(self.full_text.split("\n")):
            if(RawPaper.isTitle(line,idx,self.contents)): lines.append((line,idx))

        hasContentTable = False
        counter = 0
        startIdx = 0
        for idx,line in enumerate(self.contents):
            if (line[0].lower().endswith("introduction")): counter +=1
            if (hasContentTable and counter==2): 
                startIdx = idx
                break
            if  line[0].lower().endswith("contents"): hasContentTable=True
        self.contents = self.contents[startIdx:]
        return lines
        
    def isTitle(sentence,idx,contentsList=[]):
        string = re.sub(r'\b(the|a|and|an|for|nor|but|or|yet|so|in|on|for|up|of|to|with|The|A|And|An|For|Nor|But|Or|Yet|So|In|On|For|Up|Of|To|With)\b','',sentence)
        hasListEnum = bool(re.search(r'^([0-9]+[\.]|[0-9]+(\.[0-9]+)*|M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})(\.M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})+)+) ',string))
        if(hasListEnum): string = re.sub("^[^ ]* ","",string)
        isTitleCase = not bool(re.search(r'\b[a-z][^ ]',string))
        notBeginEndsInPunctualization = bool(re.search(r'^[^,.=]{1}.*[^,.=]{1}$',string))
        hasMinLen = not len(string)<3
        hasMinLetters = bool(re.search(r'[a-zA-Z]{1}',string))
        isAbstract = bool(re.search(r'^(Abstract|ABSTRACT)(\. |\-|—)',string))

        isTitle = (hasMinLen and isTitleCase and hasMinLetters and (not RawPaper.isCitation(sentence)) and notBeginEndsInPunctualization) or isAbstract
        if((hasListEnum and isTitle) or (string in knownTitles) or (isAbstract)): contentsList.append((sentence,idx))
        # if(idx==118 and False): ##DEBUG
            # print(sentence+"->"+string," ",idx," ",hasListEnum," ",hasMinLen," ", isTitleCase ," ",  hasMinLetters ," ",  (not Paper.isCitation(sentence)) ," ",  notBeginEndsInPunctualization)
            # print(contentsList)
        return isTitle
    
    def isCitation(sentence):
        string = re.sub(r'\b(the|a|and|an|for|nor|but|or|yet|so|in|on|for|up|of|to|with|The|A|And|An|For|Nor|But|Or|Yet|So|In|On|For|Up|Of|To|With)\b','',sentence)
        isTitleCase = not bool(re.search(r'\b[a-z][^ ]',string))
        startsWithCitation = bool(re.search(r'^\[[0-9]+\] ',string))
        return isTitleCase and startsWithCitation
    
    def extract_abstract(self):
        abs_start_line = None
        abs_end_line = None
        lines = self.full_text.split("\n")
        
        for idx,c in enumerate(self.contents):
            sentence = c[0]

            if(sentence.lower().endswith("abstract") or sentence.lower().startswith("abstract. ") or sentence.lower().startswith("abstract—") or sentence.lower().startswith("abstract-")):
                abs_start_line = c[1]
                abs_end_line = self.contents[idx+1][1]
                break

        if(not abs_start_line or not abs_end_line): return ""

        self.sections_indexes[PaperSections.PAPER_ABSTRACT] = (abs_start_line,abs_end_line)

        return "\n".join(lines[abs_start_line:abs_end_line])

    def extract_introduction(self):
        intro_end_line = -1
        intro_start_line = -1
        intro_found = False
        
        lines = self.full_text.split("\n")
        
        for idx,c in enumerate(self.contents):
            sentence = c[0]
            hasListEnum = bool(re.search(r'^([0-9]+[\.]|[0-9]+(\.[0-9]+)*|M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})(\.M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})+)+) ',sentence))
            

            if(sentence.lower().endswith("introduction") and not intro_found): 
                intro_found = True
                intro_start_line = c[1]
                if((len(self.contents)-1-idx)>0 ):
                    intro_end_line = self.contents[idx+1][1]
                    # self.intro =  self.intro = "\n".join(lines[c[1]:self.contents[idx+1][1]])
                    # return  
            elif(intro_found and hasListEnum and (sentence.startswith("I.") or sentence.startswith("1.") and (len(self.contents)-1-idx)>0 )):
                intro_end_line = self.contents[idx+1][1]
            elif(intro_found): break

        self.sections_indexes[PaperSections.PAPER_INTRO] = (intro_start_line,intro_end_line)
                
        return "\n".join(lines[intro_start_line:intro_end_line])

    def extract_corpus(self):
        '''
            asert that introduction and conclusion are already extracted
        '''

        if None in self.sections_indexes[PaperSections.PAPER_INTRO] or self.sections_indexes[PaperSections.PAPER_CONCLUSION][0] is None: 
            raise Exception("Must extract introduction and conclusion before corpus")

        corpus_start_line = self.sections_indexes[PaperSections.PAPER_INTRO][1]
        corpus_end_line = self.sections_indexes[PaperSections.PAPER_CONCLUSION][0]

        lines = self.full_text.split("\n")
        
        self.sections_indexes[PaperSections.PAPER_CORPUS] = (corpus_start_line,corpus_end_line)
                
        return "\n".join(lines[corpus_start_line:corpus_end_line])

    def extract_conclusion(self):
        conc_start_line = -1
        conc_found = False
        
        lines = self.full_text.split("\n")
        
        for idx,c in enumerate(self.contents):
            sentence = c[0]

            if(sentence.lower().endswith("conclusion") or sentence.lower().endswith("future work") or sentence.lower().endswith("conclusions")):
                conc_start_line = c[1]
        
        self.sections_indexes[PaperSections.PAPER_CONCLUSION] = (conc_start_line,None)     
                
        return "\n".join(lines[conc_start_line:])

    def extract_sections(self):
        
        self.sections_dict[PaperSections.PAPER_ABSTRACT] = self.extract_abstract()
        self.sections_dict[PaperSections.PAPER_INTRO] = self.extract_introduction()
        self.sections_dict[PaperSections.PAPER_CONCLUSION] = self.extract_conclusion()
        self.sections_dict[PaperSections.PAPER_CORPUS] = self.extract_corpus()

        return self.sections_dict



def removeWordWrap(text):
    return re.sub(r'-\n','',text)

def removeEOL(text):
    return re.sub(r'\n',' ',text)


class RepoExportTypes():
    TYPE_CSV = "csv"
    TYPE_JSON = "json"

class RepositoryExtractor():

    def __init__(self, path, log_path="./paper_to_txt.log"):
        self.repo_path = path 
        self.papers = []
        self.log_path = log_path
        self.file_num = -1

    def extract(self,limit=None):
        dir_files = os.listdir(self.repo_path)
        self.file_num = len(dir_files)
        failed_files = []
        failed = 0
        printProgressBar(0,self.file_num,prefix="Extracting txt",suffix="---",length=50)
        for idx,file in enumerate(dir_files):
            if(limit and idx > limit): break
            if(not file.endswith(".pdf")): continue
            full_path = os.path.join(self.repo_path, file)
            try:
                p = RawPaper.fromPdf(path=full_path)
            except PDFSyntaxError as e:
                failed_files.append(file)
                failed += 1
                pass
            self.papers.append(p)
            printProgressBar(idx+1,self.file_num,prefix="Extracting txt [{}/{}]".format(idx+1,self.file_num),suffix=file[:20]+"[{} failed]".format(failed)+"   ",length=50)
        print()

    def extractMP(self,limit=None,processes=None):
        dir_files = [f for f in os.listdir(self.repo_path) if f.endswith(".pdf")]
        failed_files = []
        failed = 0

        
        if(limit): inputs = list(enumerate(dir_files[:limit]))
        else: inputs = list(enumerate(dir_files))

        self.file_num = len(inputs)

        with mp.Pool(processes=processes) as pool:
            results = pool.starmap(self.extractSingle,inputs)
        for paper in results:
            if(paper): self.papers.append(paper)
            else: failed+=1

        printProgressBar(self.file_num,self.file_num,prefix="Extracting txt ~[{}/{}]".format(self.file_num,self.file_num),suffix="completed",length=50)
        return failed

    def extractSingle(self,idx,file,print_mod=10):
        if(not file.endswith(".pdf")): return None
        full_path = os.path.join(self.repo_path, file)

        try:
            p = RawPaper.fromPdf(path=full_path)
        except PDFSyntaxError as e:
            return None
        
        if(psutil.Process().cpu_num() == 0):
            printProgressBar(idx+1,self.file_num,prefix="Extracting txt ~[{}/{}]".format(idx+1,self.file_num),suffix="",length=50)


        return p

    def export(self,csv_path,section=PaperSections.PAPER_INTRO, type=RepoExportTypes.TYPE_CSV, remove_word_wrap=True):
        with open(csv_path,"wb") as csv_out:
            csv_text = "\f\n".join([removeEOL(x.sections_dict[section]) for x in self.papers])
            csv_text = removeWordWrap(csv_text)
            csv_out.write(csv_text.encode("utf-8"))

    def exportSections(self,csv_dir,sections=PaperSections.as_list(),type=RepoExportTypes.TYPE_CSV, remove_word_wrap=True):
        for section in sections:
            path = os.path.join(csv_dir,section+".csv")
            self.export(path,section=section)


if __name__ == "__main__":
    path = "../datasets/downloaded_test/"

    csv_path = "../Results/extraction/"

    repo = RepositoryExtractor(path)

    repo.extract()
    repo.exportSections(csv_path)
    for i,p in enumerate(repo.papers):
        p.extract_sections()
        p.dump(outPath="../Results/extraction/paper-%s.txt"%i)
        
