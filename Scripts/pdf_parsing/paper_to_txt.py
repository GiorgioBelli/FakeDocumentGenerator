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


knownTitles = ["Abstract","introduction","Introduction", "INTRODUCTION","Conclusion","conclusion", "contents","Contents", "CONTENTS","Preliminaries", "Related Works","Related Work", "References"]

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
        

class Paper():
    def __init__(self,text="",path=""):
        self.text = text
        self.path=path
        self.contents = []
        
        self.intro = ""
    
    def fromPdf(path=""):
        # print("processing",path.split("\\")[-1],end="")
        if(not path): raise Exception("Empty path is not valid")
        p = Paper(path=path)
        p.text = re.sub(r'\f',"\n", pdf_to_txt(p.path))
        p.text = removeWordWrap(p.text)
        p.extract_sections()
        p.extract_introduction()

        # print("\t[done]")
        
        return p
    
    def dump(self, outPath="", includeSections=False, mode="wb"):
        if(not outPath): raise Exception("Empty path is not valid")
        text=""
        with open(outPath,mode) as fout:
#            if(includeSections): text = "="*16+"SECTIONS"+"="*16+"\n"+"\n".join([str(x[1])+":\t"+x[0] for x in p.extract_sections()])+"\n"+"="*40+"\n"
            if(includeSections): text += "\n"+"="*16+"COMMON FORMAT"+"="*16+"\n"+"\n".join([str(x[1])+":\t"+x[0] for x in self.contents])+"\n"+"="*40+"\n"
            
            self.extract_introduction()
            text +="="*20+"INTRO"+"="*20+"\n"
            text += self.intro+"\n"
            text+="="*40+"\n"
            
            text +="="*20+"COMPLETE TEXT"+"="*20+"\n"
            text += self.text
            fout.write((text).encode('utf-8'))

        
    def extract_sections(self):
        lines = []
        for idx,line in enumerate(self.text.split("\n")):
            if(Paper.isTitle(line,idx,self.contents)): lines.append((line,idx))

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

        isTitle = hasMinLen and isTitleCase and hasMinLetters and (not Paper.isCitation(sentence)) and notBeginEndsInPunctualization
        if((hasListEnum and isTitle) or (string in knownTitles)): contentsList.append((sentence,idx))
        # if(idx==118 and False): ##DEBUG
            # print(sentence+"->"+string," ",idx," ",hasListEnum," ",hasMinLen," ", isTitleCase ," ",  hasMinLetters ," ",  (not Paper.isCitation(sentence)) ," ",  notBeginEndsInPunctualization)
            # print(contentsList)
        return isTitle
    
    def isCitation(sentence):
        string = re.sub(r'\b(the|a|and|an|for|nor|but|or|yet|so|in|on|for|up|of|to|with|The|A|And|An|For|Nor|But|Or|Yet|So|In|On|For|Up|Of|To|With)\b','',sentence)
        isTitleCase = not bool(re.search(r'\b[a-z][^ ]',string))
        startsWithCitation = bool(re.search(r'^\[[0-9]+\] ',string))
        return isTitleCase and startsWithCitation
    
    def extract_introduction(self):
        intro_end_line = -1
        intro_start_line = -1
        intro_found = False
        
        lines = self.text.split("\n")
        
        for idx,c in enumerate(self.contents):
            sentence = c[0]
            hasListEnum = bool(re.search(r'^([0-9]+[\.]|[0-9]+(\.[0-9]+)*|M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})(\.M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})+)+) ',sentence))
            

            if(sentence.lower().endswith("introduction") and not intro_found): 
                intro_found = True
                intro_start_line = c[1]
                if((len(self.contents)-1-idx)>0 ):
                    intro_end_line = self.contents[idx+1][1]
#                    self.intro =  self.intro = "\n".join(lines[c[1]:self.contents[idx+1][1]])
#                    return  
            elif(intro_found and hasListEnum and (sentence.startswith("I.") or sentence.startswith("1.") and (len(self.contents)-1-idx)>0 )):
                intro_end_line = self.contents[idx+1][1]
            elif(intro_found): break
                
        self.intro = "\n".join(lines[intro_start_line:intro_end_line])

def removeWordWrap(text):
    return re.sub(r'-\n','',text)

if __name__ == "__main__":
    pdf_repo = "C:\\Users\\GIORGIO-DESKTOP\\Documents\\Universita\\Tesi\\datasets\\downloaded_papers\\"
    papers = []
    l = []
    dir_files = os.listdir(pdf_repo)
    file_num = len(dir_files)
    failed_files = []
    failed = 0
    printProgressBar(0,file_num,prefix="Extracting txt",suffix="---",length=50)
    for idx,file in enumerate(dir_files):
        if(not file.endswith(".pdf")): continue
        full_path = os.path.join(pdf_repo, file)
        failed_files.append(file)
        try:
            p = Paper.fromPdf(path=full_path)
        except PDFSyntaxError as e:
            failed_files.append(file)
            failed += 1
            pass
        papers.append(p)
        printProgressBar(idx+1,file_num,prefix="Extracting txt [{}/{}]".format(idx+1,file_num),suffix=file[:20]+"[{} failed]".format(failed)+"   ",length=50)

    if failed_files:
        with open("paper_to_txt.log","wb") as csv_out:
            log_text = "\n".join(failed_files)
            csv_out.write(log_text.encode("utf-8"))
    
    csv_path = "C:\\Users\\GIORGIO-DESKTOP\\Desktop\\intros.csv"

    with open(csv_path,"wb") as csv_out:
        csv_text = "\f\n".join([x.intro for x in papers])
        csv_out.write(csv_text.encode("utf-8"))
        

    #TODO trovare un modo di scaricaricare molti pdf (libgen o sci-hub) se non di CS di qualsiasi cosa sceintifica
    