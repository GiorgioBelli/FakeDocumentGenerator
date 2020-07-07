# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:58:02 2020

@author: GIORGIO-DESKTOP
"""

#from docx import Document
#
#
#doc = Document("C:\\Users\\GIORGIO-DESKTOP\\Desktop\\arxiv_paper.docx")
#outfile = "C:\\Users\\GIORGIO-DESKTOP\\Desktop\\out.txt"
#
#out_fp = open(outfile, "w",encoding="utf-8");
#
#string = ""
#for i,p in enumerate(doc.paragraphs):
##    string += "\np{}:\n\t{}".format(i,p.text)
#    string = p.text
##    print(string[-1])
#    
#    if(p.style.name == "Heading 1"):
#        out_fp.write("\n"+p.style.name+"\n"+string)
#
##    if(string and (string[-1] == '.'):
##        out_fp.write("\n"+string)
##        string = ""
#
#
#out_fp.close()

from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
clf = DecisionTreeClassifier(max_depth = 1000)
x_train,x_test,y_train,y_test = train_test_split(x,y)

fig = clf.fit(x_train,y_train)
tree.plot_tree(fig)
plt.show()