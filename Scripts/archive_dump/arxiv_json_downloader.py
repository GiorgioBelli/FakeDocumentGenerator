'''
json structure:
[
    {
        "author": "[{'name': 'Marcus Hutter'}, {'name': 'Jan Poland'}]",
        "day": 16,
        "id": "cs/0504078v1",
        "link": "[{'rel': 'alternate', 'href': 'http://arxiv.org/abs/cs/0504078v1', 'type': 'text/html'}, {'rel': 'related', 'href': 'http://arxiv.org/pdf/cs/0504078v1', 'type': 'application/pdf', 'title': 'pdf'}]",
        "month": 4,
        "summary": "When applying aggregating strategies to Prediction with Expert Advice, the learning rate must be adaptively tuned. The natural choice of sqrt(complexity/current loss) renders the analysis of Weighted Majority derivatives quite complicated. In particular, for arbitrary weights there have been no results proven so far. The analysis of the alternative \"Follow the Perturbed Leader\" (FPL) algorithm from Kalai & Vempala (2003) (based on Hannan's algorithm) is easier. We derive loss bounds for adaptive learning rate and both finite expert classes with uniform weights and countable expert classes with arbitrary weights. For the former setup, our loss bounds match the best known results so far, while for the latter our results are new.",
        "tag": "[{'term': 'cs.AI', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}, {'term': 'cs.LG', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}, {'term': 'I.2.6; G.3', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}]",
        "title": "Adaptive Online Prediction by Following the Perturbed Leader",
        "year": 2005
    }
    ....
]

'''
import requests
import json
import sys
import re
import urllib

class Link():
    def __init__(self, rel, href, rel_type):
        self.rel = rel 
        self.href = href 
        self.rel_type = rel_type 

    def __str__(self):
        return "rel: {}\nhref: {}\ntype: {}\n".format(
            self.author, 
            self.day, 
            self.id,
        )


class Entry():
    def __init__(self, author, day, id, link, month, summary, tag, title, year):
        self.author = author 
        self.day = day 
        self.id = id 
        self.link = [Link(l["rel"],l["href"],l["type"],) for l in link] 
        # self.link = link 
        self.month = month 
        self.summary = summary 
        self.tag = tag 
        self.title = title 
        self.year = year 


    def __str__(self):
        return "author: {}\nday: {}\nid: {}\nlink: {}\nmonth: {}\nsummary: {}\ntag: {}\ntitle: {}\nyear: {}\n".format(
            self.author, 
            self.day, 
            self.id, 
            self.link, 
            self.month, 
            self.summary, 
            self.tag, 
            self.title, 
            self.year 
        )


def link_string_to_list(string):
    return json.loads(string.replace("'","\""))

if __name__ == "__main__":


    data = []
    papers = []

    print("Loading JSON", end="", flush=True)

    with open(sys.argv[1]) as inf:
        data = json.load(inf)

    print("\t OK", flush=True)


    print("Creating data structures", end="", flush=True)
    
    for entry in data:
        papers.append(Entry(entry["author"], entry["day"], entry["id"], link_string_to_list(entry["link"]), entry["month"], entry["summary"], entry["tag"], entry["title"], entry["year"]))
    print("\t OK", flush=True)
    
    print("Downloading resources:", flush=True)
    
    counter = 0

    for index,entry in enumerate(papers):
        resource_name = entry.id
        url = "https://arxiv.org/e-print/"+ resource_name

        print("\t",resource_name,"\t retrieving...", end="", flush=True)

        try:
            # urllib.request.urlretrieve(url, "./{}".format(resource_name))
            r = requests.head(url)
            if(not r.ok): 
                print("\t request error {}".format(r.status_code),flush=True)
                counter+=1
                continue
        except urllib.error.HTTPError:
            print("\t request error",flush=True)
            continue

        print("\t OK",flush=True)

    print("error: ", counter)
