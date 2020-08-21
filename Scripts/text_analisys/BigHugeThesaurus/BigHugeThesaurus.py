#! /usr/bin/python3

from enum import Enum
import requests
import json

class ApiFormats():

        JSON = "json"
        PHP = "php"
        XML = "xml"

class ApiConfig():

    def __init__(self,root_domain, api_path, api_key,return_format=ApiFormats.JSON):
        self.root_domain = root_domain
        self.api_path = api_path
        self.api_key = api_key
        self.format = return_format

    def getApiUrl(self):
        return "/".join([self.root_domain,self.api_path,self.api_key])

    def getWebsiteUrl(self):
        return self.root_domain

    def handleReturnCode(self,code):
        if code == 200:
            return self.handleStatus200()
        elif code == 500: 
            return self.handleStatus500()
        elif code == 404: 
            return self.handleStatus404()
        else:
            print("[WARNING] responses status code not valid")
            return (True,"[WARNING] responses status code not valid")

    def handleStatus200(self):
        return (True, "OK")

    def handleStatus404(self):
        return (False,"Word Not Found")

    def handleStatus500(self):
        return (False,"APIKEY expired/Query Limit/Other Errors")
    

class ThesaurusEntry():
    def __init__(self,synonyms):
        self.synonyms = synonyms

class Thesaurus:
    def __init__(self,api_config):
        assert(isinstance(api_config,ApiConfig))
        self.api_config = api_config

    def synonyms(self, word):

        word = str(word)

        return self.send_request(word)

    def parse_json_entry(self,json):

        syns = []

        if(syns is None): return ThesaurusEntry([])
        elif(isinstance(json,dict)):
            noun_syns = json.get("noun",{}).get("syn",[])
            verb_syns = json.get("verb",{}).get("syn",[])

            syns = noun_syns+verb_syns
        elif(isinstance(json,list)):
            syns = json
        return ThesaurusEntry(syns)

    def send_request(self,word):
        url = "/".join([self.api_config.getApiUrl(),word,str(self.api_config.format)])

        response = requests.get(url)
        isSuccessfull, msg = self.api_config.handleReturnCode(response.status_code)
        if not isSuccessfull: 
            print("[RESPONSE ERROR] word '{}' has returned with message '{}'".format(word,msg))
            return []

        return self.parse_json_entry(response.json()).synonyms

class ThesaurusWebsite(Thesaurus):

    def parse_json_entry(self,json):
        syns = []

        if(isinstance(json,dict)):
            data = json.get("data",[])
            if data is None: return ThesaurusEntry([])

            for entry in data:
                for syn in entry.get("synonyms",[]):
                    term = syn.get("term",None)
                    if term: syns.append(term)

        return ThesaurusEntry(syns)

    def send_request(self,word):
        url = "/".join([self.api_config.getWebsiteUrl(),word])+"?limit=10"

        response = requests.get(url)
        
        try: jsn = response.json()
        except json.decoder.JSONDecodeError:
            print("[RESPONSE ERROR] word '{}' has returned bad json".format(word))
            return []

        isSuccessfull, msg = self.api_config.handleReturnCode(response.status_code)
        if not isSuccessfull: 
            print("[RESPONSE ERROR] word '{}' has returned with message '{}'".format(word,msg))
            return []

        print(response.text,"\n")
        return self.parse_json_entry(jsn).synonyms


if __name__ == "__main__":

    api_config = ApiConfig(
        "https://words.bighugelabs.com",
        "api/2",
        "f5acf68d71dad138a4374ec8e7c3522a",
    )

    config_website = ApiConfig(
        "https://tuna.thesaurus.com/relatedWords",
        "",
        "",
    )

    

    ts = Thesaurus(api_config)
    tsw = ThesaurusWebsite(config_website)

    syns = tsw.synonyms("")
    print(syns)
