# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:07:00 2020

@author: GIORGIO-DESKTOP
"""

#paper = {
#        "title": "De-anonymizing Social Networks",
#        "abstract": "Operators of online social networks are increasingly sharing potentially "
#            "sensitive information about users and their relationships with advertisers, application "
#            "developers, and data-mining researchers. Privacy is typically protected by anonymization, "
#            "i.e., removing names, addresses, etc. We present a framework for analyzing privacy and "
#            "anonymity in social networks and develop a new re-identification algorithm targeting "
#            "anonymized social-network graphs. To demonstrate its effectiveness on real-world networks, "
#            "we show that a third of the users who can be verified to have accounts on both Twitter, a "
#            "popular microblogging service, and Flickr, an online photo-sharing site, can be re-identified "
#            "in the anonymous Twitter graph with only a 12% error rate. Our de-anonymization algorithm is "
#            "based purely on the network topology, does not require creation of a large number of dummy "
#            "\"sybil\" nodes, is robust to noise and all existing defenses, and works even when the overlap "
#            "between the target network and the adversary's auxiliary information is small.",
#        "keywords": "data mining, data privacy, graph theory, social networking (online)"
#        }
#
#



# import classifier.classifier as CSO
# result = CSO.run_cso_classifier("The close connection between reinforcement learning (RL) algorithms and dynamic programming algorithms has fueled research on RL within the machine learning community.", modules = "both", enhancement = "first")
# CSO.
# print(type(result))
# print(result)
from multi_rake import Rake

text_en = "A surfactant composition for agricultural chemicals containing fatty acid polyoxyalkylene alkyl ether expressed by the following formula (I): ABCD; wherein the fatty acid polyoxyalkylene alkyl ether has a narrow ratio of 55% by mass or more, where the narrow ratio is expressed by the following formula: EFGH."

text_en = '''The close connection between reinforcement learning (RL) algorithms and dynamic programming algorithms has fueled research on RL within the machine learning community. Yet, despite increased theoretical understanding, RL algorithms remain applicable to simple tasks only. In this paper I use the abstract framework afforded by the connection to dynamic programming to discuss the scaling issues faced by RL researchers. I focus on learning agents that have to learn to solve multiple structured RL tasks in the same environment. I propose learning abstract environment models where the abstract actions represent “intentions” of achieving a particular state. Such models are variable temporal resolution models because in different parts of the state space the abstract actions span different number of time steps. The operational definitions of abstract actions can be learned incrementally using repeated experience at solving RL tasks. I prove that under certain conditions solutions to new RL tasks can be found by using simulated experience with abstract actions alone.'''

from nltk.corpus import stopwords
stopwords_en_set=set(stopwords.words('english'))


rake = Rake(language_code="en")

keywords = rake.apply(text_en)
keywords.sort(key=lambda x: x[0])
print(keywords)