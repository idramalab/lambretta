#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
'''
We use the ekphrasis text Pre-processor 
'''
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'user'],
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
   
)


# In[3]:


from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
'''
Example playground : https://demo.allennlp.org/open-information-extraction
'''
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")


# In[36]:


#import spacy library
import spacy
  
#load core english library
nlp = spacy.load("en_core_web_sm")

'''
Get Verbs , and supporting Arguments to every result. 
'''
regex = r'\[(ARG\d+|V):\s*([^]]+)\]'
def get_propostions_from_tweet(tweet):
    #First , segment by newline
    #Then , sentence segment tweet 
    propositions=[]
    lines=tweet.split("\n")
    for line in lines:
        doc=nlp(line)
        for sent in doc.sents:
            txt=sent.text
            txt=txt.lower()
            txt=" ".join(text_processor.pre_process_doc(txt))
            txt=txt.replace("<url>"," ").replace("<user>"," ").replace("<date>"," ").replace("<email>"," ")
            openie_results=predictor.predict(txt.strip())
            for result in openie_results["verbs"]:
                matches = re.findall(regex, result["description"])
                propstring=""
                for item in matches:
                    propstring+=item[1]+" "
                propositions.append(propstring)
    return propositions
    


# In[37]:


print(get_propostions_from_tweet(",President Trump just called in to the #ArizonaHearing - watching this on OANN - no other news network is carrying this hearing on electionfraud"))


# In[ ]:


'''
Now , score the propositions retrieved above through classify_proposition and filter out claims
'''


# In[20]:


from collections import defaultdict
import re
import requests
import json
'''
api_key : You can get the API key from https://idir.uta.edu/claimbuster/api/request/key/
Seems to be a pretty easy process with automatic approval 
'''
api_key = ""
def classify_proposition(input_proposition,threshold=0.490):
    '''
    Given a proposition that could potentially be a claim ; scores it using the ClaimBuster 
    Threshold could be adjusted based upon claims / projects. For our case, 0.490 worked after Precision-Recall evaluation
    '''
    api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/{input_proposition}"
    request_headers = {"x-api-key": api_key}
    api_response = requests.get(url=api_endpoint, headers=request_headers).json()
    score=api_response["results"][0]["score"]
    if score>=threshold: #if the score is above the threshold,then classify it as Claim
        return True
    else:
        return False


# In[ ]:




