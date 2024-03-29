#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import time
from datetime import datetime
from dateutil.parser import parse
import re
import itertools
import json


# In[ ]:


from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-mpnet-base-v2")
#Select other models from here : https://www.sbert.net/docs/pretrained_models.html


# In[ ]:


f=open("dict_claim_query.json")#This comes from candidate_query_generator
keyword_queries=json.load(f)


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


# In[ ]:


path_fetch_results="/data/ppaudeldata/VoterFraud/awk_output_test.json"# Same as awk_output_export_path on fetch_results.py file 
path_write_results_scoring="results_scoring.json"
xx=open(path_fetch_results) #Replace with full
'''
Calculate semantic similarity based features ; for each claim and the result set generated by the candidate queries.
We use SentenceTransfomer model for scoring two types of similarity ; within the results , and between claim and results
'''
for x in xx:
    try:
        x=x.rstrip()
        jsx=json.loads(x)
        query=jsx["keyword"]
        query_split=query.split(" ")
        query=' '.join(query_split)
        print("Working on ... ",query)
        data=jsx["data"]
        if  len(data)==0:
            print("Empty results for ...",query)
            for claim in keyword_queries[query]:
                claim_query_distance=util.cos_sim(model.encode(query),model.encode(claim)).cpu().detach().numpy().tolist()
                with open(path_write_results_scoring,"a+") as of:
                    json.dump({"claim":claim,"query":query,"distance_query":[],"distance_results":[],"distance_query_claim":claim_query_distance},of)
                    of.write("\n")
            continue
        sorted_data=sorted(data, key=lambda d: parse(d['stamp']))
        total=len(sorted_data)
        #Creating spanning subset 
        #Computing sentence embeddings take time and the sampled data could grow very really large ( in cases of noise). Thus we can limit up to 1000 items on each of the joins 
        left_join=sorted_data[0:int(0.2*len(sorted_data))][0:1000]
        right_join=sorted_data[-int(0.2*len(sorted_data)):][0:1000]
        mid=int(len(sorted_data)*0.5)
        mid_join=sorted_data[mid-(int(total*0.1)):mid+(int(total*0.1))][0:1000]
        #Sliced data is spanning subset discussed in the paper 
        sliced_data=left_join+mid_join+right_join
        texts=[]
        cleaned_effective=[]
        for item in sliced_data:
            txt=item["text"]
            txt=txt.lower()
            txt=" ".join(text_processor.pre_process_doc(txt))
            txt=txt.replace("\n"," ")
            txt=txt.replace("<url>"," ").replace("<user>"," ").replace("<date>"," ").replace("<email>"," ")
            cleaned_effective.append(txt.strip())
        dists_within=[]
        for a, b in itertools.combinations(cleaned_effective, 2):
            dists_within.append(util.cos_sim(model.encode(a),model.encode(b)).cpu().detach().numpy().tolist())
            #dists_within.append(model.wmdistance(a,b))
        for claim in keyword_queries[query]:
            claimtxt=claim.lower()
            claimtxt=claimtxt.lower()
            claimtxt=" ".join(text_processor.pre_process_doc(claimtxt))
            claimtxt=claimtxt.replace("\n"," ")
            claimtxt=claimtxt.replace("<url>"," ").replace("<user>"," ").replace("<date>"," ").replace("<email>"," ")
            dists_btn=[]
            for rslt in cleaned_effective:
                dists_btn.append(util.cos_sim(model.encode(rslt),model.encode(claimtxt)).cpu().detach().numpy().tolist())
            with open(path_write_results_scoring,"a+") as of:
                claim_query_distance=util.cos_sim(model.encode(query),model.encode(claim)).cpu().detach().numpy().tolist()
                json.dump({"claim":claim,"query":query,"distance_query":dists_btn,"distance_results":dists_within,"distance_query_claim":claim_query_distance},of)
                of.write("\n")
    except Exception as err:
        print(err)
        ff=open("error_generate_semantic.txt","a+")
        ff.write("Error on ..."+jsx["keyword"]+"..."+str(err))
        ff.write("\n")
        ff.close()

