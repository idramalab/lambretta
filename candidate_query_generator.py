#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords 

stop_words=set(stopwords.words('english'))


# In[ ]:


'''
query_tweets count
'''


# In[ ]:


dd=0
import pandas as pd
df=pd.read_csv("training_claims.csv")
candidates=[]
from collections import defaultdict
count_grams=defaultdict(int)
candidate_claim=defaultdict(list)
for index, row in df.iterrows():
    if 1:
        claim=row['claim'].strip().lower().replace("  "," ")
        raw_cleaned_claim=[]
        for item in claim.split(" "):
            if item in stop_words:
                continue
            raw_cleaned_claim.append(item.strip())
        cleaned_tweet=' '.join(raw_cleaned_tweet).strip()
        for idx in range(2,5):#Ranging ngram from 2-5 
            NGRAMS=ngrams(sequence=raw_cleaned_claim, n=idx)
            for gram in NGRAMS:
                allgrams=[]
                for inner in gram:
                    allgrams.append(inner)
                    count_grams[inner]+=1
                clean=' '.join(allgrams)
                candidates.append(clean)
                candidate_claim[clean].append(claim)


# In[ ]:


import json
'''
candidate_claim dictionary contains all ngram candidate queries ; and corresponding set of claims associated with it 
One ngram candidate query could be linked with multiple claims ; hence the value of this dictionary is a list 
'''
with open('dict_claim_query.json', 'w') as of:
    of.write(json.dumps(candidate_claim))


# In[ ]:


'''
candidate_queries is a list of n-gram combinations ranging from 2-5 that we will use to query (awk) the data store , 
and generate the subsequent feature values 
'''
ff=open("candidate_queries.txt","w")
for item in candidates:
    ff.write(item)
    ff.write("\n")
ff.close()


# In[ ]:




