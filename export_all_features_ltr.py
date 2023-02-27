#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###Textrank 
from collections import OrderedDict
import numpy as np
import spacy
from spacy.symbols import ORTH

#Special cases. of tokenization for frequently occuring tokens that aren't handled well by the default Spacy tokenizer
nlp = spacy.load('en_core_web_sm')
special_case_us = [{ORTH: "u.s"}]
special_case_us_2=[{ORTH:"u.s."}]
special_case_mail_in=[{ORTH:"mail-in"}]
special_case_new=[{ORTH:"new"}]
nlp.tokenizer.add_special_case("u.s",special_case_us)
nlp.tokenizer.add_special_case("u.s.",special_case_us_2)
nlp.tokenizer.add_special_case("mail-in",special_case_mail_in)
nlp.tokenizer.add_special_case("new",special_case_new)

'''
Adapted from this source : 
https://gist.github.com/DeepanshKhurana/91b123ba2e5e0be739370d2021cc820c

'''
class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if 1:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        rslt={}
        for i, (key, value) in enumerate(node_weight.items()):
            #print(key + ' - ' + str(value))
            rslt[key]=value
            if i > number:
                break
        return rslt
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        #self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy 
from sklearn.feature_extraction.text import TfidfTransformer


# In[ ]:


'''
Create TF-IDF vectorizer 
'''
#First, generate the document by accumulating all the claims 
docs=[]
for item in jsx:
    tr4w = TextRank4Keyword()
    claim=item["claim"]
    docs.append(jsx["claim"])
cv=CountVectorizer()
word_count_vector=cv.fit_transform(docs)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
feature_names=cv.get_feature_names()


# In[ ]:


def generate_tf_idf(claim,query):
    '''
    For every claim and query , return the TF-IDF scores of each terms in the query 
    '''
    doc=claim
    #First, generate tf-idf for the given document 
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    quer_split=query.split(" ")
    tf_idf_scores=[]
    for item in sorted_items:
        if feature_names[item[0]] in quer_split: #if token is part of queyr 
            tf_idf_scores.append(item[1])
    return tf_idf_scores


# In[ ]:


def generate_textrank(claim,query):
    '''
    For every claim and query , return the TextRank scores of each terms in the query 
    '''
    tr4w = TextRank4Keyword()
    tr4w.analyze(claim, window_size=3, lower=True)
    claim_split=claim.split(" ")
    rslt=tr4w.get_keywords(len(claim_split))
    textrank_scores=[]
    quer_split=query.split(" ")
    for itemx in quer_split:
        if itemx in rslt:
            textrank_scores.append(rslt[itemx])
    return textrank_scores


# In[ ]:


import pandas as pd
df=pd.read_csv("training_claims.csv")
claim_groundtruth={}
for index, row in df.iterrows():
    claim=row['claim'].strip().lower().replace("  "," ")
    groundtruth=row['groundtruthkeyword'].strip().lower()
    claim_groundtruth[claim]=groundtruth


# In[ ]:


path_import_results_scoring="results_scoring.json"
with open(path_import_results_scoring) as infile:
    semantic_results_scored=json.load(infile)


# In[ ]:


'''
semantic_results_scored already has features from semantic similarities.
We'll now  Text Rank and TF-IDF related features to complete the feature set 
'''


# In[ ]:


from collections import defaultdict
claim_queries=defaultdict(list)
for item in semantic_results_scored:
    ltr_feature={}
    query=item["query"]
    claim=item["claim"]
    ltr_feature["query"]=query
    ltr_feature["claim"]=claim
    textrank_scores=generate_textrank(claim,query)
    ltr_feature["textrank_mean"]=np.mean(textrank_scores)
    ltr_feature["textrank_median"]=np.median(textrank_scores)
    tfidf_scores=generate_tf_idf(claim,query)
    ltr_feature["tfidf_mean"]=np.mean(tfidf_scores)
    ltr_feature["tfidf_median"]=np.median(tfidf_scores)
    #Now Reconcile the previous Semantic Similarity scores 
    ltr_feature["distance_query_mean"]=np.mean(jsx["distance_query"])
    ltr_feature["distance_query_median"]=np.median(jsx["distance_query"])
    ltr_feature["distance_results_mean"]=np.mean(jsx["distance_results"])
    ltr_feature["distance_results_median"]=np.median(jsx["wmd_results"])
    ltr_feature["data_count"]=len(jsx["distance_query"])
    #Final feature : Semantic similarity between query and result 
    ltr_feature["distance_query_claim"]=jsx["distance_query_claim"]
    claim_queries[claim].append(ltr_feature)
    if claim_groundtruth[claim]==query:
        ltr_feature["label"]=1 #Anything non-zero is a valid label for relevance
    else:
        ltr_feature["label"]=0


# In[ ]:


import random
import math
random.seed(12313)


# In[ ]:


#Perform the Train-test Split 
all_claims = list(claim_queries.keys())
train_size = int(len(all_claims)*0.75) # 75 % of data is training set
claim_train_set=random.sample(all_claims,train_size)
claim_test_set=[]
for item in all_claims:
    if item not in claim_train_set:
        claim_test_set.append(item)


# In[ ]:


'''
Features need to be exported to LETOR format. https://arxiv.org/pdf/1306.2597.pdf  
The first entry of a row is the relevance of query w.r.t document (0 or 1 in our case). 
Then , there is a qid to group queries for each document (claim in our case ) that has the same qid, 
followed by the set of featureid: featurevalue, featureid: featurevalue.
Finally, we append Claim and corresponding candidate query; for debugging purposes later on 
Example: 
1 qid:0 1:0.30 2:0.55 .... 10:0.35 #Claim: #Query:
...
0 qid:37 1:0.123 2: 0.871 ... 10:0.55 #Claim: #Query: 

'''
features_list=["data_count","distance_query_mean","distance_query_median",
              "distance_results_mean","distance_results_median",
              "tfidf_mean","tfidf_median","textrank_mean","textrank_median",
              "distance_query_claim"]
import numpy as np
import math
claimcount=-1
yy=open("export_ltr_train.txt","w")
for claim in claim_train_set:
    rslts=claim_queries[claim]
    claimcount+=1
    for rslt in rslts:
        label=rslt["label"] 
        if label!=0: #Anything except label 0 are relevant 
            label=1
        yy.write("%d"%label)
        yy.write(" ")
        yy.write("qid:"+str(claimcount))
        for fidx in range(0,len(features_list)):
            yy.write(" ")
            yy.write(str(fidx+1)+":")
            val=rslt[features_list[fidx]]
            if np.isnan(val) or math.isinf(val):
                val=2 #Can experiment with how to handle nan Later 
            yy.write(str(val))
        #Out of loop 
        yy.write(" ")
        yy.write("#Claim: "+claim+" Query:"+rslt['query']) #For debugging purpose only 
        yy.write("\n")
yy.close()


# In[ ]:


features_list=["data_count","distance_query_mean","distance_query_median",
              "distance_results_mean","distance_results_median",
              "tfidf_mean","tfidf_median","textrank_mean","textrank_median",
              "distance_query_claim"]
import numpy as np
import math
claimcount=-1
yy=open("export_ltr_test.txt","w")
for claim in claim_test_set:
    rslts=claim_queries[claim]
    claimcount+=1
    for rslt in rslts:
        label=rslt["label"] 
        if label!=0: #Anything except label 0 are relevant 
            label=1
        yy.write("%d"%label)
        yy.write(" ")
        yy.write("qid:"+str(claimcount))
        for fidx in range(0,len(features_list)):
            yy.write(" ")
            yy.write(str(fidx+1)+":")
            val=rslt["meta"][features_list[fidx]]
            if np.isnan(val) or math.isinf(val):
                val=2 #Can experiment with how to handle nan Later 
            yy.write(str(val))
        #Out of loop 
        yy.write(" ")
        yy.write("#Claim: "+claim+" Query:"+rslt['query']) #For debugging purpose only 
        yy.write("\n")
yy.close()

