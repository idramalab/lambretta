#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import json


# In[ ]:





# In[ ]:


xx=open("candidates_test.txt")
search=[]
for x in xx:
    x=x.rstrip()
    search.append(x)


# In[ ]:


awk_source_path="../data_source"
'''
data_source file is the source file containing json of tweets, and other metadata in the format of 
"tweetid,timeoftweet,tweettext"
'''
def awk_query(query):
    querystring=query
    awkroot="awk 'BEGIN{IGNORECASE=1} "
    addedawk=""
    splitter=querystring.split(' ')
    #Construct the query with "AND" operation 
    for item in splitter:
        addedawk+='/'+item+'/'
        if cc<len(splitter):#for the last item we don't need && 
            addedawk+=' && '
    addedawk+="' "+awk_source_path+" > cmdtmp";
    cmd=awkroot+addedawk
    print("Running for query :  ",querystring," command ",cmd)
    os.system(cmd)
    #Assumping tmp has been created
    xx=open("cmdtmp")
    results=[]
    for x in xx:
        x=x.rstrip()
        x=x.split(",")
        tid=x[0]
        stamp=x[1]
        text=' '.join(x[2:])
        results.append({"tid":tid,"stamp":stamp,"text":text})
    os.system("rm cmdtmp")


# In[ ]:


awk_output_export_path="/data/ppaudeldata/VoterFraud/awk_output_test.json"

'''
About awk_query function : this can be replaced with other functionalities such as Elastic Search.
As long as the function returns an  array of dict with the metadata tweetid, timestamp, and text.
We use awk for simplicity reasons here. 
'''
for query in search:
    try:
        result=awk_query(query)
        with open(awk_output_export_path,"a+") as of:
            json.dump({"keyword":query,"data":results},of)
            of.write("\n")
    except Exception as err:
        ff=open("awkerrors_test.txt","a+")
        ff.write("quer"+querystring+" Error : "+str(err))
        ff.write("\n")

