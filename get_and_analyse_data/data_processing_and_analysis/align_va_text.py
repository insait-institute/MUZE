import json
from tqdm import tqdm


with open("/home/username/open_clip/get_data/text_data/va/va_id_to_fulltext_short.json","r") as f:
    new_all_id_fulltexts=json.load(f)

keys=list(new_all_id_fulltexts.keys())

# %pip install git+https://github.com/LIAAD/yake
import yake

def get_keywords(text,numOfKeywords=20):
    text=text.replace("\n","")

    language = "en"
    max_ngram_size = 3
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 2
    # numOfKeywords = 10

    kw_extractor = yake.KeywordExtractor(lan=language, 
                                        n=max_ngram_size, 
                                        dedupLim=deduplication_thresold, 
                                        dedupFunc=deduplication_algo, 
                                        windowsSize=windowSize, 
                                        top=numOfKeywords)
    keywords = kw_extractor.extract_keywords(text)

    # for kw in keywords:
    #     print(kw)

    kw=[i[0] for i in keywords]
    keywords_temp=[]
    for i in kw:
        i_temp=i.split(" ")
        keywords_temp.extend([j.lower() for j in i_temp])

    keywords_string=""
    for i in keywords_temp:
        if i not in keywords_string:
            keywords_string+=i+" "
    # print(keywords_string)

    return keywords_string


import re

long_text_keys=["summaryDescription","physicalDescription","marksAndInscriptions","objectHistory","briefDescription"]
id_to_keywords_from_fulltexts={}

# print(new_all_id_fulltexts[keys[0]])
print("total",len(keys))
count=0
for key in tqdm(keys):
    count+=1
    obj=""
    if count<len(keys)//1:
        distincts={}
        for elem in new_all_id_fulltexts[key]:
            if elem in long_text_keys:
                to_add=get_keywords(new_all_id_fulltexts[key][elem],10)
            else:
                to_add=new_all_id_fulltexts[key][elem]
            to_add=to_add.replace("\n","")
            to_add=to_add.replace(" ,",",")
            to_add=re.sub("</?\w+>","",to_add)
            to_add=re.sub("\([^()]*\)","",to_add)
            if to_add and to_add!=" ":
                # print(to_add)
                obj+=to_add+"; "
            obj=obj.replace(", ;",";").replace(" ;",";").replace(" ,",",")

        new_obj=""
        for w in obj.split(" "):
            word=w.replace(";","").replace(",","").replace(".","")
            if word.lower() in distincts:
                if ";" in w:
                    new_obj+="; "
            else:
                distincts[word.lower()]=1
                new_obj+=w+" "
        new_obj=new_obj.replace("; ;","; ").replace(" ;",";")
        id_to_keywords_from_fulltexts[key]=new_obj
        # break




with open("/home/username/open_clip/get_data/text_data/va/va_id_to_keywords_fulltext4.json","w") as f:
    json.dump(id_to_keywords_from_fulltexts,f)