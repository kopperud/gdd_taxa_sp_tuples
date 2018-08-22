
import glob
import pandas as pd
import re
import itertools
import numpy as np
import networkx as nx
import json
import ntpath

#pattern = re.compile(',\s*(?=([^\"]*"[^\"]*\")*[^\"]*$)')


# In[801]:


const1 = "ENT10812879a"
const2 = '","'

def feature_to_list_postgres(s, key):
    if k not in {"sentid", "docid"}:
        s = s.replace("{","").replace("}","")
        s = s.replace(const2, const1)
        s = s.split(",")
        res = ["," if x == const1 else x for x in s]
    else:
        res = s
        
    return(res)    

def feature_to_list(s, key):
    if key not in {"sentid", "docid"}:
        s = s.replace("{","").replace("}","")
        s = "[" + s + "]"
        res = eval(s)

        if key == "wordidx":
            res = [int(x) for x in res]
    else:
        res = s
        
    return(res)

def sentence_to_dict(i, sentence, fpath):
    row = dict()
    dep = sorted(sentence["basicDependencies"], key = lambda x: x["dependent"])
    
    row["dep_paths"] = [x["dep"] for x in dep]
    row["dep_parents"] = [x["governor"] for x in dep]
    row["word"] = [x["word"] for x in sentence["tokens"]]
    row["ners"] = [x["ner"] for x in sentence["tokens"]]
    row["poses"] = [x["pos"] for x in sentence["tokens"]]
    row["wordidx"] = [x["index"] for x in sentence["tokens"]]
    row["lemmas"] = [x["lemma"] for x in sentence["tokens"]]

    row["docid"] = ntpath.basename(fpath)
    row["sentid"] = i+1
    
    return(row)

def index_to_tokens(x, row):
    start = x[0]
    end = start + len(x)
    return(row["word"][start:end])

def consecutive(data, stepsize=1):
    res = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    res = np.array(res)
    return(res)

def tokens_nonconsecutive_ner(word, ners, label = "TAXA"):  
    tagged = [i for i,(w, ner) in enumerate(zip(word, ners)) if ner == label]

    ner_label_indices = consecutive(tagged)
    
    tagged_offset = tagged[1:]
    tagged_offset.append(None)
    
    w = np.array(word)
    targets = [w[i] for i in ner_label_indices]
    
    entities = []
    
    for i, (idx, target) in enumerate(zip(ner_label_indices, targets)):
        res = dict()
        res["idx"] = idx.tolist()
        res[label] = w[idx].tolist()
        entities.append(res)

    return(entities)

pattern_genus = re.compile("[A-Z][.]")

class smart_dict(dict):
    def __missing__(self, key):
        return(key)

def obtain_candidates(df, span = "INTERVALNAME"):
    candidates = []
    
    for i, row in df.iterrows():
        if len({"TAXA", span}.intersection(set(row["ners"]))) > 1 and len(row["word"]) < 70:
            taxa = tokens_nonconsecutive_ner(row["word"], row["ners"], "TAXA")
            intervals = tokens_nonconsecutive_ner(row["word"], row["ners"], span)

            #Deabbreviate genus names
            is_taxa = np.array(row["ners"]) == "TAXA"
            is_abbrev = np.array([True if pattern_genus.match(x) else False for x in row["word"]])

            abbrevs = set(np.array(row["word"])[is_taxa & is_abbrev])

            if abbrevs:
                d = smart_dict()
                for abbrev in abbrevs:
                    d[abbrev] = replace_abbrev(abbrev, df, i, 8)
                
                for entity in taxa:
                    for k, v in d.items():
                        entity["TAXA"] = [v if x == k else x for x in entity["TAXA"]]
            
            ## set up dependency tree in networkx
            G = nx.Graph()

            dep = row["dep_parents"]
#            nodes = [x+1 for x in range(len(dep))]
            nodes = row["wordidx"]
            parents = [int(x) for x in dep]
#            edges = zip(parents, nodes)
            edges = zip(nodes, parents)

            G.add_edges_from(edges)

            ## All combinations of the spans (product)
            for p in itertools.product(taxa, intervals):
                sdp = dict()
                a, b = sorted([p[0], p[1]], key = lambda x: x["idx"])
                
                try:
                    sdp["idx"] = nx.shortest_path(G, a["idx"][-1]+1, b["idx"][0]+1)
                    #sdp["idx"] = nx.shortest_path(G, a["idx"][-1], b["idx"][0])
                except:
                    print(row["docid"])
                    print(row["sentid"])

                    raise Exception("Could not compute SPD")
                    
                sdp["words"] = [row["word"][i-1] for i in sdp["idx"]]

                ## Compute SPD for each
                candidate = dict()

                candidate["sdp"] = sdp

                candidate["TAXA"] = p[0]
                candidate[span] = p[1]
                candidate["sentid"] = int(i)
                candidate["sentence"] = row["word"]
                candidate["docid"] = row["docid"].replace(".json", "")

                candidates.append(candidate)

    return(candidates)

def replace_abbrev(abbrev, df, index, count):
    previous_words = df.iloc[index]["word"]
    prev_ner = df.iloc[index]["ners"]    

    genus_toks = [x for i, x in enumerate(previous_words) if x.startswith(abbrev[0]) and len(x) > 2 and prev_ner[i] == "TAXA"]
    
    if genus_toks:
        replacement = genus_toks[-1]
    else:
        if int(count) > 0:
            replacement = replace_abbrev(abbrev, df, index, count -1)
        else:
            replacement = abbrev
    return(replacement)




