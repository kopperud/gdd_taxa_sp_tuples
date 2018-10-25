
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

header = ["docid", "sentid", "wordidx", "word", "poses", "ners", "lemmas", "dep_paths", "dep_parents"]

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

def tokens_nonconsecutive_ner(word, ners, label):  
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

def obtain_candidates(fpath, span0 = "TAXA", span1 = "INTERVALNAME", source = None):
    if ".json" in fpath:  
        l = []

        with open(fpath, "r") as f:
            raw = f.read()
            if not raw or "java.util" in raw:
                return(None)
            else:
                s = json.loads(raw)
        if not s:
            return(None)
        sentences = s[0]["sentences"]
	

        for i, sentence in enumerate(sentences):
            l.append(sentence_to_dict(i, sentence, fpath))

        df = pd.DataFrame(l)

    else:
        df = pd.read_csv(fpath, header=None, names = header, sep ="\t")

        for k, v in df.iteritems():
            df[k] = [feature_to_list(x, k) for x in v]

    res = parse_candidate(df, span0 = span0, span1 = span1)
    candidates = res["candidates"]
    abbreviations = res["abbreviations"]

    ## Establish document ID. Specific for each source, not for GDD
    if candidates:
        if ".json" in fpath:
            # if berning
            if source == "berning":
                docid = ntpath.basename(fpath.replace(".json",""))
            # if archive
            elif source == "archive":
                docid = fpath.split("/")[-2]
            # if lidgaard
            elif source == "lidgaard" or source == "btk_lg":
                docid = '/'.join(fpath.split("/")[-2:])
            # Assign document ids

            for item in candidates:
                item["docid"] = docid

        return(res)
    else:
        return(None)



def parse_candidate(df, span0 = "TAXA", span1 = "INTERVALNAME"):
    candidates = []
    abbreviations = []
    
    for i, row in df.iterrows():
        if len({span0, span1}.intersection(set(row["ners"]))) > 1 and len(row["word"]) < 70:
            taxa = tokens_nonconsecutive_ner(row["word"], row["ners"], span0)
            intervals = tokens_nonconsecutive_ner(row["word"], row["ners"], span1)

            #Deabbreviate genus names
            if span0 == "TAXA":
                is_taxa = np.array(row["ners"]) == span0
                is_abbrev = np.array([True if pattern_genus.match(x) else False for x in row["word"]])

                abbrevs = set(np.array(row["word"])[is_taxa & is_abbrev])

                if abbrevs:
                    d = smart_dict()
                    for abbrev in abbrevs:
                        d[abbrev] = replace_abbrev(abbrev, df, i, 8)
                    
                    abbreviations.append(d) 

                    for entity in taxa:
                        for k, v in d.items():
                            entity[span0] = [v if x == k else x for x in entity[span0]]

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

                candidate[span0] = p[0]
                candidate[span1] = p[1]
                candidate["sentid"] = int(i)
                candidate["sentence"] = row["word"]
                candidate["docid"] = row["docid"].replace(".json", "")

                candidates.append(candidate)
    res = {"candidates": candidates,
           "abbreviations": abbreviations}
    return(res)

def replace_abbrev(abbrev, df, index, count):
    previous_words = df.iloc[index]["word"]
    prev_ner = df.iloc[index]["ners"]    

    genus_toks = [x for i, x in enumerate(previous_words) if x.startswith(abbrev[0]) and len(x) > 2 and prev_ner[i] == "TAXA"]

    if genus_toks:
        replacement = genus_toks[-1]
    else:
        if int(count) > 0:
            replacement = replace_abbrev(abbrev, df, index-1, count -1)
        else:
            replacement = abbrev
    return(replacement)


def obtain_taxa(fpath, source = None):
    if ".json" in fpath:  
        l = []

        with open(fpath, "r") as f:
            raw = f.read()
            if not raw or "java.util" in raw:
                return(None)
            else:
                s = json.loads(raw)
        if not s:
            return(None)
        sentences = s[0]["sentences"]
	

        for i, sentence in enumerate(sentences):
            l.append(sentence_to_dict(i, sentence, fpath))

        df = pd.DataFrame(l)

    else:
        df = pd.read_csv(fpath, header=None, names = header, sep ="\t")

        for k, v in df.iteritems():
            df[k] = [feature_to_list(x, k) for x in v]

    candidates = parse_taxa(df)


def parse_taxa(df):
    candidates = []
    abbreviations = []
    span0 = "TAXA"
    
    for i, row in df.iterrows():
        if len({span0}.intersection(set(row["ners"]))) > 1 and len(row["word"]) < 70:
            taxa = tokens_nonconsecutive_ner(row["word"], row["ners"], span0)

            #Deabbreviate genus names
            if span0 == "TAXA":
                is_taxa = np.array(row["ners"]) == span0
                is_abbrev = np.array([True if pattern_genus.match(x) else False for x in row["word"]])

                abbrevs = set(np.array(row["word"])[is_taxa & is_abbrev])

                if abbrevs:
                    abbrev_info = smart_dict()
                    for abbrev in abbrevs:
                        abbrev_info[abbrev] = replace_abbrev(abbrev, df, i, 15)

                    d = abbrev_info.copy()
                    abbreviations.append(abbrev_info) 

                    for k, v in d.items():
                        d[k] = v[1]

                    for entity in taxa:
                        for k, v in d.items():
                            entity[span0] = [v if x == k else x for x in entity[span0]]


                candidate = dict()
                candidate[span0] = taxa
                candidate["sentid"] = int(i)
                candidate["sentence"] = row["word"]
                candidate["docid"] = row["docid"].replace(".json", "")

                if abbrevs:
                    candidate["abbreviations"] = abbrev_info

                candidates.append(candidate)
    return(candidates)


