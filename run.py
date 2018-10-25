
from utils import *
import tqdm
import multiprocessing
import json
from functools import partial

#fpaths = glob.glob("/home/storage/corenlp_wrapper/output/*/*.json")[:]
#fpaths = glob.glob("/uio/kant/nhm-sfs-u2/bjorntko/TDM Project/duct_tape/data_berning/output/*.json")
#fpaths = glob.glob("/uio/kant/nhm-sfs-u2/bjorntko/TDM Project/duct_tape/data_berning/contents/*.json")
#fpaths = glob.glob("/home/storage/corenlp_parse/archive/**/*.json")
#fpaths = glob.glob("data/tmp_parse.json")
#fpaths = glob.glob("/home/storage/corenlp_parse/lidgaard/**/*.json")
#fpaths = glob.glob("/home/storage/corenlp_parse/v2/de/lidgaard/**/*.json")
#fpaths = glob.glob("/home/storage/corenlp_parse/v2/en/berning/*.json")
fpaths = glob.glob("data/berning/*.json")

n_threads = 32
source = "lidgaard"

if false:
    with multiprocessing.Pool(n_threads) as p:
        foo = partial(obtain_candidates, 
	        span0 = "TAXA",  # 0-th is for "TAXA"
	        span1 = "INTERVALNAME", 
	        source = source)
        docs = p.map(foo, fpaths)
else:
    with multiprocessing.Pool(n_threads) as p:
    foo = partial(obtain_taxa, 
        source = source)
    docs = p.map(foo, fpaths)

## Remove "None" entries
docs = list(filter(None.__ne__, docs))

## flatten list
candidates = []
abbreviations = []
for sublist in docs:
    for item in sublist["candidates"]:
        candidates.append(item)
    for item in sublist["abbreviations"]:
        abbreviations.append(item)
#candidates = [item for sublist in docs for item in sublist]
res = {"candidates": candidates,
        "abbreviations": abbreviations}

## Write output
with open("output/candidates.json", "w") as f:
    json.dump(res, f, indent=4, sort_keys=True)


