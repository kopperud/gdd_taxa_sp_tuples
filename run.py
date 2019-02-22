from utils import *
import tqdm
import multiprocessing
import json
from functools import partial


#fpaths = glob.glob("/home/storage/corenlp_parse/v3/de/lidgard/**/*.json")
#fpaths = glob.glob("/home/storage/corenlp_parse/v3/en/elsevier/*.json")
#fpaths = glob.glob("data/*.json")
#fpaths = glob.glob("/home/storage/corenlp_parse/v3/en/lidgard/**/*.json")

#fpaths = glob.glob("/path/to/*.csv") ## Or json
fpaths = glob.glob("/path/to/*.json") 

if not fpaths:
    raise Exception("Please edit /path/to/*.json in file \"run.py\"")

n_threads = 32
source = "gdd"

spans = ["TAXA", "LOCATION"]

total = len(fpaths)

with multiprocessing.Pool(n_threads) as p:
    foo = partial(obtain_candidates,
        spans = spans, 
        source = source)
    docs = tqdm.tqdm(list(p.imap(foo, fpaths)), total = total)


## Remove "None" entries
docs = list(filter(None.__ne__, docs))

## flatten list
candidates = [item for sublist in docs for item in sublist]

## Write output
with open("output/candidates.json", "w") as f:
    json.dump(candidates, f, indent=4, sort_keys=True)


