
from utils import *
import tqdm
import multiprocessing
import json

fpaths = glob.glob("/home/storage/corenlp_wrapper/output/*/*.json")[:]
#fpaths = glob.glob("/uio/kant/nhm-sfs-u2/bjorntko/TDM Project/duct_tape/data_berning/output/*.json")
fpaths = glob.glob("/uio/kant/nhm-sfs-u2/bjorntko/TDM Project/duct_tape/data_berning/contents/*.json")
#fpaths = glob.glob("/home/storage/corenlp_parse/archive/**/*.json")
fpaths = glob.glob("data/tmp_parse.json")


n_threads = 32


with multiprocessing.Pool(n_threads) as p:
    docs = p.map(obtain_candidates, fpaths)

## Remove "None" entries
docs = list(filter(None.__ne__, docs))

## flatten list
candidates = [item for sublist in docs for item in sublist]


## Write output
with open("output/candidates.json", "w") as f:
    json.dump(candidates, f, indent=4, sort_keys=True)


