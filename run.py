
from utils import *
import tqdm
import json

fpaths = glob.glob("data/nlp390_bryozoa/*")
fpaths = glob.glob("data/*.tsv")
#fpaths = glob.glob("/home/storage/corenlp_wrapper/output/*/*.json")


## Read the csv files

header = ["docid", "sentid", "wordidx", "word", "poses", "ners", "lemmas", "dep_paths", "dep_parents"]

for fpath in tqdm.tqdm(fpaths):
    if ".json" in fpath:  
        l = []

        with open(fpath, "r") as f:
            raw = f.read()
            if not raw or "java.util" in raw:
                continue ## Skip current iteration in loop
            else:
                s = json.loads(raw)
        sentences = s[0]["sentences"]

        for i, sentence in enumerate(sentences):
            l.append(sentence_to_dict(i, sentence, fpath))

        df = pd.DataFrame(l)

    else:
        df = pd.read_csv(fpath, header=None, names = header, sep ="\t")

        for k, v in df.iteritems():
            df[k] = [feature_to_list(x, k) for x in v]
                


    candidates = obtain_candidates(df)

    if candidates:
        gddid = candidates[0]["docid"]


        with open("output/{}.json".format(gddid), "w") as f:
            json.dump(candidates, f, indent=4, sort_keys=True)






