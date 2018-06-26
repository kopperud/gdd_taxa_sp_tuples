
from utils import *
import tqdm

fpaths = glob.glob("data/nlp390_bryozoa/*")
fpaths = glob.glob("data/*.tsv")



## Read the csv files

header = ["docid", "sentid", "wordidx", "word", "poses", "ners", "lemmas", "dep_paths", "dep_parents"]

for fpath in tqdm.tqdm(fpaths):

    df = pd.read_csv(fpath, header=None, names = header, sep ="\t")

    for k, v in df.iteritems():
        df[k] = [feature_to_list(x, k) for x in v]
    candidates = obtain_candidates(df)

    if candidates:
        gddid = candidates[0]["gddid"]


        with open("output/{}.json".format(gddid), "w") as f:
            json.dump(candidates, f, indent=4, sort_keys=True)






