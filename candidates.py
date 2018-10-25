import json, glob, tqdm

with open("output/candidates.json", "r") as f:
    candidates = json.loads(f.read())

#print("\n\n")
print("=================================================")
print("\n")
print("The object 'candidates' is now loaded. n = {}".format(len(candidates)))
