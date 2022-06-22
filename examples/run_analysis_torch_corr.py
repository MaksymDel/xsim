import sys

import pickle
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from util import get_hf_model_ids, get_langs_list


print("Entered script")

model_class = sys.argv[1]
device = sys.argv[2]

center = True
task="corr"

# if task not in ["acc-cent", "acc-procrustes", "acc", "cka"]:
#     raise ValueError("task name is wrong")


batch_size = 100
num_lines = 10000

hf_model_ids = get_hf_model_ids(model_class)
#langs = ["en_rand"] + get_langs_list(model_class)
langs = get_langs_list(model_class)

print(hf_model_ids)
print(langs)

scores_dict = defaultdict(lambda: defaultdict(list))
# mean_cosmatrix = defaultdict(lambda: defaultdict(list))

for hf_model_id in list(reversed(hf_model_ids)):
    print(f"\n\n{hf_model_id}")
    
    # load datasets for needed model
    dataset = {}
    for lang in langs:
        if model_class.startswith("norm"):
            savedir = f"{model_class}_{hf_model_id.split('/')[-2]}"
        else:
            savedir = hf_model_id.split('/')[-1]
        dataset[lang] = load_from_disk(f"../experiments/encoded_datasets/xnli/{savedir}/{lang}")

    
    src = dataset['en']
    num_layers = sum([n.startswith("mean") for n in src.column_names])
    print(f"{num_layers} layers", flush=True)

 
    for lang in langs:
        if lang == "en":
          continue
        
        print(f"\n pair: en-{lang}", flush=True)

        tgt = dataset[lang]
        
        src_layers = []
        tgt_layers = []
        
        print("loading dataset...", flush=True)
        for l in range(num_layers):
            src_layers.append(src[f'mean_{l}'])
            tgt_layers.append(tgt[f'mean_{l}'])
        
        
        print("computing...", flush=True)
        src_torch = torch.Tensor(src_layers).to(torch.device(f"cuda:{device}"))
        tgt_torch = torch.Tensor(tgt_layers).to(torch.device(f"cuda:{device}"))

        if center:
            src_torch -= src_torch.mean(dim=1, keepdim=True)
            tgt_torch -= tgt_torch.mean(dim=1, keepdim=True)

        scores = F.cosine_similarity(src_torch, tgt_torch, dim=1).mean(dim=1)
        scores = scores.cpu().numpy()

        scores_dict[hf_model_id][f"en-{lang}"] = scores
        # mean_cosmatrix[hf_model_id][f"en-{lang}"].append(d)
        print(f"scores: {scores}", end = ' ', flush=True)
            
print('\n\nFinished \n')


# save
scores_dfs = dict(scores_dict)
scores_dfs = {k: dict(v) for k, v in scores_dfs.items()}

savename = f"../experiments/encoded_datasets/xnli/{model_class}-{task}-all_models.pkl"
pickle.dump(scores_dfs, open(savename, 'wb'))
print(f"saved scores at {savename}")
