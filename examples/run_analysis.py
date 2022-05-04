import sys

import pickle
from collections import defaultdict

from datasets import load_from_disk

from util import matching_accuracy, cka_score, get_hf_model_ids, get_langs_list


print("Entered script")

model_class = sys.argv[1]
task = sys.argv[2] 

if task not in ["acc-cent", "acc-procrustes", "acc", "cka"]:
    raise ValueError("task name is wrong")


batch_size = 100
num_lines = 10000

hf_model_ids = get_hf_model_ids(model_class)
langs = get_langs_list(model_class)

print(task)
print(hf_model_ids)
print(langs)
center = task == "acc-cent"
procrustes = task == "acc-procrustes"

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
    print(f"{num_layers} layers")

    for lang in langs:
        if lang == "en":
          continue
        
        print(f"\n pair: en-{lang}")

        tgt = dataset[lang]
        
        for l in range(num_layers):
            if task.startswith("acc"):
                s = matching_accuracy(src[f'mean_{l}'], tgt[f'mean_{l}'], procrustes=procrustes, center=center)
            elif task == "cka":
                s = cka_score(src[f'mean_{l}'], tgt[f'mean_{l}'])
            else:
                raise ValueError
            
            scores_dict[hf_model_id][f"en-{lang}"].append(s)
            # mean_cosmatrix[hf_model_id][f"en-{lang}"].append(d)
            print(f"l{l}: {s}", end = ' ')
            
print('\n\nFinished \n')


# save
scores_dfs = dict(scores_dict)
scores_dfs = {k: dict(v) for k, v in scores_dfs.items()}

savename = f"../experiments/encoded_datasets/xnli/{model_class}-{task}-all_models.pkl"
pickle.dump(scores_dfs, open(savename, 'wb'))
print(f"saved scores at {savename}")
