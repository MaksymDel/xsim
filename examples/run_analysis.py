import sys

import pickle
from collections import defaultdict

from datasets import load_from_disk

from util import matching_accuracy, cka_score


print("Entered script")

model_class = sys.argv[1]
task = sys.argv[2] 

if model_class not in ["mT5", "xlmr", "xglm"]:
    raise ValueError("model class is wrong")

if task not in ["acc-cent", "acc-procrustes", "acc", "cka"]:
    raise ValueError("task name is wrong")


batch_size = 100
num_lines = 10000

if model_class == "mT5":
    hf_model_ids = ['google/mt5-small',
                    'google/mt5-base',
                    'google/mt5-large',
                    'google/mt5-xl',
                    'google/mt5-xxl']

elif model_class == "xlmr":
    hf_model_ids = ['xlm-roberta-base',
                    'xlm-roberta-large',
                    'facebook/xlm-roberta-xl',
                    'facebook/xlm-roberta-xxl']
                    
elif model_class == "xglm":
    hf_model_ids = ['facebook/xglm-564M',
                    'facebook/xglm-1.7B',
                    'facebook/xglm-2.9B',
                    'facebook/xglm-4.5B',
                    'facebook/xglm-7.5B']    
else:
    raise ValueError("wrong model class specified")


langs = ['en', 'fr', 'de', 'et', 'ru']

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
        dataset[lang] = load_from_disk(f"../experiments/encoded_datasets/xnli/{hf_model_id.split('/')[-1]}/{lang}")

    
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
