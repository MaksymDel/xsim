import sys

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from util import encode_batch, get_hf_model_ids, get_langs_list
import torch


model_class = sys.argv[1]
device = sys.argv[2]
#model_class = "mT5"
#model_class = "xlmr"


batch_size = 100
num_lines = 10000
# num_lines = 100

hf_model_ids = get_hf_model_ids(model_class)
langs = get_langs_list(model_class)

dataset = {}
for l in langs:
    dataset[l] = load_dataset('csv', 
                               delimiter='\t',
                               header=0,
                               quoting=3,
                               data_files=f"../experiments/multilingual/xnli_extension/data/multinli.train.{l}.tsv",
                               split=f'train[0:{num_lines}]')
    
for hf_model_id in list(reversed(hf_model_ids)):
    print('\n loading ', hf_model_id, '\n')
    
    # load model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    if model_class == "mT5":
        model = AutoModel.from_pretrained(hf_model_id).encoder        
    else:
        model = AutoModel.from_pretrained(hf_model_id)
    
    _ = model.to(torch.device(f"cuda:{device}"))

    for lang in langs:
        print(f"\n encoding {lang}")

        dataset_enc = dataset[lang].map(
            function=encode_batch, 
            fn_kwargs={
                'field': 'hypo', 
                'tokenizer': tokenizer, 
                'model': model,
                'detok': True,
                'lang_code': lang,
                "encode_token1": False,
                "encode_cls": False
            },
             batched=True,
             batch_size=batch_size
        )
    
        if model_class.startswith("norm"):
            savedir = f"{model_class}_{hf_model_id.split('/')[-2]}"
        else:
            savedir = hf_model_id.split('/')[-1]

        dataset_enc.save_to_disk(f"../experiments/encoded_datasets/xnli/{savedir}/{lang}")

        # if lang == "en":
        #     lang = "en_rand"
        #     print(f"encoding {lang}")
        #     dataset_enc = dataset_enc.shuffle(seed=42)
        #     dataset_enc.save_to_disk(f"../experiments/encoded_datasets/xnli/{savedir}/{lang}")

print("Finshed")