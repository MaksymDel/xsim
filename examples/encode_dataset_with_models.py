import sys

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from util import encode_batch


model_class = sys.argv[1]
#model_class = "mT5"
#model_class = "xlmr"


if model_class not in ["mT5", "xlmr", "xglm"]:
    raise ValueError("model class is wrong")


batch_size = 100
num_lines = 10000
# num_lines = 100


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

# hf_model_ids = ['xlm-roberta-base',
#                 'xlm-roberta-large']

# langs = ['en', 'fr', 'de']

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
    
    _ = model.cuda()

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
        
        dataset_enc.save_to_disk(f"../experiments/encoded_datasets/xnli/{hf_model_id.split('/')[-1]}/{lang}")

print("Finshed")