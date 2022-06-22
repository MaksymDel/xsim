from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from numpy import dot
from numpy.linalg import norm


from procrustes import orthogonal
from ecco import analysis
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel



def load_models_tokenizers(model_class, hf_model_ids=None, HfModelClass=None):
    if HfModelClass is None:
        HfModelClass = AutoModel
    
    if hf_model_ids is None:
        hf_model_ids = get_hf_model_ids(model_class)

    if type(hf_model_ids) == list:
        hf_model_ids = add_model_names(model_class, hf_model_ids)

    models = {}
    tokenizers = {}
    for model_name, hf_model_id in hf_model_ids.items():
        print('\n loading ', hf_model_id, '\n')
        
        # load model
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

        if model_class == "mT5":
            model = HfModelClass.from_pretrained(hf_model_id).encoder        
        else:
            model = HfModelClass.from_pretrained(hf_model_id)
        
        _ = model.cuda()

        models[model_name] = model
        tokenizers[model_name] = tokenizer

    return models, tokenizers


def add_model_names(model_class, hf_model_ids):
    if model_class.startswith("norm"):
        model_names = [hf_model_id.split('/')[-2] for hf_model_id in hf_model_ids]
    else:
        model_names = [hf_model_id.split('/')[-1] for hf_model_id in hf_model_ids]
    hf_model_ids = {model_name: hf_model_id for model_name, hf_model_id in zip(model_names, hf_model_ids)}
    return hf_model_ids


def load_parallel_xnli_datasets(langs, num_lines=10000):
    datasets = {}
    for l in langs:
        datasets[l] = load_dataset('csv', 
                                delimiter='\t',
                                header=0,
                                quoting=3,
                                data_files=f"../experiments/multilingual/xnli_extension/data/multinli.train.{l}.tsv",
                                split=f'train[0:{num_lines}]')
    return datasets    



def encode_dataset(dataset, lang, tokenizer, model, batch_size, surgery_prediction_head=None):
    encoded_dataset = dataset.map(
        function=encode_batch, 
        fn_kwargs={
            'field': 'hypo', 
            'tokenizer': tokenizer, 
            'model': model,
            'detok': True,
            'lang_code': lang,
            "encode_token1": False,
            "encode_cls": False,
            "surgery_prediction_head": surgery_prediction_head
        },
            batched=True,
            batch_size=batch_size
    )
    return encoded_dataset


def encode_batch(batch,
                 field, 
                 tokenizer, 
                 model, 
                 field2=None, 
                 detok=False, 
                 lang_code=None, 
                 encode_token1=False,
                 encode_cls=True,
                 surgery_prediction_head=None):

    p1 = batch[field]
    p2 = None

    if field2 is not None:
        p2 = batch[field2]

    
    if detok:
        from sacremoses import MosesDetokenizer
        md = MosesDetokenizer(lang=lang_code)

        p1 = [md.detokenize(s.split()) for s in p1]
        if p2 is not None:
            p2 = [md.detokenize(s.split()) for s in p2]


    tok_batch = tokenizer(p1, p2,
                        return_tensors="pt",
                        padding="longest", 
                        return_attention_mask=True,
                        truncation=True,
                        max_length=100)
    
    for k, v in tok_batch.items():
        tok_batch[k] = v.to(model.device)
    
    with torch.no_grad():
        enc_batch = model(**tok_batch,
                          return_dict=True,
                          output_hidden_states=True)

    # sent_reps_all_layes_mean = []
    # sent_reps_all_layes_cls = []
    out_dict = {}
    num_layers = len(enc_batch.hidden_states)
    for layer_num in range(num_layers):

        #mean
        sent_reps_curr_layer_mean = masked_mean(
            enc_batch.hidden_states[layer_num], tok_batch.attention_mask.unsqueeze(2).bool(), 1
        )
        out_dict[f'mean_{layer_num}'] = sent_reps_curr_layer_mean.detach().cpu().numpy()

        #sent_reps_all_layes_mean.append(sent_reps_curr_layer_mean.detach().cpu().numpy())

        #cls
        if encode_cls:
            sent_reps_curr_layer_cls = enc_batch.hidden_states[layer_num][:, 0]
            out_dict[f'cls_{layer_num}'] = sent_reps_curr_layer_cls.detach().cpu().numpy()


        if encode_token1:
            sent_reps_curr_layer_token1 = enc_batch.hidden_states[layer_num][:, 1]
            out_dict[f'token1_{layer_num}'] = sent_reps_curr_layer_token1.detach().cpu().numpy()

        #sent_reps_all_layes_cls.append(sent_reps_curr_layer_cls.detach().cpu().numpy())
    
    if surgery_prediction_head is not None:
        prediction_head_dict = surgery_prediction_head(enc_batch.hidden_states[-1])
        prediction_head_dict = {k: v.detach().cpu().numpy() for k, v in prediction_head_dict.items()}
        out_dict.update(prediction_head_dict)

    return out_dict


def masked_mean(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    """
    # taken from https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
    To calculate mean along certain dimensions on masked values
    # Parameters
    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension
    # Returns
    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """

    def tiny_value_of_dtype(dtype: torch.dtype):
        if not dtype.is_floating_point:
            raise TypeError("Only supports floating point dtypes.")
        if dtype == torch.float or dtype == torch.double:
            return 1e-13
        elif dtype == torch.half:
            return 1e-4
        else:
            raise TypeError("Does not support dtype " + str(dtype))

    replaced_vector = vector.masked_fill(~mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def compute_cosine_gpu(a, b, center=False, procrustes=False):
    a, b = np.array(a), np.array(b)
    
    if procrustes:
        result = orthogonal(a, b, scale=True, translate=True)
        aq = np.dot(result.new_a, result.t)
        a = aq
        b = result.new_b

    elif center: # just center
        a -= a.mean(1, keepdims=True)
        b -= b.mean(1, keepdims=True)
    
    a, b = torch.Tensor(a), torch.Tensor(b)
    a, b = a.cuda(), b.cuda()
    
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    return res.cpu().numpy()


def matching_accuracy(a, b, center=False, procrustes=False):
    d = compute_cosine_gpu(a, b, procrustes=procrustes, center=center)
    s = accuracy_score(list(range(len(d))), d.argmax(axis=1))
    return s


def cka_score(a, b):
    a, b = np.array(a), np.array(b)
    return analysis.cka(a.T, b.T)


def pwcca_score(a, b):
    a, b = np.array(a), np.array(b)
    return analysis.pwcca(a.T, b.T)


def svcca_score(a, b):
    a, b = np.array(a), np.array(b)
    return analysis.svcca(a.T, b.T)


def cca_score(a, b):
    a, b = np.array(a), np.array(b)
    return analysis.cca(a.T, b.T)

# euuclid
def neuron_dists(a, b):
    a, b = np.array(a).T, np.array(b).T
    all_distances = []
    for neuron_1, neuron_2 in zip(a, b):
        all_distances.append(np.linalg.norm(neuron_1-neuron_2))
    return all_distances

def euclid_score(a, b):
    all_distances = neuron_dists(a, b)
    return sum(all_distances)

def normalize_euclid_score(a, b, norm_factor):
    total_d = euclid_score(a,b)
    return 1 - total_d / norm_factor
###
def cosine_score(a, b):
    a, b = np.array(a).T, np.array(b).T
    all_distances = []
    for neuron_1, neuron_2 in zip(a, b): # todo: do not recompute enlglish, use gpu
        s = abs(dot(neuron_1, neuron_2) / (norm(neuron_1) * norm(neuron_2)))
        all_distances.append(s)
    return sum(all_distances) / len(all_distances)

def torch_corr_all(a, b):
    a, b = torch.Tensor(a), torch.Tensor(b)
    a, b = a.cuda(), b.cuda()
    # center
    a -= a.mean(dim=0)
    b -= b.mean(dim=0)
    # cosine
    score = F.cosine_similarity(a, b, dim=0)
    return score.cpu().numpy()

def torch_cosines_all(a, b):
    a, b = torch.Tensor(a), torch.Tensor(b)
    a, b = a.cuda(), b.cuda()
    # cosine
    score = F.cosine_similarity(a, b, dim=0)
    return score.cpu().numpy()

def torch_corr_score(a, b):
    return abs(torch_corr_all(a, b)).mean()

def get_langs_list(model_class):
    if model_class.startswith("norm"):
        langs = ['en', 'fr', 'et', 'bg']
    elif model_class.startswith("mbert") or model_class.startswith("xlmrb"):
        langs = ['en', 'et', 'lt', 'lv', 'fr', "pl"]
    else:
        langs = ['en', 'fr', 'et', 'ru', 'de']
    return langs


def get_hf_model_ids(model_class):
    normdir="/gpfs/space/home/maksym95/third-paper/saved_models"
    if model_class == "norm_1M":
        hf_model_ids = [f'{normdir}/scale_post/checkpoint-1000000',
                        f'{normdir}/scale_pre/checkpoint-1000000',
                        f'{normdir}/scale_normformer/checkpoint-1000000']

    # if model_class == "norm_1M":
    #     hf_model_ids = [f'{normdir}/scale_normformer/checkpoint-1000000'],

    elif model_class == "norm_1M_v2":
        hf_model_ids = [f'{normdir}/scale_normformer-v2/checkpoint-1000000']

    elif model_class == "norm":
        hf_model_ids = [f'{normdir}/scale_zero/checkpoint-100000',
                        f'{normdir}/scale_post/checkpoint-100000',
                        f'{normdir}/scale_pre/checkpoint-100000',
                        f'{normdir}/scale_normformer/checkpoint-100000']

    elif model_class == "mbert":
        hf_model_ids = ['bert-base-multilingual-cased']

    elif model_class == "mT5":
        hf_model_ids = ['google/mt5-small',
                        'google/mt5-base',
                        'google/mt5-large',
                        'google/mt5-xl',
                        'google/mt5-xxl']

    elif model_class == "xlmrb":
        hf_model_ids = ['xlm-roberta-base']


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
    
    return hf_model_ids
