from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import accuracy_score

from procrustes import orthogonal
from ecco import analysis


def encode_batch(batch,
                 field, 
                 tokenizer, 
                 model, 
                 field2=None, 
                 detok=False, 
                 lang_code=None, 
                 encode_token1=False,
                 encode_cls=True):

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