import torch


def encode_batch(batch, field, tokenizer, model):
    tok_batch = tokenizer(batch[field],
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
        
    sent_reps_all_layes_mean = []
    sent_reps_all_layes_cls = []
    out_dict = {}
    for layer_num in range(len(enc_batch.hidden_states)):
        
        #mean
        sent_reps_curr_layer_mean = masked_mean(
            enc_batch.hidden_states[layer_num], tok_batch.attention_mask.unsqueeze(2).bool(), 1
        )
        out_dict[f'mean_{layer_num}'] = sent_reps_curr_layer_mean.detach().cpu().numpy()

        #sent_reps_all_layes_mean.append(sent_reps_curr_layer_mean.detach().cpu().numpy())

        #cls
        sent_reps_curr_layer_cls = enc_batch.hidden_states[layer_num][:, 0]
        out_dict[f'cls_{layer_num}'] = sent_reps_curr_layer_cls.detach().cpu().numpy()

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