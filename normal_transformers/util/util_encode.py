from os import execle
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput
import pdb


def extract_reps_sent_batch(src, tokenizer_hf, encoder_hf, is_encoder_decoder=False):
    # tok
    src = tokenizer_hf.batch_encode_plus(
        src,
        padding="longest",
        return_tensors="pt",
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        max_length=100,
    )
    # src["input_ids"] = src["decoder_input_ids"]
    # src["decoder_input_ids"] = None
    # res
    # print(src)
    for k, v in src.items():
        src[k] = v.to(encoder_hf.device)

    with torch.no_grad():
        hf_model_output = encoder_hf.forward(
            **src,
            return_dict=True,
            output_hidden_states=True,
            # output_attentions=True,
        )
        # pdb.set_trace()
        if not is_encoder_decoder:
            hiddens_all_layers = hf_model_output.hidden_states
        else:

            hiddens_all_layers = (
                hf_model_output.encoder_hidden_states
                + hf_model_output.decoder_hidden_states
            )

    sent_reps_all_layes_mean = []
    sent_reps_all_layes_cls = []
    for layer_num in range(len(hiddens_all_layers)):
        hiddens_curr_layer = hiddens_all_layers[layer_num]
        sent_reps_curr_layer_cls = hiddens_curr_layer[:, 0]

        if not is_encoder_decoder:
            sent_reps_curr_layer_mean = masked_mean(
                hiddens_curr_layer, src["attention_mask"].unsqueeze(2).bool(), 1
            )
        else:
            sent_reps_curr_layer_mean = hiddens_curr_layer.mean(1)

        sent_reps_all_layes_cls.append(sent_reps_curr_layer_cls.detach().cpu().numpy())
        sent_reps_all_layes_mean.append(
            sent_reps_curr_layer_mean.detach().cpu().numpy()
        )

    return {"mean": sent_reps_all_layes_mean, "cls": sent_reps_all_layes_cls}


def extract_reps_sent(
    data, tokenizer_hf, encoder_hf, batch_size, is_encoder_decoder=False
):
    # Sent embeddings
    encoded_sent_mean = []
    encoded_sent_cls = []

    print("Extracting reps...")
    it = 0
    for i in range(0, len(data), batch_size):
        if it % 10 == 0:
            print(it)

        batch = data[i : i + batch_size]
        batch_sent_reps = extract_reps_sent_batch(
            batch, tokenizer_hf, encoder_hf, is_encoder_decoder=is_encoder_decoder
        )

        encoded_sent_mean.append(batch_sent_reps["mean"])
        encoded_sent_cls.append(batch_sent_reps["cls"])

        it += 1

    # num_layers x num_examples x num_features
    try:
        encoded_sent_mean = np.concatenate(encoded_sent_mean, 1)
        encoded_sent_cls = np.concatenate(encoded_sent_cls, 1)
    except:
        pdb.set_trace()
    return {"mean": encoded_sent_mean, "cls": encoded_sent_cls}


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
