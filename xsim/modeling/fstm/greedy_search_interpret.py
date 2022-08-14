from typing import Optional

import torch
from torch.nn import functional as F

from transformers.generation_logits_process import (
    LogitsProcessorList,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

"""
Usage:

from types import MethodType
from da.greedy_search_interpret import greedy_search_interpret

model_hf.greedy_search = MethodType(greedy_search_interpret, model_hf)
"""


def greedy_search_interpret(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    **model_kwargs,
):
    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    max_length = max_length if max_length is not None else self.config.max_length
    pad_token_id = (
        pad_token_id if pad_token_id is not None else self.config.pad_token_id
    )
    eos_token_id = (
        eos_token_id if eos_token_id is not None else self.config.eos_token_id
    )

    # init sequence length tensors
    (
        sequence_lengths,
        unfinished_sequences,
        cur_len,
    ) = self._init_sequence_length_for_generation(input_ids, max_length)

    all_logits = []
    all_decoder_hidden_states = []
    all_decoder_attentions = []
    all_decoder_cross_attentions = []

    while cur_len < max_length:
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        # accamulate decoder internals
        all_decoder_hidden_states.append(outputs["decoder_hidden_states"])
        all_decoder_attentions.append(outputs["decoder_attentions"])
        all_decoder_cross_attentions.append(outputs["cross_attentions"])
        # take logits
        next_token_logits = outputs.logits[:, -1, :]
        # accamulate logits

        # pre-process distribution
        scores = logits_processor(input_ids, next_token_logits)

        all_logits.append(outputs.logits[:, -1, :])

        # argmax
        next_tokens = torch.argmax(scores, dim=-1)

        # add code that transfomers next_tokens to tokens_to_add
        if eos_token_id is not None:
            assert (
                pad_token_id is not None
            ), "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (
                1 - unfinished_sequences
            )

        # add token and increase length by one
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # update sequence length
        if eos_token_id is not None:
            (
                sequence_lengths,
                unfinished_sequences,
            ) = self._update_seq_length_for_generation(
                sequence_lengths,
                unfinished_sequences,
                cur_len,
                next_tokens == eos_token_id,
            )

        # update model kwargs
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sequences.max() == 0:
            break

        # increase cur_len
        cur_len = cur_len + 1

    # reshape decoder outs
    num_tokens = len(all_decoder_hidden_states)
    num_layers = len(all_decoder_hidden_states[-1])

    # seq_len, num_layers, hidden -> num_layers, seq_len, hidden
    decoder_hidden_states = list(map(list, zip(*all_decoder_hidden_states)))
    decoder_hidden_states = [
        torch.cat(hiddens, dim=1) for hiddens in decoder_hidden_states
    ]

    # seq_len, num_layers, hidden -> num_layers, seq_len, hidden
    # decoder_attentions = list(map(list, zip(*all_decoder_attentions)))
    # decoder_attentions = [torch.cat(hiddens, dim=1) for hiddens in decoder_attentions]

    #     for t in range(num_tokens)
    #     for l in range(num_layers):
    #         decoder_hidden_states.append(all_decoder_hidden_states[:,l])

    return {
        "output_ids": input_ids,
        "encoder_hidden_states": model_inputs["encoder_outputs"]["hidden_states"],
        "encoder_attentions": model_inputs["encoder_outputs"]["attentions"],
        "decoder_hidden_states": decoder_hidden_states,
        "decoder_attentions": all_decoder_attentions,
        "decoder_cross_attentions": all_decoder_cross_attentions,
        "logits": all_logits,
    }
