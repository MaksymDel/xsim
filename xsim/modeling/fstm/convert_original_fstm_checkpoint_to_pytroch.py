import os
from os.path import basename, dirname
from shutil import copyfile

import json
from collections import OrderedDict

import torch
import sentencepiece as spm

from fairseq.models.transformer import TransformerModel
from fairseq.tasks.translation import TranslationTask
from fairseq.hub_utils import GeneratorHubInterface

from transformers import WEIGHTS_NAME, logging
from transformers.models.fsmt.configuration_fsmt import FSMTConfig
from transformers.models.fsmt.modeling_fsmt import FSMTForConditionalGeneration


def convert_fsmt_checkpoint_to_pytorch(
    fsmt_checkpoint_path, pytorch_dump_folder_path, data_path, spm_model_path=None
):
    # assumes join dicitionary

    json_indent = 2

    # prep
    assert os.path.exists(fsmt_checkpoint_path)
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    print(f"Writing results to {pytorch_dump_folder_path}")

    chkpt = torch.load(fsmt_checkpoint_path)
    chkpt["cfg"]["task"].data = data_path
    chkpt["cfg"]["model"].data = data_path
    torch.save(chkpt, fsmt_checkpoint_path)

    task_args, model_args = chkpt["cfg"]["task"], chkpt["cfg"]["model"]

    task = TranslationTask.setup_task(task_args)
    model = task.build_model(model_args)

    # model config
    fsmt_model_config_file = os.path.join(pytorch_dump_folder_path, "config.json")

    model_conf = {
        "architectures": ["FSMTForConditionalGeneration"],
        "model_type": "fsmt",
        "activation_dropout": model_args.activation_dropout,
        "activation_function": "relu",
        "attention_dropout": model_args.attention_dropout,
        "d_model": model_args.decoder_embed_dim,
        "dropout": model_args.dropout,
        "init_std": 0.02,
        "max_position_embeddings": model_args.max_source_positions,
        "num_hidden_layers": model_args.encoder_layers,
        "src_vocab_size": len(task.source_dictionary),
        "tgt_vocab_size": len(task.target_dictionary),
        "langs": [task_args.source_lang, task_args.target_lang],
        "encoder_attention_heads": model_args.encoder_attention_heads,
        "encoder_ffn_dim": model_args.encoder_ffn_embed_dim,
        "encoder_layerdrop": model_args.encoder_layerdrop,
        "encoder_layers": model_args.encoder_layers,
        "decoder_attention_heads": model_args.decoder_attention_heads,
        "decoder_ffn_dim": model_args.decoder_ffn_embed_dim,
        "decoder_layerdrop": model_args.decoder_layerdrop,
        "decoder_layers": model_args.decoder_layers,
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "is_encoder_decoder": True,
        "scale_embedding": not model_args.no_scale_embedding,
        "tie_word_embeddings": model_args.share_all_embeddings,
        "share_decoder_input_output_embed": model_args.share_decoder_input_output_embed,
    }

    # good hparam defaults to start with
    model_conf["num_beams"] = 5
    model_conf["early_stopping"] = False
    model_conf["length_penalty"] = 1.0

    print(f"Generating {fsmt_model_config_file}")
    with open(fsmt_model_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(model_conf, ensure_ascii=False, indent=json_indent))

    # model
    hub_gen = TransformerModel.from_pretrained(
        dirname(fsmt_checkpoint_path),
        checkpoint_file=basename(fsmt_checkpoint_path),
        data_name_or_path=task_args.data,
    )

    model_state_dict = hub_gen.models[0].state_dict()

    # rename keys to start with 'model.'
    model_state_dict = OrderedDict(
        ("model." + k, v) for k, v in model_state_dict.items()
    )

    # remove unneeded keys
    ignore_keys = [
        "model.model",
        "model.encoder.version",
        "model.decoder.version",
        # "model.encoder_embed_tokens.weight",
        # "model.decoder_embed_tokens.weight",
        "model.encoder.embed_positions._float_tensor",
        "model.decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        model_state_dict.pop(k, None)

    # print(model_state_dict.keys())

    config = FSMTConfig.from_pretrained(pytorch_dump_folder_path)
    model_new = FSMTForConditionalGeneration(config)

    # check that it loads ok
    model_new.load_state_dict(model_state_dict, strict=False)

    # save
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    print(f"Generating {pytorch_weights_dump_path}")
    torch.save(model_state_dict, pytorch_weights_dump_path)

    pytorch_vocab_dump_path = os.path.join(pytorch_dump_folder_path, "vocab.txt")
    print(f"Generating {pytorch_vocab_dump_path}")
    assert hub_gen.src_dict.indices == hub_gen.tgt_dict.indices
    with open(pytorch_vocab_dump_path, "w") as f:
        for item in hub_gen.src_dict.indices:
            f.write("%s\n" % item)

    if spm_model_path is not None:
        copyfile(spm_model_path, f"{pytorch_dump_folder_path}/spm_model.spm")

    print("Conversion is done!")
