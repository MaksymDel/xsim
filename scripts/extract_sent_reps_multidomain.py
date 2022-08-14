import os
import sys
from numpy import exp
from transformers.models.fsmt.configuration_fsmt import FSMTConfig

sys.path.append(".")

from transformers import AutoModel, AutoTokenizer, AutoConfig
from xsim.modeling.fstm import (
    FSMTForConditionalGeneration,
    FSMTTokenizer,
    FSMTConfig,
    greedy_search_interpret,
)
from xsim.util.util_data import (
    pickle_dump_to_file,
    read_data_four_domains
)
from xsim.util.util_encode import extract_reps_sent
from xsim.util import constants
from types import MethodType


# exp_name = "xnli_extension"
# exp_name = "xnli_extension_onlyhyp"
# exp_name = "xnli_15way"
# loop over
# langs = ["en", "bg"]
# langs = ["en_shuf"]
# model_names_or_dirs = [
#     "bert-base-multilingual-uncased",
#     "bert-base-multilingual-cased",
#     "xlm-roberta-base",
#     "distilbert-base-multilingual-cased",
#     "xlm-roberta-large",
#     "xlm-mlm-100-1280",
# ]


if __name__ == "__main__":
    # exp_name = sys.argv[1]
    # assert exp_name in constants.exp_names_all
    lang_pair = "de-en"
    
    lang_src, lang_tgt = "de-en".split("-")

    exp_name = f"four-domains-{lang_src}_{lang_tgt}"
    exp_dir = f"experiments/multidomain/{exp_name}"

    # langs = ["en_shuf"] + constants.xnli_extension_langs_all
    # model_names_or_dirs = constants.model_names_or_dirs_all

    batch_size = 500

    domains = constants.domain_names
    sent_rep_types = ["mean"]

    model_name_or_dir = "bert-base-german-cased"
    # model_name_or_dir = "bert-base-cased"
    #model_name_or_dir = "xlm-roberta-base"
    # model_name_or_dir = f"{exp_dir}/nmt-model-chkp1/hf"

    print(f"Instantiating {model_name_or_dir}...")
    if "hf" not in model_name_or_dir:
        bpe_applied = False
        savedir = f"{exp_dir}/{model_name_or_dir}/sentence_embeddings"
        is_encoder_decoder = AutoConfig.from_pretrained(
            model_name_or_dir
        ).is_encoder_decoder
        tokenizer_hf = AutoTokenizer.from_pretrained(model_name_or_dir)
        encoder_hf = AutoModel.from_pretrained(model_name_or_dir).cuda()  # cuda
    else:
        mdir = model_name_or_dir.split("/")[0]
        savedir = f"{model_name_or_dir[:-3]}/sentence_embeddings"
        bpe_applied = True
        is_encoder_decoder = FSMTConfig.from_pretrained(
            model_name_or_dir
        ).is_encoder_decoder
        tokenizer_hf = FSMTTokenizer.from_pretrained(model_name_or_dir)
        encoder_hf = FSMTForConditionalGeneration.from_pretrained(
            model_name_or_dir
        ).cuda()  # cuda
        encoder_hf.greedy_search = MethodType(greedy_search_interpret, encoder_hf)

    os.makedirs(savedir, exist_ok=True)

    for domain in domains:

        data = read_data_four_domains(
            f"{lang_src}-{lang_tgt}", exp_dir, domain, bpe_applied=bpe_applied, max_lines=3000
        )

        data_encoded = extract_reps_sent(
            data=data,
            tokenizer_hf=tokenizer_hf,
            encoder_hf=encoder_hf,
            batch_size=batch_size,
            is_encoder_decoder=is_encoder_decoder,
        )

        for sent_rep_type in sent_rep_types:
            pickle_dump_to_file(
                data_encoded[sent_rep_type],
                f"{savedir}/sentemb-{sent_rep_type}-{domain}.pkl",
                verbose=False,
            )
    print("Extracted")
