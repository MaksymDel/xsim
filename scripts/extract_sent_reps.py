import os
import sys

sys.path.append(".")

from transformers import AutoModel, AutoTokenizer, AutoConfig
from xsim.util.util_data import (
    pickle_dump_to_file,
    read_data,
)
from xsim.util.util_encode import extract_reps_sent
from xsim.util import constants

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
    exp_name = sys.argv[1]
    assert exp_name in constants.exp_names_all
    exp_dir = f"experiments/multilingual/{exp_name}"

    # langs = ["en_shuf"] + constants.xnli_extension_langs_all
    # model_names_or_dirs = constants.model_names_or_dirs_all

    batch_size = 100
    model_names_or_dirs = [
        # "bert-base-multilingual-uncased",
        # "bert-base-multilingual-cased",
        #"xlm-roberta-base",
        # "distilbert-base-multilingual-cased",
        #"xlm-roberta-large",
        # "facebook/xlm-roberta-xl",
        "facebook/xlm-roberta-xxl",
        # "facebook/xglm-564M",
        # "facebook/xglm-1.7B"
        # "xlm-mlm-100-1280",
    ]
    langs = [
 #       "en",
        # "en_shuf",
        "ar",
        "az",
#        "bg",
        "cs",
        "da",
    ]  # constants.xnli_extension_langs_7
    # langs = [
    #     "en",
    #     # "en_shuf",
    #     "bg",
    #     "ru",
    #     "de",
    #     "fi"
    # ]  # constants.xnli_extension_langs_7
    #langs = constants.xnli_extension_langs_all
    sent_rep_types = ["mean"]

    for model_name_or_dir in model_names_or_dirs:

        savedir = f"{exp_dir}/{model_name_or_dir}/sentence_embeddings"

        print(f"Instantiating {model_name_or_dir}...")
        tokenizer_hf = AutoTokenizer.from_pretrained(model_name_or_dir)
        encoder_hf = AutoModel.from_pretrained(model_name_or_dir).cuda()  # cuda
        is_encoder_decoder = AutoConfig.from_pretrained(
            model_name_or_dir
        ).is_encoder_decoder

        os.makedirs(savedir, exist_ok=True)

        for lang in langs:
            data = read_data(exp_name=exp_name, exp_folder=exp_dir, lang_code=lang)
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
                    f"{savedir}/sentemb-{sent_rep_type}-{lang}.pkl",
                    verbose=False,
                )
        print("Extracted")
