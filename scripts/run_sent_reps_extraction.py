import os
import sys

sys.path.append(".")

from transformers import AutoModel, AutoTokenizer
from normal_transformers.util.util_data import (
    pickle_dump_to_file,
    read_data,
)
from normal_transformers.util.util_encode import extract_reps_sent
from normal_transformers.util import constants

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
    exp_dir = f"experiments/{exp_name}"

    langs = ["en_shuf"] + constants.xnli_extension_langs_all
    model_names_or_dirs = constants.model_names_or_dirs_all
    sent_rep_types = ["mean"]

    for model_name_or_dir in model_names_or_dirs:
        # hack
        if model_name_or_dir == "xlm-roberta-large":
            batch_size = 500
        else:
            batch_size = 1000

        savedir = f"{exp_dir}/{model_name_or_dir}/sentence_embeddings"

        print(f"Instantiating {model_name_or_dir}...")
        tokenizer_hf = AutoTokenizer.from_pretrained(model_name_or_dir)
        encoder_hf = AutoModel.from_pretrained(model_name_or_dir).cuda()  # cuda

        os.makedirs(savedir, exist_ok=True)

        for lang in langs:
            data = read_data(exp_name=exp_name, exp_folder=exp_dir, lang_code=lang)
            data_encoded = extract_reps_sent(
                data=data,
                tokenizer_hf=tokenizer_hf,
                encoder_hf=encoder_hf,
                batch_size=batch_size,
            )

            for sent_rep_type in sent_rep_types:
                pickle_dump_to_file(
                    data_encoded[sent_rep_type],
                    f"{savedir}/sentemb-{sent_rep_type}-{lang}.pkl",
                    verbose=False,
                )
        print("Extracted")
