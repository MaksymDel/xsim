import os
import sys

sys.path.append('.')

from transformers import AutoModel, AutoTokenizer
from normal_transformers.util.util_common import pickle_dump_to_file, read_data_xnli
from normal_transformers.util.util_encode import extract_reps_sent


if __name__ == '__main__':
    batch_size = 512  # probably can do 512

    exp_folder = "experiments"
    savedir_base = f"{exp_folder}/assets/multilingual"
    datadir = f"{exp_folder}/data"

    # loop over
    langs = ["en", "ar", "az", "bg", "cs", "da", "en_shuf"]
    #langs = ["en_shuf"]
    # langs = "ar az bg cs da de el en es et fi fr hi hu kk lt lv nl no pl ru sv sw tr ur uz vi zh".split()
    model_names_or_dirs = [
        # "xlm-roberta-large",
        "xlm-roberta-base",
        "bert-base-multilingual-cased",
        "bert-base-multilingual-uncased",
        # "distilbert-base-multilingual-cased",
        # "xlm-mlm-100-1280"
    ]

    for model_name_or_dir in model_names_or_dirs:

        print(f"Instantiating {model_name_or_dir}...")
        tokenizer_hf = AutoTokenizer.from_pretrained(model_name_or_dir)
        encoder_hf = AutoModel.from_pretrained(model_name_or_dir).cuda()  # cuda

        savedir = f"{savedir_base}/{model_name_or_dir}/xnli_encoded"
        os.makedirs(savedir, exist_ok=True)

        for lang in langs:
            data = read_data_xnli(xnli_folder=datadir, lang_code=lang)
            data_encoded = extract_reps_sent(
                data=data,
                tokenizer_hf=tokenizer_hf,
                encoder_hf=encoder_hf,
                batch_size=batch_size
            )

            # print(f"{lang} | mean sent reps size: {data_encoded['mean'].shape}")
            # print(f"{lang} | cls sent reps size: {data_encoded['cls'].shape}")

            savefile_path_base = f"{savedir}/multinli.train"

            pickle_dump_to_file(data_encoded['mean'], f"{savefile_path_base}_sentemb_mean.{lang}.pkl")
            pickle_dump_to_file(data_encoded['cls'], f"{savefile_path_base}_sentemb_cls.{lang}.pkl")
