# depricated
import os
import sys

sys.path.append('..')

from xsim.util.util_analysis import compute_similarity_all_layers
from xsim.util.util_common import pickle_dump_to_file, pickle_load_from_file


def compute_sim_scores(data_encoded, sim_name, langs):
    sim_scores = {}
    src_code = "en"
    src_enc = data_encoded[src_code]
    for tgt_code in langs:
        if tgt_code != src_code:
            tgt_enc = data_encoded[tgt_code]

            print(f"{sim_name}: {src_code}-{tgt_code}")
            score = compute_similarity_all_layers(M1=src_enc, M2=tgt_enc, sim_name=sim_name)

            sim_scores[f"{src_code}-{tgt_code}"] = score
    return sim_scores


if __name__ == '__main__':

    exp_folder = "experiments"
    savedir_base = f"{exp_folder}/assets/multilingual"
    datadir = f"{exp_folder}/data"


    # loop over
    langs = ["en", "ar", "az", "bg", "cs", "da", "en_shuf"]
    # langs = "ar az bg cs da de el en es et fi fr hi hu kk lt lv nl no pl ru sv sw tr ur uz vi zh".split()
    model_names_or_dirs = [
        # "xlm-roberta-large",
        "bert-base-multilingual-uncased",
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
        # "distilbert-base-multilingual-cased",
        # "xlm-mlm-100-1280"
    ]
    sim_names = ["cca", "pwcca", "svcca", "cka"]
    sent_rep_types = ["mean", "cls"]

    for model_name_or_dir in model_names_or_dirs:
        print(f"{model_name_or_dir}...")

        # read data
        data_encoded = {}
        data_encoded_mean = {}
        data_encoded_cls = {}
        for lang in langs:
            loadfile_path_base = f"{savedir_base}/{model_name_or_dir}/xnli_encoded/multinli.train"

            data_encoded_mean[lang] = pickle_load_from_file(f"{loadfile_path_base}_sentemb_mean.{lang}.pkl",
                                                            verbose=False)
            data_encoded_cls[lang] = pickle_load_from_file(f"{loadfile_path_base}_sentemb_cls.{lang}.pkl",
                                                           verbose=False)
        data_encoded_all = {"mean": data_encoded_mean, "cls": data_encoded_cls}

        # compute metrics
        for sent_rep_type in sent_rep_types:
            data_encoded = data_encoded_all[sent_rep_type]
            print(f"* {sent_rep_type}...")
            for sim_name in sim_names:
                print(f"** {sim_name}...")

                sim_scores = compute_sim_scores(data_encoded, sim_name, langs)

                savedir = f"{savedir_base}/{model_name_or_dir}/sim_scores"
                os.makedirs(savedir, exist_ok=True)

                savefile_path = f"{savedir}/xnli6_{sim_name}_{sent_rep_type}.pkl"
                pickle_dump_to_file(sim_scores, savefile_path)
