import csv
import os
import sys
import itertools
from typing import AsyncIterable

sys.path.append(".")

from xsim.util.util_analysis import compute_similarity_all_layers
from xsim.util.util_data import pickle_load_from_file
from xsim.util import constants


# Example runs:
# python scripts/compute_metric.py mBERT_uncased cka mean en-en_shuf


def main(exp_name, model_name_or_dir, sim_name, sent_rep_type, lang_pair, verbose=True):
    #assert exp_name in constants.exp_names_all
    assert sent_rep_type in constants.sent_rep_types_all
    #assert model_name_or_dir in constants.model_names_or_dirs_all

    check = False
    for nm in constants.similarities_all:
        if nm in sim_name:
            check = True
    assert check

    src_lang, tgt_lang = lang_pair.split("-")

    savedir = f"experiments/{exp_name}/{model_name_or_dir}/similarity_scores"
    os.makedirs(savedir, exist_ok=True)
    savefile_path = (
        f"{savedir}/score-{sim_name}-{sent_rep_type}-{src_lang}_{tgt_lang}.csv"
    )

    file_sentemb_src = f"experiments/{exp_name}/{model_name_or_dir}/sentence_embeddings/sentemb-{sent_rep_type}-{src_lang}.pkl"
    file_sentemb_tgt = f"experiments/{exp_name}/{model_name_or_dir}/sentence_embeddings/sentemb-{sent_rep_type}-{tgt_lang}.pkl"

    if verbose:
        print(
            f"Computing {model_name_or_dir} {sim_name} {sent_rep_type} {src_lang} {tgt_lang}"
        )

    data_encoded_src = pickle_load_from_file(file_sentemb_src, verbose=verbose)
    data_encoded_tgt = pickle_load_from_file(file_sentemb_tgt, verbose=verbose)

    sim_scores = compute_similarity_all_layers(
        M1=data_encoded_src, M2=data_encoded_tgt, sim_name=sim_name
    )

    # save
    with open(savefile_path, "w", newline="") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(sim_scores)
    if verbose:
        print(f"Saved to {savefile_path}")


if __name__ == "__main__":
    # exp_name = sys.argv[1]
    # model_name_or_dir = sys.argv[2]
    # sim_name = sys.argv[3]
    # sent_rep_type = sys.argv[4]
    # lang_pair = sys.argv[5]
    # main(exp_name, model_name_or_dir, sim_name, sent_rep_type, lang_pair)

    exp_name = "multilingual/xnli_extension"
    # model_name_or_dir = "bert-base-multilingual-cased"
    #model_name_or_dir = "distilbert-base-multilingual-cased"
    #model_name_or_dir = "xlm-roberta-base"
    #model_name_or_dir = "facebook/xlm-roberta-xl"
    model_name_or_dir = "facebook/xlm-roberta-xxl"
    sim_name = "google_cka"
    sent_rep_type = "mean"

    all_pairs = list(itertools.combinations(constants.xnli_extension_langs_all, 2)) + [
        (l, l) for l in constants.xnli_extension_langs_all
    ]
    all_pairs = ["en_ar", "en_az", "en_cs", "en_da"]
    all_pairs = [p.split("_") for p in all_pairs]

    num_pairs = len(all_pairs)
    for i, pair in enumerate(all_pairs):
        print(f" {pair}: {i} / {num_pairs}")
        src_lang, tgt_lang = pair[0], pair[1]

        main(
            exp_name,
            model_name_or_dir,
            sim_name,
            sent_rep_type,
            f"{src_lang}-{tgt_lang}",
            verbose=False,
        )
