import csv
import os
import sys

sys.path.append(".")

from normal_transformers.util.util_analysis import compute_similarity_all_layers
from normal_transformers.util.util_data import pickle_load_from_file
from normal_transformers.util import constants

# Example runs:
# python scripts/compute_metric.py mBERT_uncased cka mean en-en_shuf

if __name__ == "__main__":
    exp_name = sys.argv[1]
    model_name_or_dir = sys.argv[2]
    sim_name = sys.argv[3]
    sent_rep_type = sys.argv[4]
    lang_pair = sys.argv[5]

    assert exp_name in constants.exp_names_all
    assert sent_rep_type in constants.sent_rep_types_all
    assert model_name_or_dir in constants.model_names_or_dirs_all

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

    print(
        f"Computing {model_name_or_dir} {sim_name} {sent_rep_type} {src_lang} {tgt_lang}"
    )

    data_encoded_src = pickle_load_from_file(file_sentemb_src, verbose=True)
    data_encoded_tgt = pickle_load_from_file(file_sentemb_tgt, verbose=True)

    sim_scores = compute_similarity_all_layers(
        M1=data_encoded_src, M2=data_encoded_tgt, sim_name=sim_name
    )

    # save
    with open(savefile_path, "w", newline="") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(sim_scores)
    print(f"Saved to {savefile_path}")
