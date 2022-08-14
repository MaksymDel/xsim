This repository contains codes for the following two papers:

- [Cross-lingual Similarity of Multilingual Representations Revisited](#cross-lingual-similarity-of-multilingual-representations-revisited)
  * [Links](#links-1)
  * [Installation](#installation)
  * [Fetch data](#fetch-data)
  * [XLM-R Normformer Experiments](#xlm-r-normformer-experiments)
  * [Meta's XLM-R and XGLM Experiments](#meta-s-xlm-r-and-xglm-experiments)
  * [Final Notebook](#final-notebook)
- [Similarity of Sentence Representations in Multilingual LMs: Resolving Conflicting Literature and a Case Study of Baltic Languages](#similarity-of-sentence-representations-in-multilingual-lms--resolving-conflicting-literature-and-a-case-study-of-baltic-languages)
  * [Links](#links)


# Cross-lingual Similarity of Multilingual Representations Revisited

This repository is to reproduce results from our paper: Cross-lingual Similarity of Multilingual Representations Revisited.

## Links

Preprint link: TBA

[Notebook: Cross-lingual Similarity of Multilingual Representations Revisited](examples/emnlp22_anon.ipynb)


## Installation
```

conda create -n norm python=3.8
conda activate norm

pip install torch 
pip install transformers
pip install -U sacremoses
pip install sentencepiece
pip install protobuf
pip install scipy, pandas, matplotlib
conda install -c conda-forge notebook

```

## Fetch data

```bash
mkdir experiments
cd experiments
mkdir multilingual
cd multilingual  

# xnli extension
mkdir xnli_extension
git clone https://github.com/salesforce/xnli_extension xnli_ext_repo
mv xnli_ext_repo/data xnli_extension
rm -rf xnli_ext_repo

# xnli 15way
mkdir xnli_15way
wget https://dl.fbaipublicfiles.com/XNLI/XNLI-15way.zip
unzip XNLI-15way.zip
rm XNLI-15way.zip
mv XNLI-15way xnli_15way/data

cd ../..
```

Run the following from examples directory.

## XLM-R Normformer Experiments

```bash
python -u encode_dataset_with_models.py norm_1M

python -u run_analysis.py norm_1M cka
python -u run_analysis.py norm_1M acc 
python -u run_analysis_torch_corr.py norm_1M corr
```

## Meta's XLM-R and XGLM Experiments

```bash
cd examples

python -u encode_dataset_with_models.py xlmr
python -u encode_dataset_with_models.py xglm

python -u run_analysis_torch_corr.py xlmr corr
python -u run_analysis_torch_corr.py xglm corr
```

## Final Notebook
Now you can run the analysis from the [Notebook](examples/emnlp22_anon.ipynb).

# Similarity of Sentence Representations in Multilingual LMs: Resolving Conflicting Literature and a Case Study of Baltic Languages

## Links

Preprint link: [arxiv](https://arxiv.org/abs/2109.01207)

Notebook with results: [Notebook: Cross-lingual Similarity of Multilingual Representations Revisited](examples/1.%20sim-search-BalticHLT.ipynb)

