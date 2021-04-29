# normal-transformers
**_"Normalization is all you need."_**

...in case of Multi-task, Multi-domain, or Multilingual Transformers.

## Structure
This library consists of the following modules:
* **Modeling:** code for different Transformer variants
* **Training:** code to train Transformers
* **Encoding**: code to extract representations from trained Transformer
* **Analysis**: code to analyse extracted representations
* **Plotting**: code for unified plots to represent analysis results

Additional folders include **Tests** and **Examples**.

## Installation
```
conda create -n norm python=3.8
conda activate norm

pip install torch torchvision torchaudio
pip install transformers
pip install -U sacremoses
pip install sentencepiece
pip install protobuf
pip install scipy, pandas, matplotlib
conda install -c conda-forge notebook

# also install R for plots
pip install rpy2 --user

# install R packages
R
install.packages("rlang")
install.packages("lazteval")
install.packages("ggplot2")

```

## Fetch data
```
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
```

# run scripts
```
CUDA_VISIBLE_DEVICES=1 python scripts/run_sent_reps_extraction.py xnli_extension
bash scripts/compute_metrics_parallel.sh
```


Now you can run analysis notebooks to reproduce plots.