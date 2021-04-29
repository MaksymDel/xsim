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
cd experiments
mkdir data
cd data

# xnli extension
git clone https://github.com/salesforce/xnli_extension
mv xnli_extension/data/* xnli_extension

# xnli 15way
wget https://dl.fbaipublicfiles.com/XNLI/XNLI-15way.zip
unzip XNLI-15way.zip
rm XNLI-15way.zip
mv XNLI-15way xnli_15way
```

# run scripts
```
python run_sent_reps_extraction.py
python run metrics_computation.py # run this one with numpy==1.16.0 ("tmp" env)
jupyter notebook
```


Now you can run notebooks and reproduce results.