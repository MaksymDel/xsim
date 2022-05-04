# Interlingua in Multilingual Language Models Revised
## Notebook with results:

[Notebook: Interlingua in Multilingual Language Models Revised](examples/multilingual-case_study.ipynb)


#### My notes
```
srun -p gpu --gres gpu:a100-80g --mem=120G -t 192:00:00 --pty bash
srun -p gpu --gres gpu:a100-40g --mem=120G -t 192:00:00 --pty bash



module load any/python/3.8.3-conda
module load cuda/11.3.1
conda activate paper3

jupyter notebook --no-browser --port 1234
ssh -NL 1234:localhost:1234 maksym95@rocket.hpc.ut.ee
```


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

# Run scripts
```
new: 
sbatch run_task.sh

old:
CUDA_VISIBLE_DEVICES=1 python scripts/run_sent_reps_extraction.py xnli_extension
bash scripts/compute_metrics_parallel.sh
```

Now you can run analysis from the `notebooks` forlder and reproduce the plots.
