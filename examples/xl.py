import os
import csv

from sqlalchemy import true

from xsim.util.util_data import pickle_load_from_file

model_names_mapping = {
    "xlm-roberta-base": "xlm-roberta-base",
    "xlm-roberta-large": "xlm-roberta-large",
    "facebook/xlm-roberta-xl": "xlm-roberta-xl",
    "facebook/xlm-roberta-xxl": "xlm-roberta-xxl",
    
}
    

savedir = f"assets/multilingual"

def load_sim_scores_csv(model_name_or_dir, exp_name, sim_name, sent_rep_type, lang_pairs):
    loaddir = f"experiments/{exp_name}/{model_name_or_dir}/similarity_scores/"
    #print(loaddir)
    sim_scores_dict = {}
    for lang_pair in lang_pairs:
        lp = lang_pair.replace("-", "_")
        fn = f"score-google_{sim_name}-{sent_rep_type}-{lp}.csv"
        
        try:
            #print(f'{loaddir}/{fn}')
            with open(f'{loaddir}/{fn}') as csvfile:
                scores = [float(s) for s in list(csv.reader(csvfile))[0]]
                sim_scores_dict[lang_pair] = scores
                #print(scores)
        except:
            try:
                lp = lp.split("_")
                lp = "_".join([lp[1], lp[0]])
                fn = f"score-google_{sim_name}-{sent_rep_type}-{lp}.csv"
      
                with open(f'{loaddir}/{fn}') as csvfile:
                    scores = [float(s) for s in list(csv.reader(csvfile))[0]]
                    sim_scores_dict[lang_pair] = scores
            except:
                pass

                

    return sim_scores_dict


def load_sim_scores_into_df(model_name_or_dir, exp_name, sim_names, sent_rep_type, lang_pairs, lang_pairs_exclude=[]):
    # load metrics data
    sim_scores_all_metrics = {} 
    for sim_name in sim_names:
        scores = load_sim_scores_csv(
                                model_name_or_dir, 
                                exp_name, 
                                sim_name, 
                                sent_rep_type, 
                                lang_pairs)
        #print(scores)
        for lpair in lang_pairs_exclude:
            del scores[lpair]
            

        sim_scores_all_metrics[sim_name] = scores
        
    
    dict_of_sim_scors_dicts = sim_scores_all_metrics
    # craft padnas dataframe from it
    list_of_dfs = []
    for sim_name, sim_scores_dict in dict_of_sim_scors_dicts.items():
        tmp = {str(k).replace("-", "_"): v for k, v in sim_scores_dict.items()}
        sim_scores = tmp
        del tmp

        sim_scores_df = pd.DataFrame.from_dict(sim_scores)
        sim_scores_df["Layer"] = [i+1 for i in range(len(sim_scores_df))]
        
        sim_scores_df["Similarity"] = [str(sim_name).upper() for i in range(len(sim_scores_df))]
        sim_scores_df["Model"] = [model_names_mapping[model_name_or_dir] for i in range(len(sim_scores_df))]

        lang_pair_names = [v for v in sim_scores_df.columns.values if v != "Layer" and v != "Similarity" and v != "Model"]
        sim_scores_df = pd.melt(sim_scores_df, id_vars=["Layer", "Similarity", "Model"], value_vars=lang_pair_names,
                               var_name='lang_pair', value_name="Score")

        sim_scores_df = sim_scores_df.astype({"Layer": int})
        list_of_dfs.append(sim_scores_df)
    
    final_df = pd.concat(list_of_dfs, ignore_index=True) 
    
    return final_df

import pandas as pd

# # GLOBALS
# exp_name = "multilingual/xnli_extension"
# model_name_or_dir = "xlm-roberta-base"
# sent_rep_type = "mean"
# #sim_names = ["pwcca", "cka", "svcca_20"]
# sim_names = ["cka"]
# lang_pairs = ["en-ar", "en-az", "en-bg", "en-cs", "en-da"]
# # load df

# sim_scores_df = load_sim_scores_into_df(
#     model_name_or_dir=model_name_or_dir, 
#     exp_name=exp_name, 
#     sim_names=sim_names, 
#     sent_rep_type=sent_rep_type, 
#     lang_pairs=lang_pairs)


def load_sim_scores_into_df_models(model_name_or_dir, exp_name, sim_names, sent_rep_type, lang_pairs, lang_pairs_exclude=[]):
    # load metrics data
    x_axes_name = "Network_depth"
    sim_scores_all_metrics = {} 
    for sim_name in sim_names:
        scores = load_sim_scores_csv(
                                model_name_or_dir, 
                                exp_name, 
                                sim_name, 
                                sent_rep_type, 
                                lang_pairs)
        
        for lpair in lang_pairs_exclude:
            del scores[lpair]
            

        sim_scores_all_metrics[sim_name] = scores
        
    
    dict_of_sim_scors_dicts = sim_scores_all_metrics
    # craft padnas dataframe from it
    list_of_dfs = []
    for sim_name, sim_scores_dict in dict_of_sim_scors_dicts.items():
        tmp = {str(k).replace("-", "_"): v for k, v in sim_scores_dict.items()}
        sim_scores = tmp
        del tmp

        sim_scores_df = pd.DataFrame.from_dict(sim_scores)
        sim_scores_df[f"{x_axes_name}"] = [(i+1) / len(sim_scores_df) for i in range(len(sim_scores_df))]
        
        sim_scores_df["Similarity"] = [str(sim_name).upper() for i in range(len(sim_scores_df))]
        sim_scores_df["Model"] = [model_names_mapping[model_name_or_dir] for i in range(len(sim_scores_df))]

        lang_pair_names = [v for v in sim_scores_df.columns.values if v != f"{x_axes_name}" and v != "Similarity" and v != "Model"]
        sim_scores_df = pd.melt(sim_scores_df, id_vars=[f"{x_axes_name}", "Similarity", "Model"], value_vars=lang_pair_names,
                               var_name='lang_pair', value_name="Score")

        sim_scores_df = sim_scores_df.astype({f"{x_axes_name}": float})
        list_of_dfs.append(sim_scores_df)
    
    final_df = pd.concat(list_of_dfs, ignore_index=True) 
    
    return final_df



# # GLOBALS
# exp_name = "multilingual/xnli_extension"
# model_name_or_dir = "xlm-roberta-base"
# sent_rep_type = "mean"
# #sim_names = ["pwcca", "cka", "svcca_20"]
# sim_names = ["cka"]
# lang_pairs = ["en-ar", "en-az", "en-bg", "en-cs", "en-da"]
# # load df



# GLOBALS
exp_name = "multilingual/xnli_extension"
model_names_or_dirs = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "facebook/xlm-roberta-xl",
    "facebook/xlm-roberta-xxl",
]
sent_rep_type = "mean"
sim_names = ["cka"]
#lang_pairs = ["en-ar", "en-az", "en-bg", "en-cs", "en-da"]
lang_pairs = ["en-bg"]


all_dfs = []
for model_name_or_dir in model_names_or_dirs:
    curr_sim_scores_df = load_sim_scores_into_df_models(
        model_name_or_dir=model_name_or_dir, 
        exp_name=exp_name, 
        sim_names=sim_names, 
        sent_rep_type=sent_rep_type, 
        lang_pairs=lang_pairs,
        lang_pairs_exclude=[])
    all_dfs.append(curr_sim_scores_df)
sim_scores_df = pd.concat(all_dfs, ignore_index=True) 


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#p = sns.pairplot(sim_scores_df, hue="Model", x_vars=["Network_depth"], y_vars="Score")
p = sns.FacetGrid(col="lang_pair", hue="Model", data=sim_scores_df,  margin_titles=True, col_wrap=1, aspect=4)
p.map(sns.pointplot, "Network_depth", "Score")

fig = p.figure.savefig("assets/xl.png")