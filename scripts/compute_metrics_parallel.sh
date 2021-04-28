src_lang=en
model_name=xlm-roberta-base
# model_name=bert-base-multilingual-cased
#model_name=bert-base-multilingual-uncased
#sim_names = ["anatome_svcca_0.99", "anatome_cka", "anatome_pwcca", "google_pwcca"]

for sent_rep in "mean"
  do
    for metric in "google_cka" "anatome_svcca_0.99"
      do
      for tgt_lang in  "ar" "az" "bg" "cs" "da" "en_shuf"
        do
          python scripts/compute_metric.py $model_name $metric $sent_rep $src_lang-$tgt_lang &
        done
      done
      wait
done
wait