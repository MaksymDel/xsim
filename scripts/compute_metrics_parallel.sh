src_lang=en
# model_name=xlm-roberta-base
# model_name=bert-base-multilingual-cased
model_name=bert-base-multilingual-uncased

for sent_rep in "mean"
  do
    for metric in "svcca_50"
      do
      for tgt_lang in  "ar" "az" "bg" "cs" "da" "en_shuf"
        do
          python scripts/compute_metric.py $model_name $metric $sent_rep $src_lang-$tgt_lang &
        done
      done
      wait
done
wait