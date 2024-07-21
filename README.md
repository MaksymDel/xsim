# Our Research in Cross-lingual/Cross-domain Representational Similarity (xsim)

This repository contains links and codes for the following papers:

- [Cross-lingual Similarity of Multilingual Representations Revisited (AACL 2022)](#cross-lingual-similarity-of-multilingual-representations-revisited)
- [Translation Transformers Rediscover Inherent Data Domains (WMT 2021)](#translation-transformers-rediscover-inherent-data-domains)
- [Similarity of Sentence Representations in Multilingual LMs: Resolving Conflicting Literature and a Case Study of Baltic Languages (BalticHLT 2022)](#similarity-of-sentence-representations-in-multilingual-lms-resolving-conflicting-literature-and-a-case-study-of-baltic-languages)

## Cross-lingual Similarity of Multilingual Representations Revisited

Paper link: [aclanthology](https://aclanthology.org/2022.aacl-main.15/)\
Results notebook: [link](examples/aacl_2022.ipynb)\
Experiments home: [link](Cross-lingual_Similarity_of_Multilingual_Representations_Revisited.md)\
BibTeX:
```
@inproceedings{del-fishel-2022-cross,
    title = "Cross-lingual Similarity of Multilingual Representations Revisited",
    author = "Del, Maksym  and
      Fishel, Mark",
    booktitle = "Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = nov,
    year = "2022",
    address = "Online only",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.aacl-main.15",
    pages = "185--195",
    abstract = "Related works used indexes like CKA and variants of CCA to measure the similarity of cross-lingual representations in multilingual language models. In this paper, we argue that assumptions of CKA/CCA align poorly with one of the motivating goals of cross-lingual learning analysis, i.e., explaining zero-shot cross-lingual transfer. We highlight what valuable aspects of cross-lingual similarity these indexes fail to capture and provide a motivating case study demonstrating the problem empirically. Then, we introduce Average Neuron-Wise Correlation (ANC) as a straightforward alternative that is exempt from the difficulties of CKA/CCA and is good specifically in a cross-lingual context. Finally, we use ANC to construct evidence that the previously introduced {``}first align, then predict{''} pattern takes place not only in masked language models (MLMs) but also in multilingual models with causal language modeling objectives (CLMs). Moreover, we show that the pattern extends to the \textit{scaled versions} of the MLMs and CLMs (up to 85x original mBERT). Our code is publicly available at https://github.com/TartuNLP/xsim",
}
```

## Translation Transformers Rediscover Inherent Data Domains

Paper link: [aclanthology](https://aclanthology.org/2021.wmt-1.65/)\
Notebook with results: [link](examples/automatic_domains_clustering.ipynb)\
Experiments home: [external repo](https://github.com/TartuNLP/inherent-domains-wmt21)\
Production code: [external repo](https://github.com/TartuNLP/domain_clusters)\
BibTeX:
```
@inproceedings{del-etal-2021-translation,
    title = "Translation Transformers Rediscover Inherent Data Domains",
    author = "Del, Maksym  and
      Korotkova, Elizaveta  and
      Fishel, Mark",
    booktitle = "Proceedings of the Sixth Conference on Machine Translation",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wmt-1.65",
    pages = "599--613",
    abstract = "Many works proposed methods to improve the performance of Neural Machine Translation (NMT) models in a domain/multi-domain adaptation scenario. However, an understanding of how NMT baselines represent text domain information internally is still lacking. Here we analyze the sentence representations learned by NMT Transformers and show that these explicitly include the information on text domains, even after only seeing the input sentences without domains labels. Furthermore, we show that this internal information is enough to cluster sentences by their underlying domains without supervision. We show that NMT models produce clusters better aligned to the actual domains compared to pre-trained language models (LMs). Notably, when computed on document-level, NMT cluster-to-domain correspondence nears 100{\%}. We use these findings together with an approach to NMT domain adaptation using automatically extracted domains. Whereas previous work relied on external LMs for text clustering, we propose re-using the NMT model as a source of unsupervised clusters. We perform an extensive experimental study comparing two approaches across two data scenarios, three language pairs, and both sentence-level and document-level clustering, showing equal or significantly superior performance compared to LMs.",
}
```

## Similarity of Sentence Representations in Multilingual LMs: Resolving Conflicting Literature and a Case Study of Baltic Languages

Paper link: [arxiv](https://arxiv.org/abs/2109.01207)\
Results notebook: [link](examples/1.%20sim-search-BalticHLT.ipynb)\
BibTeX:
```
@article{Del2021SimilarityOS,
  title={Similarity of Sentence Representations in Multilingual LMs: Resolving Conflicting Literature and a Case Study of Baltic Languages},
  author={Maksym Del and Mark Fishel},
  journal={Balt. J. Mod. Comput.},
  year={2021},
  volume={10}
}
```
