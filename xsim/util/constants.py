exp_names_all = ["xnli_extension", "xnli_extension_onlyhyp", "xnli_15way"]
sent_rep_types_all = ["mean", "cls"]
similarities_all = ["cca", "pwcca", "svcca", "cka"]

model_names_or_dirs_all = [
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "distilbert-base-multilingual-cased",
    "xlm-mlm-100-1280",
    "xlm-roberta-large",
]

xnli_langs_all = "ar az bg cs da de el en es et fi fr hi hu kk lt lv nl no pl ru sv sw tr ur uz vi zh".split()
xnli_extension_langs_7 = ["en", "en_shuf", "ar", "az", "bg", "cs", "da"]
xnli_extension_langs_all = [
    "no",
    "uz",
    "pl",
    "sv",
    "ur",
    "et",
    "zh",
    "lt",
    "lv",
    "el",
    "es",
    "de",
    "nl",
    "vi",
    "th",
    "ar",
    "fr",
    "ru",
    "hu",
    "bg",
    "en",
    "az",
    "fi",
    "tr",
    "hi",
    "cs",
    "da",
    "kk",
    "sw",
]

domain_names = ["Europarl", "OpenSubtitles", "JRC-Acquis", "EMEA"]
