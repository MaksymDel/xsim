import pickle
import random

from numpy import expand_dims

random.seed(22)


def read_data(exp_name, exp_folder, lang_code, max_lines=None):
    data_folder = f"{exp_folder}/data"

    if exp_name == "xnli_extension":
        return read_data_xnli_etension(
            xnli_folder=data_folder,
            lang_code=lang_code,
            max_lines=max_lines,
            only_hypo=False,
        )
    elif exp_name == "xnli_extension_onlyhyp":
        data_folder = data_folder.replace("xnli_extension_onlyhyp", "xnli_extension")
        return read_data_xnli_etension(
            xnli_folder=data_folder,
            lang_code=lang_code,
            max_lines=max_lines,
            only_hypo=True,
        )

    elif exp_name == "xnli_15way":
        return read_data_xnli_15way(
            xnli_folder=data_folder, lang_code=lang_code, max_lines=max_lines
        )
    else:
        raise ValueError(f"Worng exp type: {exp_name}")




def read_data_four_domains(
    lang_pair,
    exp_folder,
    domain_name,
    bpe_applied=False,
    max_lines=None):
    if lang_pair == "en-et":
        return read_data_four_domains_en_et(
                exp_folder,
                domain_name, 
                bpe_applied, 
                max_lines)
    elif lang_pair == "de-en":
        return read_data_four_domains_de_en(
                exp_folder,
                domain_name, 
                bpe_applied, 
                max_lines)
    else:
        raise NotImplementedError
    


def read_data_four_domains_en_et(
    exp_folder,
    domain_name,
    bpe_applied=False,
    max_lines=None):
    data_folder = f"{exp_folder}/data"
    if bpe_applied:
        filename = f"sp-cl-{domain_name}.en-et.docs.dev.en"
    else:
        filename = f"cl-{domain_name}.en-et.docs.dev.en"

    filepath = f"{data_folder}/{filename}"

    print(f"Loading from {filepath}")
    lines = []
    with open(filepath, "r") as f:
        for i, l in enumerate(f.readlines()):
            l = l.strip()
            lines.append(l)

            if max_lines is not None and i == max_lines:
                break
    print("Loaded")

    return lines



def read_data_four_domains_de_en(
    exp_folder,
    domain_name,
    bpe_applied=False,
    max_lines=None):
    data_folder = f"{exp_folder}/data"
    if bpe_applied:
        filename = f"sp-cl-{domain_name}.de-en.docs.dev-cl.de"
    else:
        filename = f"cl-{domain_name}.de-en.docs.dev-cl.de"

    filepath = f"{data_folder}/{filename}"

    print(f"Loading from {filepath}")
    lines = []
    with open(filepath, "r") as f:
        for i, l in enumerate(f.readlines()):
            l = l.strip()
            lines.append(l)

            if max_lines is not None and i == max_lines:
                break
    print("Loaded")

    return lines


def handle_shuffle_langcode(lang_code):
    shuffle = False
    if "_shuf" in lang_code:
        shuffle = True
        lang_code = lang_code[0:2]
    return shuffle, lang_code


def read_data_xnli_15way(xnli_folder, lang_code, max_lines=None):

    shuffle, lang_code = handle_shuffle_langcode(lang_code)

    filename = f"xnli.15way.orig.tsv"
    filepath = f"{xnli_folder}/{filename}"

    print(f"Loading from {filepath}")
    sentences = []
    lang_codes = []
    tsv_lang_index = -1  # stub
    with open(filepath, "r") as f:
        for i, l in enumerate(f.readlines()):
            l = l.strip()

            if i == 0:
                lang_codes = l.split("\t")
                tsv_lang_index = lang_codes.index(lang_code)
                continue

            l = l.split("\t")
            sent = l[tsv_lang_index]
            sentences.append(sent)

            if max_lines is not None and i == max_lines:
                break
    print("Loaded")

    if shuffle:
        print("Shuffling")
        random.shuffle(sentences)

    return sentences


def read_data_xnli_etension(xnli_folder, lang_code, max_lines=None, only_hypo=False):
    from sacremoses import MosesDetokenizer

    md = MosesDetokenizer(lang=lang_code)

    shuffle, lang_code = handle_shuffle_langcode(lang_code)

    filename = f"multinli.train.{lang_code}.tsv"
    filepath = f"{xnli_folder}/{filename}"

    print(f"Loading from {filepath}")
    pairs = []
    with open(filepath, "r") as f:
        for i, l in enumerate(f.readlines()):
            l = l.strip()
            if i == 0:
                continue

            l = l.split("\t")
            s1, s2 = l[0].split(), l[1].split()  # md want a list of tokens
            s1, s2 = md.detokenize(s1), md.detokenize(s2)

            pairs.append((s1, s2))

            if max_lines is not None and i == max_lines:
                break
    print("Loaded")

    if shuffle:
        print("Shuffling")
        random.shuffle(pairs)

    if only_hypo:
        print("Selecting hypothesis only")
        pairs = [p[1] for p in pairs]

    return pairs


def read_data_basic(filename_data, verbose=True):
    if verbose:
        print(f"Loading from {filename_data}")

    data_raw = []
    with open(filename_data, "r") as f:
        for l in f.readlines():
            data_raw.append(l.rstrip())

    if verbose:
        print("Loaded")
    return data_raw


def pickle_dump_to_file(obj, fn, verbose=True):
    if verbose:
        print(f"Saving to {fn}")
    with open(fn, "wb") as f:
        pickle.dump(obj, f)
    if verbose:
        print("Saved")


def pickle_load_from_file(fn, verbose=True):
    if verbose:
        print(f"Loading from {fn}")
    with open(fn, "rb") as f:
        obj = pickle.load(f)
    if verbose:
        print("Loaded")
    return obj
