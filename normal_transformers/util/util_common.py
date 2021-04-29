import pickle
import random

random.seed(22)


def read_data(data_name, folder, lang_code, max_lines=None, **kwargs):
    if data_name == "xnli_extension":
        return read_data_xnli_etension(
            xnli_folder=folder, lang_code=lang_code, max_lines=max_lines, **kwargs
        )

    elif data_name == "xnli_15way":
        return read_data_xnli_15way(
            xnli_folder=folder, lang_code=lang_code, max_lines=max_lines, **kwargs
        )

    else:
        raise ValueError(f"Worng data type: {data_name}")


def read_data_xnli_15way(xnli_folder, lang_code, max_lines=None, **kwargs):
    shuffle = False
    if "_shuf" in lang_code:
        shuffle = True
        lang_code = lang_code[0:2]

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


def read_data_xnli_etension(
    xnli_folder, lang_code, max_lines=None, only_hypo=False, **kwargs
):
    from sacremoses import MosesDetokenizer

    md = MosesDetokenizer(lang=lang_code)

    shuffle = False
    if "_shuf" in lang_code:
        shuffle = True
        lang_code = lang_code[0:2]

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
            s1, s2 = l[0], l[1]
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
