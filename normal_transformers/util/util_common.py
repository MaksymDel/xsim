import pickle


def read_data(filename_data):

    print(f"Loading from {filename_data}")
    data_raw = []
    with open(filename_data, "r") as f:
        for l in f.readlines():
            data_raw.append(l.rstrip())

    print("Loaded")
    return data_raw


def pickle_dump_to_file(obj, fn, verbose=True):
    if verbose:
        print(f"Saving to {fn}")
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
    if verbose:
        print("Saved")


def pickle_load_from_file(fn, verbose=True):
    if verbose:
        print(f"Loading from {fn}")
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
    if verbose:
        print("Loaded")
    return obj