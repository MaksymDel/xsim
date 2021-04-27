import numpy as np
# from scipy.spatial import distance

from .cka import feature_space_linear_cka


import sys

sys.path.insert(0, "svcca")

import cca_core

"""
For all functions M1 & M2 shapes are (num_layers, num_examples, num_features).

Output shape is (num_layers, 1)
"""

def get_svcca_similarity(acts1, acts2, K=20, verbose=False, epsilon=None):
    ''' Compute svcca similarity, adapted from tutorial on
    https://github.com/google/svcca/tree/master/tutorials.
    '''
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:K] * np.eye(K), V1[:K])
    svacts2 = np.dot(s2[:K] * np.eye(K), V2[:K])
    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=epsilon, verbose=verbose,  compute_coefs=False)
    return svcca_results


def get_similarity(acts1, acts2, verbose=False, epsilon=1e-10, method='mean'):
    import pwcca

    if method == 'all':
        similarity_dict = cca_core.get_cca_similarity(acts1, acts2, epsilon=epsilon, compute_coefs=False)
        return np.mean(similarity_dict['cca_coef1'])
    if method == 'mean':
        similarity_dict = cca_core.get_cca_similarity(acts1, acts2, epsilon=epsilon, compute_coefs=False)
        return similarity_dict['mean'][0]  # contains two times the same value.
    elif 'svcca' in method:
        k = int(method.split("_")[1])
        similarity_dict = get_svcca_similarity(acts1, acts2, K=k, verbose=verbose, epsilon=epsilon)
        return similarity_dict['mean'][0]
    elif method == 'pwcca':
        pwcca_mean, w, __ = pwcca.compute_pwcca(acts1, acts2, epsilon=epsilon)
        return pwcca_mean
    elif method == 'cka':
        return feature_space_linear_cka(acts1.transpose(1, 0), acts2.transpose(1, 0))
    else:
        raise NotImplementedError(method)


def compute_similarity_all_layers_google(M1, M2, sim_name, skip_embedding_layer=True):
    if not skip_embedding_layer:
        raise NotImplementedError("PWCCA and others fail to converge for uncontextual layer for some reason.")

    if sim_name == "cca":
        sim_name = "mean"

    if M2.shape[1] > M1.shape[1]:
        print("Warning: shapes mismatch")
        M2 = M2[:, 0:M1.shape[1], :]
    num_layers, num_examples, num_features = M1.shape
    sims = []

    for i in range(num_layers - 1):
        l = i + 1  # do not compute for uncontextual embeddings
#    for l in range(num_layers):
        x = M1[l]
        y = M2[l]
        score = get_similarity(x.transpose(1, 0), y.transpose(1, 0), method=sim_name, verbose=False)
        sims.append(score)
        print(f"layer: {l}: {score}")
    return sims



def sim_linear_cka(M1, M2):
    if M2.shape[1] > M1.shape[1]:
        print("Warning: shapes mismatch")
        M2 = M2[:, 0:M1.shape[1], :]
    num_layers, num_examples, num_features = M1.shape
    sims = []
    for l in range(num_layers):
        sims.append(feature_space_linear_cka(M1[l], M2[l]))
    return sims


def sim_svcca_conseq_layers(M):
    num_layers, num_examples, num_features = M.shape
    sims = []
    for ind1 in range(num_layers - 1):
        ind2 = ind1 + 1
        reps1, reps2 = M[ind1], M[ind2]
        sims.append(svcca(reps1.transpose(1, 0), reps2.transpose(1, 0)))
        # sims.append(feature_space_linear_cka(reps1.transpose(1,0), reps2.transpose(1,0)))
    return sims


def sim_cka_conseq_layers(M):
    num_layers, num_examples, num_features = M.shape
    sims = []
    for ind1 in range(num_layers - 1):
        ind2 = ind1 + 1
        reps1, reps2 = M[ind1], M[ind2]
        sims.append(feature_space_linear_cka(reps1, reps2))
        # sims.append(feature_space_linear_cka(reps1.transpose(1,0), reps2.transpose(1,0)))
    return sims


# def +
# (M):
#     num_layers, num_examples, num_features = M.shape
#     sims = []
#     for ind1 in range(num_layers - 1):
#         ind2 = ind1 + 1
#         reps1, reps2 = M[ind1], M[ind2]
#         # sims.append(svcca(reps1.transpose(1,0), reps2.transpose(1,0)))
#         sims.append(feature_space_linear_cka(reps1, reps2))
#     return sims

#
# def neuron_interlingua_scores(reps1, reps2):
#     # reps shape is (num_examples x num_features)
#     num_layers, num_examples, num_features = reps1.shape
#     res = []
#     for l in range(num_layers):
#         per_feat_err = []
#         for fi in range(num_features):
#             # err = reps1[l, :, fi] - reps2[l, :, fi]
#             # err = np.mean(np.abs(err))
#             err = distance.correlation(reps1[l, :, fi], reps2[l, :, fi])
#             per_feat_err.append(err)
#         res.append(per_feat_err)
#     return 1 - np.array(res)

#
# def sim_cosine(M1, M2):
#     num_layers, num_examples, num_features = M1.shape
#     sims = []
#     for l in range(num_layers):
#         tmp = []
#         for e in range(num_examples):
#             sim = 1 - distance.cosine(M1[l, e, :], M2[l, e, :])
#             tmp.append(sim)
#         sim = sum(tmp) / len(tmp)
#         sims.append(sim)
#     return sims
#
#
# def sim_corr(M1, M2):
#     num_layers, num_examples, num_features = M1.shape
#     sims = []
#     for l in range(num_layers):
#         tmp = []
#         for e in range(num_examples):
#             sim = 1 - distance.correlation(M1[l, e, :], M2[l, e, :])
#             tmp.append(sim)
#         sim = sum(tmp) / len(tmp)
#         sims.append(sim)
#     return sims
#
#
# def sim_corr_avgfirst(M1, M2):
#     num_layers, num_examples, num_features = M1.shape
#     sims = []
#     M1 = np.mean(M1, 1)
#     M2 = np.mean(M2, 1)
#     for l in range(num_layers):
#         sim = distance.euclidean(M1[l, :], M2[l, :])
#         sims.append(sim)
#     return sims
#
#
# def sim_euclid(M1, M2):
#     num_layers, num_examples, num_features = M1.shape
#     sims = []
#     for l in range(num_layers):
#         tmp = []
#         for e in range(num_examples):
#             sim = 1 - distance.euclidean(M1[l, e, :], M2[l, e, :])
#             tmp.append(sim)
#         sim = sum(tmp) / len(tmp)
#         sims.append(sim)
#     return sims