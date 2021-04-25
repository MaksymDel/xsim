import numpy as np
from scipy.spatial import distance

from .cka import feature_space_linear_cka
from .pwcca import robust_pwcaa
from .svcca import svcca

"""
For all functions M1 & M2 shapes are (num_layers, num_examples, num_features).

Output shape is (num_layers, 1)
"""


def sim_cosine(M1, M2):
    num_layers, num_examples, num_features = M1.shape
    sims = []
    for l in range(num_layers):
        tmp = []
        for e in range(num_examples):
            sim = 1 - distance.cosine(M1[l, e, :], M2[l, e, :])
            tmp.append(sim)
        sim = sum(tmp) / len(tmp)
        sims.append(sim)
    return sims


def sim_corr(M1, M2):
    num_layers, num_examples, num_features = M1.shape
    sims = []
    for l in range(num_layers):
        tmp = []
        for e in range(num_examples):
            sim = 1 - distance.correlation(M1[l, e, :], M2[l, e, :])
            tmp.append(sim)
        sim = sum(tmp) / len(tmp)
        sims.append(sim)
    return sims


def sim_corr_avgfirst(M1, M2):
    num_layers, num_examples, num_features = M1.shape
    sims = []
    M1 = np.mean(M1, 1)
    M2 = np.mean(M2, 1)
    for l in range(num_layers):
        sim = distance.euclidean(M1[l, :], M2[l, :])
        sims.append(sim)
    return sims


def sim_euclid(M1, M2):
    num_layers, num_examples, num_features = M1.shape
    sims = []
    for l in range(num_layers):
        tmp = []
        for e in range(num_examples):
            sim = 1 - distance.euclidean(M1[l, e, :], M2[l, e, :])
            tmp.append(sim)
        sim = sum(tmp) / len(tmp)
        sims.append(sim)
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


def sim_pwcca(M1, M2):
    if M2.shape[1] > M1.shape[1]:
        print("Warning: shapes mismatch")
        M2 = M2[:, 0:M1.shape[1], :]
    num_layers, num_examples, num_features = M1.shape
    sims = []
    for l in range(num_layers):
        sims.append(robust_pwcaa(M1[l].transpose(1, 0), M2[l].transpose(1, 0)))
    return sims


def sim_svcca(M1, M2):
    if M2.shape[1] > M1.shape[1]:
        print("Warning: shapes mismatch")
        M2 = M2[:, 0:M1.shape[1], :]
    num_layers, num_examples, num_features = M1.shape
    sims = []
    for l in range(num_layers):
        sims.append(svcca(M1[l].transpose(1, 0), M2[l].transpose(1, 0)))
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


def neuron_interlingua_scores(reps1, reps2):
    # reps shape is (num_examples x num_features)
    num_layers, num_examples, num_features = reps1.shape
    res = []
    for l in range(num_layers):
        per_feat_err = []
        for fi in range(num_features):
            # err = reps1[l, :, fi] - reps2[l, :, fi]
            # err = np.mean(np.abs(err))
            err = distance.correlation(reps1[l, :, fi], reps2[l, :, fi])
            per_feat_err.append(err)
        res.append(per_feat_err)
    return 1 - np.array(res)
