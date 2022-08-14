from xsim.analysis import anatome
from xsim.analysis import google


def compute_similarity_all_layers(M1, M2, sim_name, skip_embedding_layer=True):
    if not skip_embedding_layer:
        raise NotImplementedError(
            "Google's PWCCA and others fail to converge for uncontextual layer for some reason."
        )

    def maybe_set_impl_type(sim_name, impl_type):
        prefix = f"{impl_type}_"
        if prefix in sim_name:
            new_sim_name = sim_name.replace(prefix, "")
            return new_sim_name, impl_type
        else:
            return sim_name, "Bad prefix"

    impl_types = ["anatome", "google"]
    for it in impl_types:
        sim_name, impl = maybe_set_impl_type(sim_name, it)
        if impl != "Bad prefix":
            break

    if impl == "anatome":
        compute_similarity_func = anatome.compute_similarity
    elif impl == "google":
        compute_similarity_func = google.compute_similarity
    else:
        raise NotImplementedError(f"{impl}")

    if M2.shape[1] > M1.shape[1]:
        print("Warning: shapes mismatch")
        M2 = M2[:, 0 : M1.shape[1], :]
    num_layers, num_examples, num_features = M1.shape
    sims = []

    for i in range(num_layers - 1):
        l = i + 1  # do not compute for uncontextual embeddings
        #    for l in range(num_layers):
        x = M1[l]
        y = M2[l]

        # x = x[:10000]
        # y = y[:10000]
        if impl == "google":
            x = x.transpose(1, 0)
            y = y.transpose(1, 0)
            # print(x.shape, y.shape)

        score = compute_similarity_func(x, y, sim_name=sim_name)

        sims.append(score)
        # print(f"layer: {l}: {score}")
    return sims


#
# def sim_svcca_conseq_layers(M):
#     num_layers, num_examples, num_features = M.shape
#     sims = []
#     for ind1 in range(num_layers - 1):
#         ind2 = ind1 + 1
#         reps1, reps2 = M[ind1], M[ind2]
#         sims.append(svcca(reps1.transpose(1, 0), reps2.transpose(1, 0)))
#         # sims.append(feature_space_linear_cka(reps1.transpose(1,0), reps2.transpose(1,0)))
#     return sims
#
#
# def sim_cka_conseq_layers(M):
#     num_layers, num_examples, num_features = M.shape
#     sims = []
#     for ind1 in range(num_layers - 1):
#         ind2 = ind1 + 1
#         reps1, reps2 = M[ind1], M[ind2]
#         sims.append(feature_space_linear_cka(reps1, reps2))
#         # sims.append(feature_space_linear_cka(reps1.transpose(1,0), reps2.transpose(1,0)))
#     return sims


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
