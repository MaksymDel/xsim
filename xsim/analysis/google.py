# Credit: this uses https://github.com/google/svcca
# and https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb

import numpy as np

# from scipy.spatial import distance

import sys

sys.path.insert(0, "third_party/svcca")

# import cca_core


def compute_similarity(acts1, acts2, verbose=False, epsilon=1e-10, sim_name="cka"):
    

    if "svcca" in sim_name:
        k = int(sim_name.split("_")[1])
        print(f"computing svcca-{k}")
        res = compute_svcca_similarity(
            acts1, acts2, K=k, verbose=verbose, epsilon=epsilon
        )
        return res["mean"][0]
        # return np.mean(res["cca_coef1"][0:20])
    elif "pwcca" in sim_name:
        import pwcca
        print(f"computing pwcca")
        pwcca_mean, w, __ = pwcca.compute_pwcca(acts1, acts2, epsilon=epsilon)
        return pwcca_mean
    elif "cka" in sim_name:
        # print(f"computing cka")
        return feature_space_linear_cka(acts1.transpose(1, 0), acts2.transpose(1, 0))
    elif "cca" in sim_name:
        print(f"computing cca")
        res = cca_core.get_cca_similarity(
            acts1, acts2, epsilon=epsilon, compute_coefs=False, verbose=False
        )
        # return np.mean(res["cca_coef1"][0:200])  # res["mean"][0]
        return res["mean"][0]

    else:
        raise NotImplementedError(sim_name)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n
):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty
        - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def compute_svcca_similarity(acts1, acts2, K=20, verbose=False, epsilon=None):
    """Compute svcca similarity, adapted from tutorial on
    https://github.com/google/svcca/tree/master/tutorials.
    """
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:K] * np.eye(K), V1[:K])
    svacts2 = np.dot(s2[:K] * np.eye(K), V2[:K])
    svcca_results = cca_core.get_cca_similarity(
        svacts1, svacts2, epsilon=epsilon, verbose=verbose, compute_coefs=False
    )
    return svcca_results


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.
    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.
    Args:
      features_x: A num_examples x num_features matrix of features.
      features_y: A num_examples x num_features matrix of features.
      debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.
    Returns:
      The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum("ij,ij->i", features_x, features_x)
        sum_squared_rows_y = np.einsum("ij,ij->i", features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity,
            sum_squared_rows_x,
            sum_squared_rows_y,
            squared_norm_x,
            squared_norm_y,
            n,
        )
        normalization_x = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_x ** 2,
                sum_squared_rows_x,
                sum_squared_rows_x,
                squared_norm_x,
                squared_norm_x,
                n,
            )
        )
        normalization_y = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_y ** 2,
                sum_squared_rows_y,
                sum_squared_rows_y,
                squared_norm_y,
                squared_norm_y,
                n,
            )
        )

    return dot_product_similarity / (normalization_x * normalization_y)
