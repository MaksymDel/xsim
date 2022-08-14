# Credit: this is a based on https://github.com/moskomule/anatome

from __future__ import annotations

import torch
from torch import Tensor
from typing import Tuple


def compute_similarity(x: Tensor, y: Tensor, sim_name: str):
    x, y = maybe_convert_to_torch(x, y)

    if "cka" in sim_name:
        res = linear_cka_distance(x, y, False)
    elif "svcca" in sim_name:
        accept_rate = float(sim_name.split("_")[1])
        res = svcca_distance(x, y, accept_rate=accept_rate, backend="svd")
    elif "pwcca" in sim_name:
        res = pwcca_distance(x, y, "svd")
    elif "cca" in sim_name:
        res = cca_distace(x, y, "svd")
    # elif sim == 'cca':
    #     res = 1 - cca(x, y, 'svd')[2].sum()
    #     # TODO: check for correctness
    #     # TODO: correlate 2 components?
    else:
        raise ValueError(f"{sim_name} is not the known similarity")

    return 1 - res.item()


def cca_distace(x, y, backand="svd"):
    div = min(x.size(1), y.size(1))
    a, b, diag = cca(x, y, backend=backand)
    return 1 - diag.sum() / div


def _zero_mean(input: Tensor, dim: int) -> Tensor:
    return input - input.mean(dim=dim, keepdim=True)


def cca_by_svd(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """CCA using only SVD.
    For more details, check Press 2011 "Canonical Correlation Clarified by Singular Value Decomposition"

    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    # torch.svd(x)[1] is vector
    u_1, s_1, v_1 = torch.svd(x)
    u_2, s_2, v_2 = torch.svd(y)
    uu = u_1.t() @ u_2
    u, diag, v = torch.svd(uu)
    # v @ s.diag() = v * s.view(-1, 1), but much faster
    a = (v_1 * s_1.reciprocal_().unsqueeze_(0)) @ u
    b = (v_2 * s_2.reciprocal_().unsqueeze_(0)) @ v
    return a, b, diag


def cca_by_qr(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """CCA using QR and SVD.
    For more details, check Press 2011 "Canonical Correlation Clarified by Singular Value Decomposition"

    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    q_1, r_1 = torch.qr(x)
    q_2, r_2 = torch.qr(y)
    qq = q_1.t() @ q_2
    u, diag, v = torch.svd(qq)
    a = torch.inverse(r_1) @ u
    b = torch.inverse(r_2) @ v
    return a, b, diag


def cca(x: Tensor, y: Tensor, backend: str) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute CCA, Canonical Correlation Analysis

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        backend: svd or qr

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    if x.size(0) != y.size(0):
        raise ValueError(
            f"x.size(0) == y.size(0) is expected, but got {x.size(0)=}, {y.size(0)=} instead."
        )

    if x.size(0) < x.size(1):
        raise ValueError(f"x.size(0) >= x.size(1) is expected, but got {x.size()=}.")

    if y.size(0) < y.size(1):
        raise ValueError(f"y.size(0) >= y.size(1) is expected, but got {y.size()=}.")

    if backend not in ("svd", "qr"):
        raise ValueError(f"backend is svd or qr, but got {backend}")

    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    return cca_by_svd(x, y) if backend == "svd" else cca_by_qr(x, y)


def _svd_reduction(input: Tensor, accept_rate: float) -> Tensor:
    left, diag, right = torch.svd(input)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(
        ratio < accept_rate,
        input.new_ones(1, dtype=torch.long),
        input.new_zeros(1, dtype=torch.long),
    ).sum()
    return input @ right[:, :num]


def svcca_distance(x: Tensor, y: Tensor, accept_rate: float, backend: str) -> Tensor:
    """Singular Vector CCA proposed in Raghu et al. 2017.

    Args:
        x: input tensor of Shape DxH, where D>H
        y: input tensor of Shape DxW, where D>H
        accept_rate: 0.99
        backend: svd or qr

    Returns:

    """

    x = _svd_reduction(x, accept_rate)
    y = _svd_reduction(y, accept_rate)
    div = min(x.size(1), y.size(1))
    a, b, diag = cca(x, y, backend)
    return 1 - diag.sum() / div


def pwcca_distance(x: Tensor, y: Tensor, backend: str) -> Tensor:
    """Projection Weighted CCA proposed in Marcos et al. 2018.

    Args:
        x: input tensor of Shape DxH, where D>H
        y: input tensor of Shape DxW, where D>H
        backend: svd or qr

    Returns:

    """

    a, b, diag = cca(x, y, backend)
    alpha = (x @ a).abs_().sum(dim=0)
    alpha /= alpha.sum()
    return 1 - alpha @ diag


def _debiased_dot_product_similarity(
    z: Tensor,
    sum_row_x: Tensor,
    sum_row_y: Tensor,
    sq_norm_x: Tensor,
    sq_norm_y: Tensor,
    size: int,
) -> Tensor:
    return (
        z
        - size / (size - 2) * (sum_row_x @ sum_row_y)
        + sq_norm_x * sq_norm_y / ((size - 1) * (size - 2))
    )


def maybe_convert_to_torch(x, y):
    if not isinstance(x, Tensor):
        x = torch.from_numpy(x)

    if not isinstance(y, Tensor):
        y = torch.from_numpy(y)

    return x, y


def linear_cka_distance(x: Tensor, y: Tensor, reduce_bias: bool) -> Tensor:
    """Linear CKA used in Kornblith et al. 19

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        reduce_bias: debias CKA estimator, which might be helpful when D is limited

    Returns:

    """
    x, y = maybe_convert_to_torch(x, y)

    if x.size(0) != y.size(0):
        print(type(x))
        print(x.size())
        print(x.shape)
        raise ValueError(
            f"x.size(0) == y.size(0) is expected, but got {x.size(0)=}, {y.size(0)=} instead."
        )

    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    dot_prod = (y.t() @ x).norm("fro").pow(2)
    norm_x = (x.t() @ x).norm("fro")
    norm_y = (y.t() @ y).norm("fro")

    if reduce_bias:
        size = x.size(0)
        # (x @ x.t()).diag()
        sum_row_x = torch.einsum("ij,ij->i", x, x)
        sum_row_y = torch.einsum("ij,ij->i", y, y)
        sq_norm_x = sum_row_x.sum()
        sq_norm_y = sum_row_y.sum()
        dot_prod = _debiased_dot_product_similarity(
            dot_prod, sum_row_x, sum_row_y, sq_norm_x, sq_norm_y, size
        )
        norm_x = _debiased_dot_product_similarity(
            norm_x.pow_(2), sum_row_x, sum_row_y, sq_norm_x, sq_norm_y, size
        )
        norm_y = _debiased_dot_product_similarity(
            norm_y.pow_(2), sum_row_x, sum_row_y, sq_norm_x, sq_norm_y, size
        )

    r = dot_prod / (norm_x * norm_y)
    return 1 - r
