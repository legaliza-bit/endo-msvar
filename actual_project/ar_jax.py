import pandas as pd
from jax.scipy import linalg
import jax.numpy as np
from tqdm import tqdm
from jax.scipy import stats
from jax.scipy import optimize
from jax import random
import jax
from functools import partial

key = random.PRNGKey(seed=42)


# @partial(jax.jit, static_argnames=['n_feat', 'n_state_feat'])
def _init_params(n_feat, n_state_feat, coeffs=None, state_coeffs=None, rho=None, sigmas=None):
    coeffs = (
        random.uniform(key=key, minval=-0.99, maxval=0.99, shape=(n_feat, 2))
        if not coeffs
        else coeffs
    )
    state_coeffs = (
        random.uniform(key=key, minval=-0.99, maxval=0.99, shape=(n_state_feat, 2))
        if not state_coeffs
        else state_coeffs
    )
    rho = random.uniform(key=key, minval=-0.99, maxval=0.99, shape=(1,)) if not rho else rho
    sigmas = random.uniform(key=key, minval=0.01, maxval=1, shape=(2,)) if not sigmas else sigmas
    return {"coeffs": coeffs,
            "state_coeffs": state_coeffs,
            "rho": rho,
            "sigmas": sigmas}


@jax.jit
def init_state_probs(state_exog, state_coeffs):
    """
    vector of P(S_t=i | mean(z))
    """
    mean_state_exog = np.mean(state_exog, axis=0).reshape(1, -1)

    # matrix of P(S_t=i | S_t=j, mean(z))
    P = calc_tvtp(mean_state_exog, state_coeffs)
    assert P.shape == (1, 2, 2), P.shape
    P = P.reshape((2, 2))
    A = np.concatenate([np.diag(np.array([1, 1])) - P, np.ones((1, 2))])
    E = np.array([0, 0, 1])

    # vector of P(S_t=i | mean(z))
    init_probs = (linalg.inv(A.T @ A) @ A.T @ E).reshape(2, 1)
    assert init_probs.shape == (2, 1), init_probs.shape
    return init_probs


@jax.jit
def calc_tvtp(state_exog, state_coeffs):
    """For both states
    matrix of P(S_t=i | S_{t-1}=j, z_t)
    """
    # shape (2, T, 2)
    T = state_exog.shape[0]
    tvtp = (
        stats.norm.cdf(
            np.kron(
                np.matmul(state_exog, state_coeffs).reshape(T, 1, 2),
                np.array([[1], [-1]])
            )
        )
    )
    # assert (tvtp.sum(axis=1) == np.ones((T, 2))).all()
    return tvtp


@jax.jit
def calc_endo_tp(state_exog, resids, sigmas, state_coeffs, rho):
    """For both states
    matrix of P(S_t=i | S_{t-1}=j, z_t, e_t)
    """

    T = state_exog.shape[0]
    endo_tp = (
        stats.norm.cdf(
            (np.kron(
                np.matmul(state_exog, state_coeffs).reshape(T, 1, 2),
                np.array([[1], [1]])
            ) - rho
             * np.kron(
                np.moveaxis(
                    resids.reshape(T, 1, 2)
                    / sigmas.reshape(1, 2), -1, -2
                ),
                np.array([[1, 1]])
            ))
            / np.sqrt(1 - rho**2) * np.array([[1], [-1]]))
    )
    # assert (endo_tp.sum(axis=1) == np.ones((T, 2))).all()
    return endo_tp


@jax.jit
def calc_residuals(endog, exog, coeffs, autoregressive=False):
    if autoregressive:
        # Add [0] for the constant
        # prev_mu = prev_mu + [0]
        # em1 = (endog[i] - mu[0]) - ((exog[i, :] - prev_mu) @ self.coeffs[:, 0])
        # em2 = (endog[i] - mu[1]) - ((exog[i, :] - prev_mu) @ self.coeffs[:, 1])
        raise NotImplementedError
    else:
        resids = endog - exog @ coeffs
        assert resids.shape == (len(endog), 2), resids.shape
        return resids


@jax.jit
def _calc_cond_density(resids, sigmas, trans_probs, endo_tp):
    """
    f(y_t | S_t=i, S_{t-1}=j, Y_{t-1}, Z_t; params)

    resids: np.ndarray
        vector of shape (2, 1)
    sigmas: np.ndarray
        vector of shape (2, )
    endo_tp: np.ndarray
        matrix of shape (2, 2)

    """
    exog_part = stats.norm.pdf(resids.reshape(2, 1) / sigmas.reshape(2, 1)) / sigmas.reshape(2, 1)
    assert exog_part.shape == (2, 1), exog_part.shape

    tp_ratio = endo_tp / trans_probs
    assert tp_ratio.shape == (2, 2), tp_ratio.shape
    return tp_ratio * exog_part


@jax.jit
def calc_cond_llhs(resids, sigmas, trans_probs, state_probs, endo_tp):
    """
    f(y_t | Y_{t-1}, Z_t; params) = f(y_t | S_t=i, S_{t-1}=j, Y_{t-1}, Z_t; params)
                                    * p_{ij,t} * P(S_{t-1} | Y_{t-1}, Z_{t-1}; params)
    """
    cond_dens = _calc_cond_density(resids, sigmas, trans_probs, endo_tp)

    llhs = cond_dens * trans_probs * state_probs.reshape(1, 2)

    assert llhs.shape == (2, 2), llhs.shape

    return llhs


@jax.jit
def hamilton_filter(params, endog, exog, state_exog):
    """
    Perform Hamilton Filtering to (1) compute the likelihood of each state,
    and (2) sample the states S_{1:T} according to these likelihoods

    vars:
        Variances in states 1 and 2
    mu:
        Vector of means
    Outputs a list of length T with the sampled states
    """
    params = params[0] if isinstance(params, list) else params
    coeffs = params["coeffs"]
    state_coeffs = params["state_coeffs"].reshape(1, 2)
    rho = params["rho"]
    sigmas = params["sigmas"]

    T = len(endog)

    filter = np.zeros((T, 2))
    total_likelihood = 0

    resids = calc_residuals(endog, exog, coeffs)

    # matrix of P(S_t | mean(z))
    state_probs = init_state_probs(state_exog, state_coeffs)

    # matrix of P(S_t=i | S_t=j, Z_t)
    tvtp = calc_tvtp(state_exog, state_coeffs)
    # matrix of P(S_t=i | S_{t-1}=j, z_t, e_t)
    endo_tp = calc_endo_tp(state_exog, resids, sigmas, state_coeffs, rho)

    for i in range(T):

        tp = tvtp[i, :, :]
        assert tp.shape == (2, 2), tp.shape

        etp = endo_tp[i, :, :]
        assert etp.shape == (2, 2), etp.shape

        assert resids[i, :].shape == (2,)

        # f(y_t | S_t=i, S_{t-1}=j, Z_t, Y_{t-1}; params)
        cond_llhs = calc_cond_llhs(resids[i, :], sigmas, tp, state_probs, etp)

        # f(y_t | Z_t, Y_{t-1}; params)
        llh = np.sum(cond_llhs)

        # P(S_t=i | Z_t, Y_{t-1}; params)
        state_probs = np.sum(cond_llhs, axis=1) / llh

        filter = filter.at[i, :].set(state_probs)
        total_likelihood += np.log(llh)

    return total_likelihood, tvtp, filter


@jax.jit
def obj_func(params, endog, exog, state_exog):
    total_likelihood, _, _ = hamilton_filter(params, endog, exog, state_exog)
    return -total_likelihood


@jax.jit
def sample_states(tvtp, filter):
    # (2) sample the states S_{1:T} according to these likelihoods
    # Perform sampling by drawing from Unif(0,1)
    # Then assign state based on sampled value and transition matrix cutoffs
    # First, sample the last state

    T = filter.shape[0]
    states = np.zeros(T)
    p1 = filter[-1, 0]
    p2 = filter[-1, 1]
    p = p1 / (p1 + p2)
    u = random.uniform(key=key, minval=0, maxval=1)  # Unif(0,1), draw 1
    states[-1] = 1 if u >= p else 0

    # Then, backward sample the rest
    for i in range(T - 2, -1, -1):
        if states[i] == 0:
            p00 = tvtp[0, 0] * filter[i, 0]
            p01 = tvtp[0, 1] * filter[i, 1]

        if states[i] == 1:
            p00 = tvtp[1, 0] * filter[i, 0]
            p01 = tvtp[1, 1] * filter[i, 1]

        u = random.uniform(key=key, minval=0, maxval=1)  # Unif(0,1), draw 1
        p = p00 / (p01 + p00)
        states[i] = 0 if u < p[0] else 1

    states = states.astype(int)  # Use integers!
    return states
