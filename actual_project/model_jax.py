import pandas as pd
from jax.scipy import linalg
import jax.numpy as np
from tqdm import tqdm
from jax.scipy import stats
from jax.scipy import optimize
from jax import random

key = random.PRNGKey(seed=42)


class EMSAR_jax:
    def __init__(self, autoregressive=False, order=None, state_order=None):
        self.autoregressive = autoregressive
        if autoregressive:
            if not order or state_order:
                raise ValueError("Specify order of autoregressions")

            self.order = order
            self.state_order = state_order

    @staticmethod
    def transform(arr):
        if isinstance(arr, pd.DataFrame):
            return arr.to_numpy()
        return arr

    @staticmethod
    def make_lagged(arr, order, axis=-1):
        """
        https://stackoverflow.com/questions/21229503/creating-an-numpy-matrix-with-a-lag
        return a running view of length 'order' over 'axis'
        the returned array has an extra last dimension, which spans the window
        """
        shape = list(arr.shape)
        shape[axis] -= order - 1
        assert shape[axis] > 0
        return np.lib.index_tricks.as_strided(
            arr, shape + [order], arr.strides + (arr.strides[axis],)
        )

    def fit(self, endog, exog, state_exog, search_init=False):
        self.n_obs = len(endog)
        self.n_feat = exog.shape[1]
        self.n_state_feat = state_exog.shape[1]

        endog, exog, state_exog = (
            self.transform(endog),
            self.transform(exog),
            self.transform(state_exog),
        )
        if self.autoregressive:
            endog = self.make_lagged(endog, self.order, axis=-1)
            exog = self.make_lagged(exog, self.order, axis=-1)
            state_exog = self.make_lagged(state_exog, self.state_order, axis=-1)

        params = self._search_init_params() if search_init else None
        self._init_params(params)

        self.states_vector = self.hamilton_filter()

    def _search_init_params(self):
        # return coeffs, state_coeffs, rho
        pass

    def _init_params(self, n_feat, n_state_feat, coeffs=None, state_coeffs=None, rho=None, sigmas=None):
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
        rho = random.uniform(key=key, minval=-0.99, maxval=0.99) if not rho else rho
        sigmas = random.uniform(key=key, minval=0.01, maxval=1, shape=(2,)) if not sigmas else sigmas
        return coeffs, state_coeffs, rho, sigmas

    def init_state_probs(self, state_exog, state_coeffs):
        """
        vector of P(S_t=i | mean(z))
        """
        mean_state_exog = np.mean(state_exog, axis=0).reshape(1, -1)

        # matrix of P(S_t=i | S_t=j, mean(z))
        P = self.calc_tvtp(mean_state_exog, state_coeffs)
        assert P.shape == (2, 1, 2), P.shape
        P = P.reshape((2, 2))
        A = np.concatenate([np.diag(np.array([1, 1])) - P, np.ones((1, 2))])
        E = np.array([0, 0, 1])

        # vector of P(S_t=i | mean(z))
        init_probs = (linalg.inv(A.T @ A) @ A.T @ E).reshape(2, 1)
        assert init_probs.shape == (2, 1), init_probs.shape
        return init_probs

    def _calc_tvtp(self, state_exog, state_coeffs):
        """For 1 state only"""
        lindex = state_exog @ state_coeffs
        assert lindex.shape == (len(state_exog), ), lindex.shape

        tvtp = np.stack([stats.norm.cdf(lindex), 1 - stats.norm.cdf(lindex)]).T
        assert tvtp.shape == (len(state_exog), 2), tvtp.shape
        return tvtp

    def calc_tvtp(self, state_exog, state_coeffs):
        """For both states
        matrix of P(S_t=i | S_{t-1}=j, z_t)
        """
        # shape (2, T, 2)
        tvtp = np.stack(
            [
                # shape (T, 2)
                self._calc_tvtp(state_exog, state_coeffs[:, 0]),
                self._calc_tvtp(state_exog, state_coeffs[:, 1]),
            ]
        )
        # assert np.array_equal(np.sum(tvtp, axis=2), np.full((2, len(state_exog)), 1))
        return tvtp

    def _calc_endo_tp(self, state_exog, resid, state_coeffs, sigma, rho):
        """
        For 1 state only.
        """
        lindex = (
            (state_exog @ state_coeffs.T - rho * resid / sigma)
            / np.sqrt(1 - rho**2)
        )
        assert lindex.shape == (1,), lindex.shape

        endo_tp = np.stack([stats.norm.cdf(lindex), 1 - stats.norm.cdf(lindex)]).T
        assert endo_tp.shape == (1, 2), endo_tp.shape

        return endo_tp

    def calc_endo_tp(self, state_exog, resids, sigmas, state_coeffs, rho):
        """For both states
        matrix of P(S_t=i | S_{t-1}=j, z_t, e_t)
        """
        endo_tp = np.stack(
            [
                # shape (1, 2)
                self._calc_endo_tp(state_exog, resids[0], state_coeffs[:, 0], sigmas[0], rho),
                self._calc_endo_tp(state_exog, resids[1], state_coeffs[:, 1], sigmas[1], rho),
            ]
        )
        # assert (np.sum(endo_tp, axis=2) == 1).all(), np.sum(endo_tp, axis=2)
        return endo_tp.reshape((2, 2))

    def calc_residuals(self, endog, exog, coeffs, autoregressive=False):
        if autoregressive:
            # Add [0] for the constant
            # prev_mu = prev_mu + [0]
            # em1 = (endog[i] - mu[0]) - ((exog[i, :] - prev_mu) @ self.coeffs[:, 0])
            # em2 = (endog[i] - mu[1]) - ((exog[i, :] - prev_mu) @ self.coeffs[:, 1])
            raise NotImplementedError
        else:
            resids = np.vstack([endog - exog.T @ coeffs[:, 0],
                                endog - exog.T @ coeffs[:, 1]])
            assert resids.shape == (2, 1), resids.shape
            return resids

    def _calc_cond_density(self, resids, sigmas, trans_probs, endo_tp):
        """
        f(y_t | S_t=i, S_{t-1}=j, Y_{t-1}, Z_t; params)

        resids: np.ndarray
            vector of shape (2, 1)
        sigmas: np.ndarray
            vector of shape (2, )
        endo_tp: np.ndarray
            matrix of shape (2, 2)

        """

        exog_part = stats.norm.pdf(resids) / sigmas.reshape(2, 1)
        assert exog_part.shape == (2, 1), exog_part.shape

        tp_ratio = endo_tp / trans_probs
        assert tp_ratio.shape == (2, 2), tp_ratio.shape

        return tp_ratio * exog_part

    def calc_cond_llhs(self, resids, sigmas, trans_probs, endo_tp, state_probs):
        """
        f(y_t | Y_{t-1}, Z_t; params) = f(y_t | S_t=i, S_{t-1}=j, Y_{t-1}, Z_t; params) 
                                        * p_{ij,t} * P(S_{t-1} | Y_{t-1}, Z_{t-1}; params)
        """
        cond_dens = self._calc_cond_density(resids, sigmas, trans_probs, endo_tp)
        assert state_probs.shape == (2, 1), state_probs.shape

        llhs = cond_dens * trans_probs * state_probs
        assert llhs.shape == (2, 2), llhs.shape

        return llhs

    def hamilton_filter(self, params, endog, exog, state_exog):
        """
        Perform Hamilton Filtering to (1) compute the likelihood of each state,
        and (2) sample the states S_{1:T} according to these likelihoods

        P :
            Initial transition probabilities, computed using mean(z)
        vars:
            Variances in states 1 and 2
        mu:
            Vector of means
        Outputs a list of length T with the sampled states
        """
        params = params[0] if isinstance(params, list) else params
        coeffs = params["coeffs"]
        state_coeffs = params["state_coeffs"]
        rho = params["rho"]
        sigmas = params["sigmas"]

        # (1) compute the likelihood of each state
        T = len(endog)

        filter = np.zeros((T, 2))
        total_likelihood = 0

        # matrix of P(S_t | mean(z))
        state_probs = self.init_state_probs(state_exog, state_coeffs)

        # matrix of P(S_t=i | S_t=j, Z_t)
        tvtp = self.calc_tvtp(state_exog, state_coeffs)

        for i in range(T):
            # Get state means for prev num_lags states
            # if i >= self.order:
            #     prev_mu = [mu[idx] for idx in S[i - self.order : i]]
            # else:
            #     [mu[idx] for idx in S[:i]]

            # # Pad in case too short
            # if len(prev_mu) < self.order:
            #     prev_mu = [0] * (self.order - len(prev_mu)) + prev_mu

            # residual based on current estimated state?
            resids = self.calc_residuals(endog[i, :],
                                         exog[i, :],
                                         coeffs)
            assert resids.shape == (2, 1), resids.shape

            endo_tp = self.calc_endo_tp(state_exog[i, :], resids, sigmas, state_coeffs, rho)

            tp = tvtp[:, i, :]
            assert tp.shape == (2, 2), tp.shape

            # f(y_t | S_t=i, S_{t-1}=j, Z_t, Y_{t-1}; params)
            cond_llhs = self.calc_cond_llhs(resids, sigmas, tp, endo_tp, state_probs)

            # f(y_t | Z_t, Y_{t-1}; params)
            llh = np.sum(cond_llhs)

            # P(S_t=i | Z_t, Y_{t-1}; params)
            prob = np.sum(cond_llhs, axis=1) / llh

            filter = filter.at[i, :].set(prob)
            total_likelihood += np.log(llh)

        self.tvtp = tvtp
        self.filter = filter

        return -total_likelihood

    def sample_states(self):
        # (2) sample the states S_{1:T} according to these likelihoods
        # Perform sampling by drawing from Unif(0,1)
        # Then assign state based on sampled value and transition matrix cutoffs
        # First, sample the last state
        tvtp = self.tvtp
        filter = self.filter
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

    def obj_func(self, params, args):
        return self.hamilton_filter(coeffs=params[0],
                                    state_coeffs=params[1],
                                    rho=params[2],
                                    sigmas=params[3],
                                    endog=args[0],
                                    exog=args[1],
                                    state_exog=args[2])

    def optimize_loglike(self, endog, exog, state_exog, n_iter=10):
        n_feat = exog.shape[1]
        n_state_feat = state_exog.shape[1]
        res = []
        for _ in range(n_iter):
            init_params = self._init_params(n_feat, n_state_feat)
            optimum = optimize.minimize(
                fun=self.obj_func,
                x0=init_params,
                args=[endog, exog, state_exog],
                method="L-BFGS-B",
            )
            if optimum.success:
                res.append(optimum)
        if res == []:
            return "No optimum found :("
        return res

    def lr_test():
        pass

    def untransform():
        pass
