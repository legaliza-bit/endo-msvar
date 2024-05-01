import numpy as np
from scipy import stats
from scipy import linalg

def simulate(coeffs, sigmas, rho, n_obs, p00, p11):
    # get eps and eta
    errors = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1, rho * sigmas[0]], [rho * sigmas[0], sigmas[0] ** 2]],
        size=n_obs,
    )
    # randomize states
    states = np.zeros(n_obs)
    p = p00 / (p00 + p11)
    u = errors[0, 1]
    states[0] = 1 if u >= p else 0
    for idx, u in enumerate(errors[1:, 1]):
        if states[idx - 1] == 0:
            if u < p00:
                states[idx] = int(states[idx - 1])
            else:
                states[idx] = int(abs(states[idx - 1] - 1))
        else:
            if u < p11:
                states[idx] = int(states[idx - 1])
            else:
                states[idx] = int(abs(states[idx - 1] - 1))

    state_coeffs = np.array([[stats.norm.ppf(p00), stats.norm.ppf(p11)]])
    state_exog = np.ones((n_obs, 1))

    exog = np.hstack([
        np.ones((n_obs, 1)), np.random.normal(0, 2, size=(n_obs, 1))
    ])
    endog = np.zeros(n_obs)
    for i in range(n_obs):
        endog[i] = exog[i, :] @ coeffs[:, int(states[i])].T + errors[i, 1]

    return {
        "state_coeffs": state_coeffs,
        "args": (
            states,
            state_exog.reshape(n_obs, -1),
            exog.reshape(n_obs, -1),
            endog.reshape(n_obs, 1)),
    }
