import jax.numpy as jnp
import numpy as np


def simulate(coeffs, sigmas, rho, n_obs, p00, p11):
    errors = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1, rho], [rho, 1]],
        size=n_obs,
    )

    state_coeffs = jnp.array([[p00, p11]])
    state_exog = jnp.hstack([np.zeros((n_obs, 1))])

    exog = np.hstack([np.ones((n_obs, 1)), np.random.normal(0, 2, size=(n_obs, 1))])
    endog = np.zeros(n_obs)

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

        endog[idx] = (
            exog[idx, :] @ coeffs[:, int(states[idx])]
            + errors[idx, 1] * sigmas[int(states[idx])]
        )

    return {
        "state_coeffs": state_coeffs,
        "args": (
            jnp.array(states),
            jnp.array(state_exog.reshape(n_obs, -1)),
            jnp.array(exog.reshape(n_obs, -1)),
            jnp.array(endog.reshape(n_obs, 1)),
        ),
    }
