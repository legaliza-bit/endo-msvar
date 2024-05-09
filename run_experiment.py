import jax.numpy as jnp
import jaxopt
import numpy as np
from jax import random
from model_exog import obj_func, sim_init_params
from tqdm import tqdm

from simulate_data import simulate

opt_params = {
    "coeffs": jnp.array([[1.0, -1.0], [1.0, -1.0]]),
    "sigmas": jnp.array([0.33, 0.67]),
    "rho": jnp.array(0.0),
}
exp_params = {"n_obs": 50, "p00": 0.7, "p11": 0.7}


def run_experiment(opt_params, exp_params):

    sim = simulate(**opt_params, **exp_params)
    states, state_exog, exog, endog = sim["args"]
    state_coeffs = sim["state_coeffs"]
    opt_params["state_coeffs"] = state_coeffs
    n_feat = 2

    solver = jaxopt.LBFGS(fun=obj_func, maxiter=500, tol=1e-8)

    N = 50
    vals = jnp.zeros((N, 1))
    coeffs = jnp.zeros((N, 2, 2))
    rhos = jnp.zeros((N, 1))
    sigmas = jnp.zeros((N, 2))
    for i in tqdm(range(N)):
        key = random.PRNGKey(seed=i)
        init_params = sim_init_params(key, n_feat, state_coeffs)
        res = solver.run(
            init_params,
            endog=endog,
            exog=exog,
            state_exog=state_exog,
            exog_switching=True,
        )

        coeffs = coeffs.at[i, :, :].set(res.params["coeffs"])
        rhos = rhos.at[i, :].set(res.params["rho"])
        sigmas = sigmas.at[i, :].set(res.params["sigmas"])
        vals = vals.at[i, :].set(res.state.value)

    return coeffs, rhos, sigmas, vals


if __name__ == "__main__":
    coeffs, rhos, sigmas, vals = run_experiment(opt_params, exp_params)
    jnp.save("coeffs.npy", coeffs)
    jnp.save("rhos.npy", rhos)
    jnp.save("sigmas.npy", sigmas)
    jnp.save("vals.npy", vals)
