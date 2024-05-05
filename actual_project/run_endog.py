import numpy as np
import jax.numpy as jnp
import jaxopt
from jax import random
import jax
from tqdm import tqdm
from model import sim_init_params, obj_func
from experiments import simulate

opt_params = {
    "coeffs": np.array([[1.0, -1.0], [1.0, -1.0]]),
    "sigmas": np.array([0.33, 0.67]),
    "rho": np.array(0.9),
}
exp_params = {"n_obs": 50, "p00": 0.7, "p11": 0.7}


@jax.jit
def run_experiment(opt_params, exp_params):

    sim = simulate(**opt_params, **exp_params)
    states, state_exog, exog, endog = sim["args"]
    state_coeffs = sim["state_coeffs"]
    opt_params["state_coeffs"] = state_coeffs
    n_feat = 2

    solver = jaxopt.LBFGS(fun=obj_func, maxiter=500, tol=1e-8)

    N = 100
    vals = []
    coeffs = np.zeros((N, 2, 2))
    rhos = np.zeros((N, 1))
    sigmas = np.zeros((N, 2))
    for i in tqdm(range(100)):
        key = random.PRNGKey(seed=i)
        init_params = sim_init_params(key, n_feat, state_coeffs)
        res = solver.run(init_params, endog=endog, exog=exog, state_exog=state_exog, exog_switching=False)
        coeffs[i, :, :] = res.params['coeffs']
        rhos[i, :] = res.params['rho']
        sigmas[i, :] = res.params['sigmas']
        vals.append(res.state.value)

    return coeffs, rhos, sigmas, vals


if __name__ == "__main__":
    coeffs, rhos, sigmas, vals = run_experiment(opt_params, exp_params)
    jnp.save('coeffs.npy', coeffs)
    jnp.save('rhos.npy', coeffs)
    jnp.save('sigmas.npy', coeffs)
    jnp.save('vals.npy', coeffs)
