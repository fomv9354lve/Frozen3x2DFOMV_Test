"""
FOMV: Field Operator for Measured Viability - Versión Didáctica
Autor: Osvaldo Morales
License: AGPL-3.0
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PARÁMETROS DEL MODELO (no modificar a menos que conozcas la dinámica)
# -----------------------------------------------------------------------------
params = {
    'alpha1': 0.1, 'alpha2': 0.2, 'delta': 0.05, 'beta1': 0.3,
    'gamma1': 0.2, 'gamma2': 0.1, 'gamma3': 0.1, 'phi1': 0.3,
    'phi2': 0.2, 'psi1': 0.2, 'psi2': 0.2, 'kappa1': 0.2, 'kappa2': 0.1,
    'Ec': 0.1, 'Er': 0.5, 'Lc': 1.5, 'Lr': 0.8,
}

# -----------------------------------------------------------------------------
# PARÁMETROS DE SIMULACIÓN (ajusta según necesidad)
# -----------------------------------------------------------------------------
sim_params = {
    'sigma': 0.05,
    'Tmax': 50,
    'R': 500,
    'Bgrid': 20,
    'Mgrid': 20,
    'B_range': [0, 1.2],
    'M_range': [0, 0.8],
    'bootstrap_reps': 100,
    'alpha': 0.05,
    'fast_samples': 20,
    'n_cores': mp.cpu_count(),
    'seed': 42
}

# -----------------------------------------------------------------------------
# FUNCIONES DE DINÁMICA (optimizadas)
# -----------------------------------------------------------------------------
def sigmoid(x): return 1/(1+np.exp(-x))

def hard_nonlinear_dynamics_vectorized(x, theta, eta):
    B, M, E, G, T, C = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5]
    B_star = B + theta['alpha1']*(1 - E) - theta['alpha2']*G
    M_star = (1 - theta['delta'])*M + theta['beta1'] * sigmoid(B - T)
    E_star = E + theta['gamma1']*G - theta['gamma2']*B - theta['gamma3']*M
    G_star = G + theta['phi1']*E - theta['phi2']*(B + M)*(1 - T)
    T_star = T - theta['psi1']*M*(1 - G) + theta['psi2']*G
    C_star = C + theta['kappa1']*T - theta['kappa2']*B
    x_star = np.column_stack([B_star, M_star, E_star, G_star, T_star, C_star]) + eta
    return np.clip(x_star, 0, 1)

def generate_noise_vectorized(sigma, n):
    beta = np.random.beta(2, 2, size=(n, 6))
    u = 2 * beta - 1
    return sigma * u

def is_collapsed_vectorized(x, theta):
    B, M, E = x[:,0], x[:,1], x[:,2]
    return (E <= theta['Ec']) | (B + M >= theta['Lc'])

def is_recovered_vectorized(x, theta):
    B, M, E = x[:,0], x[:,1], x[:,2]
    return (E >= theta['Er']) & (B + M <= theta['Lr'])

def simulate_trajectories_vectorized(x0, theta, sigma, Tmax):
    n = x0.shape[0]
    x = x0.copy()
    absorptions = np.full(n, None, dtype=object)
    times = np.full(n, Tmax, dtype=int)
    active = np.ones(n, dtype=bool)
    for t in range(Tmax):
        if not np.any(active): break
        collapsed = is_collapsed_vectorized(x[active], theta)
        recovered = is_recovered_vectorized(x[active], theta)
        idx_active = np.where(active)[0]
        absorptions[idx_active[collapsed]] = 'C'
        times[idx_active[collapsed]] = t
        absorptions[idx_active[recovered]] = 'R'
        times[idx_active[recovered]] = t
        active[idx_active[collapsed]] = False
        active[idx_active[recovered]] = False
        if not np.any(active): break
        eta = generate_noise_vectorized(sigma, np.sum(active))
        x[active] = hard_nonlinear_dynamics_vectorized(x[active], theta, eta)
    return absorptions, times

def generate_fast_samples(B, M, theta, sigma, n_samples, burnin=500, seed=None):
    if seed is not None:
        np.random.seed(seed)
    samples = []
    fast = np.random.uniform(0, 1, size=4)
    total_steps = n_samples + burnin
    for i in range(total_steps):
        x = np.array([B, M, fast[0], fast[1], fast[2], fast[3]])
        eta = generate_noise_vectorized(sigma, 1)[0]
        x_new = hard_nonlinear_dynamics_vectorized(x.reshape(1,-1), theta, eta.reshape(1,-1))
        fast = x_new[0, 2:]
        if i >= burnin:
            samples.append(fast.copy())
    samples = np.array(samples)
    return samples, (np.mean(samples[:,0]), np.mean(samples[:,1]),
                     np.mean(samples[:,2]), np.mean(samples[:,3]))

def compute_point(BM, theta, sigma, Tmax, R, fast_samples, base_seed):
    B, M = BM
    seed = base_seed + int(B * 10000 + M * 1000)  # determinista
    np.random.seed(seed)
    try:
        fast_arr, (Ea, Ga, Ta, Ca) = generate_fast_samples(B, M, theta, sigma, fast_samples)
        all_times_C = []
        q_sum = 0.0
        total_traj = 0
        for fast in fast_arr:
            x0 = np.tile(np.array([B, M, fast[0], fast[1], fast[2], fast[3]]), (R, 1))
            absorptions, times = simulate_trajectories_vectorized(x0, theta, sigma, Tmax)
            q_sum += np.sum(absorptions == 'R')
            total_traj += R
            all_times_C.extend(times[absorptions == 'C'])
        q_hat = q_sum / total_traj if total_traj>0 else np.nan
        mfpt_hat = np.mean(all_times_C) if all_times_C else np.nan
        return (B, M, q_hat, mfpt_hat, all_times_C, Ea, Ga, Ta, Ca)
    except Exception as e:
        print(f"Error en ({B:.2f},{M:.2f}): {e}")
        return (B, M, np.nan, np.nan, [], np.nan, np.nan, np.nan, np.nan)

def estimate_on_grid_parallel(B_grid, M_grid, theta, sigma, Tmax, R,
                              fast_samples, n_cores, base_seed):
    points = [(B, M) for B in B_grid for M in M_grid]
    func = partial(compute_point, theta=theta, sigma=sigma,
                   Tmax=Tmax, R=R, fast_samples=fast_samples,
                   base_seed=base_seed)
    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(pool.imap(func, points), total=len(points), desc="Puntos de grid"))
    nB, nM = len(B_grid), len(M_grid)
    Q = np.full((nB, nM), np.nan)
    MFPT = np.full((nB, nM), np.nan)
    E = np.full((nB, nM), np.nan); G = np.full((nB, nM), np.nan)
    T = np.full((nB, nM), np.nan); C = np.full((nB, nM), np.nan)
    times_data = {}
    idx = 0
    for i, B in enumerate(B_grid):
        for j, M in enumerate(M_grid):
            (_, _, q, mfpt, times, Ea, Ga, Ta, Ca) = results[idx]
            Q[i,j] = q
            MFPT[i,j] = mfpt
            E[i,j] = Ea; G[i,j] = Ga; T[i,j] = Ta; C[i,j] = Ca
            times_data[(i,j)] = times
            idx += 1
    return Q, MFPT, times_data, E, G, T, C

# -----------------------------------------------------------------------------
# EJECUCIÓN DE LA SIMULACIÓN
# -----------------------------------------------------------------------------
print("="*60)
print("🔬 FOMV: Simulación en curso...")
print("="*60)

B_grid = np.linspace(sim_params['B_range'][0], sim_params['B_range'][1], sim_params['Bgrid'])
M_grid = np.linspace(sim_params['M_range'][0], sim_params['M_range'][1], sim_params['Mgrid'])

print(f"\n🧮 Grid: {sim_params['Bgrid']}×{sim_params['Mgrid']} puntos")
print(f"⚙️  Parámetros: R={sim_params['R']}, fast_samples={sim_params['fast_samples']}, sigma={sim_params['sigma']}")
print(f"🖥️  Usando {sim_params['n_cores']} núcleos\n")

Q, MFPT, times_data, E, G, T, C = estimate_on_grid_parallel(
    B_grid, M_grid, params, sim_params['sigma'],
    sim_params['Tmax'], sim_params['R'],
    sim_params['fast_samples'], sim_params['n_cores'],
    sim_params['seed']
)

print("\n✅ Simulación completada.")
