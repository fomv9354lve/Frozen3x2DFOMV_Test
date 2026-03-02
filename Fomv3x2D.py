"""
FOMV: Field Operator for Measured Viability - Cubo Multivariable
Autor: Osvaldo Morales
Licencia: AGPL-3.0
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PARÁMETROS DEL MODELO
# =============================================================================
params = {
    'alpha1': 0.1, 'alpha2': 0.2, 'delta': 0.05, 'beta1': 0.3,
    'gamma1': 0.2, 'gamma2': 0.1, 'gamma3': 0.1, 'phi1': 0.3,
    'phi2': 0.2, 'psi1': 0.2, 'psi2': 0.2, 'kappa1': 0.2, 'kappa2': 0.1,
    'Ec': 0.1, 'Er': 0.5, 'Lc': 1.5, 'Lr': 0.8,
}

# =============================================================================
# PARÁMETROS DE SIMULACIÓN (alta resolución)
# =============================================================================
sim_params = {
    'sigma': 0.05,
    'Tmax': 50,
    'R': 500,                   # trayectorias por punto
    'Bgrid': 20,                 # 20 puntos en B
    'Mgrid': 20,                 # 20 puntos en M
    'B_range': [0, 1.2],
    'M_range': [0, 0.8],
    'bootstrap_reps': 100,
    'alpha': 0.05,
    'fast_samples': 20,          # muestras rápidas por punto
    'n_cores': mp.cpu_count(),
    'seed': 42                   # semilla base para reproducibilidad
}

# =============================================================================
# FUNCIONES DE DINÁMICA (optimizadas)
# =============================================================================
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def hard_nonlinear_dynamics_vectorized(x, theta, eta):
    """
    x: (n,6) array con estados [B, M, E, G, T, C]
    eta: (n,6) ruido aditivo
    """
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
    """
    Genera n vectores de ruido de 6 dimensiones con distribución
    proporcional a prod(0.75*(1 - u^2)) en [-1,1]^6.
    Versión eficiente usando muestreo directo con Beta(2,2).
    """
    # Generar u_i = 2*beta_i - 1, donde beta_i ~ Beta(2,2)
    # Beta(2,2) se puede generar como (U1+U2)/2? Pero usamos el generador directo.
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
    """
    Simula múltiples trayectorias desde condiciones iniciales x0.
    Retorna:
        absorptions: array de strings 'C', 'R' o None (si no absorción en Tmax)
        times: array de tiempos de absorción (Tmax si no absorción)
    """
    n = x0.shape[0]
    x = x0.copy()
    absorptions = np.full(n, None, dtype=object)  # se mantiene object por simplicidad
    times = np.full(n, Tmax, dtype=int)
    active = np.ones(n, dtype=bool)
    
    for t in range(Tmax):
        if not np.any(active):
            break
        collapsed = is_collapsed_vectorized(x[active], theta)
        recovered = is_recovered_vectorized(x[active], theta)
        
        # Índices globales de los que colapsan/recuperan
        idx_active = np.where(active)[0]
        idx_collapsed = idx_active[collapsed]
        idx_recovered = idx_active[recovered]
        
        absorptions[idx_collapsed] = 'C'
        times[idx_collapsed] = t
        absorptions[idx_recovered] = 'R'
        times[idx_recovered] = t
        
        # Actualizar máscara activa
        active[idx_collapsed] = False
        active[idx_recovered] = False
        
        if not np.any(active):
            break
        
        # Generar ruido solo para los activos
        eta = generate_noise_vectorized(sigma, np.sum(active))
        x[active] = hard_nonlinear_dynamics_vectorized(x[active], theta, eta)
    
    return absorptions, times

# =============================================================================
# GENERACIÓN DE MUESTRAS RÁPIDAS (con semilla local)
# =============================================================================
def generate_fast_samples(B, M, theta, sigma, n_samples, burnin=500, seed=None):
    """
    Genera una cadena para las variables rápidas (E,G,T,C) y devuelve
    las muestras y sus promedios.
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = []
    fast = np.random.uniform(0, 1, size=4)  # (E, G, T, C)
    total_steps = n_samples + burnin
    for i in range(total_steps):
        x = np.array([B, M, fast[0], fast[1], fast[2], fast[3]])
        eta = generate_noise_vectorized(sigma, 1)[0]
        x_new = hard_nonlinear_dynamics_vectorized(x.reshape(1,-1), theta, eta.reshape(1,-1))
        fast = x_new[0, 2:]
        if i >= burnin:
            samples.append(fast.copy())
    samples = np.array(samples)
    # Promedios de las rápidas
    E_mean = np.mean(samples[:,0])
    G_mean = np.mean(samples[:,1])
    T_mean = np.mean(samples[:,2])
    C_mean = np.mean(samples[:,3])
    return samples, (E_mean, G_mean, T_mean, C_mean)

# =============================================================================
# CÁLCULO PARA UN PUNTO (B, M) - con semilla única
# =============================================================================
def compute_point(BM, theta, sigma, Tmax, R, fast_samples, base_seed):
    B, M = BM
    # Crear semilla única a partir de la base y las coordenadas (evita correlaciones)
    seed = base_seed + hash((B, M)) % 2**32
    np.random.seed(seed)
    
    try:
        fast_arr, (E_avg, G_avg, T_avg, C_avg) = generate_fast_samples(B, M, theta, sigma, fast_samples)
        all_times_C = []
        q_sum = 0.0
        total_traj = 0
        for fast in fast_arr:
            x0 = np.tile(np.array([B, M, fast[0], fast[1], fast[2], fast[3]]), (R, 1))
            absorptions, times = simulate_trajectories_vectorized(x0, theta, sigma, Tmax)
            q_sum += np.sum(absorptions == 'R')
            total_traj += R
            all_times_C.extend(times[absorptions == 'C'])
        q_hat = q_sum / total_traj if total_traj > 0 else np.nan
        mfpt_hat = np.mean(all_times_C) if all_times_C else np.nan
        return (B, M, q_hat, mfpt_hat, all_times_C, E_avg, G_avg, T_avg, C_avg)
    except Exception as e:
        # En caso de error, devolver NaNs para que el punto sea ignorado
        print(f"Error en punto (B={B:.3f}, M={M:.3f}): {e}")
        return (B, M, np.nan, np.nan, [], np.nan, np.nan, np.nan, np.nan)

# =============================================================================
# ESTIMACIÓN EN GRID CON PARALELIZACIÓN
# =============================================================================
def estimate_on_grid_parallel(B_grid, M_grid, theta, sigma, Tmax, R,
                              fast_samples, n_cores, base_seed):
    points = [(B, M) for B in B_grid for M in M_grid]
    func = partial(compute_point, theta=theta, sigma=sigma,
                   Tmax=Tmax, R=R, fast_samples=fast_samples,
                   base_seed=base_seed)
    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(pool.imap(func, points), total=len(points), desc="Grid points"))

    nB, nM = len(B_grid), len(M_grid)
    # Inicializar matrices
    Q = np.full((nB, nM), np.nan)
    MFPT = np.full((nB, nM), np.nan)
    E_avg = np.full((nB, nM), np.nan)
    G_avg = np.full((nB, nM), np.nan)
    T_avg = np.full((nB, nM), np.nan)
    C_avg = np.full((nB, nM), np.nan)
    times_data = {}
    idx = 0
    for i, B in enumerate(B_grid):
        for j, M in enumerate(M_grid):
            (_, _, q, mfpt, times, Ea, Ga, Ta, Ca) = results[idx]
            Q[i,j] = q
            MFPT[i,j] = mfpt
            E_avg[i,j] = Ea
            G_avg[i,j] = Ga
            T_avg[i,j] = Ta
            C_avg[i,j] = Ca
            times_data[(i,j)] = times
            idx += 1
    return Q, MFPT, times_data, E_avg, G_avg, T_avg, C_avg

# =============================================================================
# BOOTSTRAP (opcional, se conserva igual)
# =============================================================================
def bootstrap_bands_from_times(times_data, B_grid, M_grid, bootstrap_reps, alpha=0.05):
    nB, nM = len(B_grid), len(M_grid)
    MFPT_hat = np.full((nB, nM), np.nan)
    MFPT_lower = np.full((nB, nM), np.nan)
    MFPT_upper = np.full((nB, nM), np.nan)
    for i in range(nB):
        for j in range(nM):
            times = times_data.get((i,j), [])
            if len(times) == 0:
                continue
            mfpt_hat = np.mean(times)
            MFPT_hat[i,j] = mfpt_hat
            boot_means = [np.mean(np.random.choice(times, size=len(times), replace=True))
                          for _ in range(bootstrap_reps)]
            MFPT_lower[i,j] = np.percentile(boot_means, 100 * alpha / 2)
            MFPT_upper[i,j] = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return MFPT_hat, MFPT_lower, MFPT_upper

# =============================================================================
# VISUALIZACIÓN 2D BÁSICA
# =============================================================================
def plot_mfpt_2d(B_grid, M_grid, MFPT, title="MFPT en (B, M)"):
    B_mesh, M_mesh = np.meshgrid(B_grid, M_grid, indexing='ij')
    plt.figure(figsize=(8,6))
    contour = plt.contourf(B_mesh, M_mesh, MFPT, levels=20, cmap='viridis')
    plt.colorbar(contour, label='MFPT')
    plt.xlabel('Backlog B')
    plt.ylabel('Memory M')
    plt.title(title)
    plt.show()

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("FOMV: Cubo multivariable con alta resolución (versión optimizada)")
    print("="*60)

    B_grid = np.linspace(sim_params['B_range'][0], sim_params['B_range'][1], sim_params['Bgrid'])
    M_grid = np.linspace(sim_params['M_range'][0], sim_params['M_range'][1], sim_params['Mgrid'])

    print(f"\nUsando {sim_params['n_cores']} núcleos. Grid: {sim_params['Bgrid']}×{sim_params['Mgrid']} puntos.")
    print("Estimando (puede tomar varios minutos)...")

    Q, MFPT, times_data, E_avg, G_avg, T_avg, C_avg = estimate_on_grid_parallel(
        B_grid, M_grid, params, sim_params['sigma'],
        sim_params['Tmax'], sim_params['R'],
        sim_params['fast_samples'], sim_params['n_cores'],
        sim_params['seed']
    )

    # Bootstrap para MFPT (opcional)
    MFPT_hat, MFPT_lower, MFPT_upper = bootstrap_bands_from_times(
        times_data, B_grid, M_grid, sim_params['bootstrap_reps'], sim_params['alpha']
    )

    # Gráfico 2D de referencia
    plot_mfpt_2d(B_grid, M_grid, MFPT_hat, title="MFPT en (B, M) - Alta resolución")

    # =========================================================================
    # CONSTRUIR DATAFRAME CON TODAS LAS VARIABLES
    # =========================================================================
    import pandas as pd
    B_vals = np.repeat(B_grid, len(M_grid))
    M_vals = np.tile(M_grid, len(B_grid))
    data = {
        'B': B_vals,
        'M': M_vals,
        'E': E_avg.flatten(),
        'G': G_avg.flatten(),
        'T': T_avg.flatten(),
        'C': C_avg.flatten(),
        'MFPT': MFPT_hat.flatten(),
        'Q': Q.flatten()
    }
    df = pd.DataFrame(data).dropna()
    # Proteger logaritmo: valores pequeños se reemplazan por 1e-10
    df['logMFPT'] = np.log(df['MFPT'].clip(lower=1e-10))
    print(f"\nDataFrame con {len(df)} puntos válidos. Columnas: {list(df.columns)}")

    # =========================================================================
    # VISUALIZACIÓN INTERACTIVA (requiere Jupyter / IPython)
    # =========================================================================
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import ipywidgets as widgets
        from IPython.display import display

        print("\n--- Visualización interactiva (requiere Jupyter) ---")
        var_options = ['B', 'M', 'E', 'G', 'T', 'C', 'MFPT', 'Q', 'logMFPT']

        x_widget = widgets.Dropdown(options=var_options, value='B', description='Eje X:')
        y_widget = widgets.Dropdown(options=var_options, value='M', description='Eje Y:')
        z_widget = widgets.Dropdown(options=var_options, value='T', description='Eje Z:')
        color_widget = widgets.Dropdown(options=var_options, value='logMFPT', description='Color:')
        size_widget = widgets.FloatSlider(value=3, min=1, max=10, step=0.5, description='Tamaño punto:')

        button = widgets.Button(description='Actualizar gráfico 3D')
        out = widgets.Output()

        def update_3d(b):
            with out:
                out.clear_output()
                fig = px.scatter_3d(df, x=x_widget.value, y=y_widget.value, z=z_widget.value,
                                    color=color_widget.value, color_continuous_scale='viridis',
                                    title=f'Cubo: {x_widget.value} vs {y_widget.value} vs {z_widget.value}',
                                    opacity=0.8)
                fig.update_traces(marker=dict(size=size_widget.value))
                fig.show()

        button.on_click(update_3d)
        display(widgets.VBox([x_widget, y_widget, z_widget, color_widget, size_widget, button, out]))

        # Widget para cortes 2D
        print("\n--- Cortes 2D con slider ---")
        fixed_var = widgets.Dropdown(options=var_options, value='M', description='Variable fija:')
        fixed_val = widgets.FloatSlider(min=df['M'].min(), max=df['M'].max(), step=0.05, description='Valor:')
        x2d = widgets.Dropdown(options=var_options, value='B', description='Eje X 2D:')
        y2d = widgets.Dropdown(options=var_options, value='T', description='Eje Y 2D:')
        color2d = widgets.Dropdown(options=var_options, value='logMFPT', description='Color 2D:')

        def update_2d(fixed_var, fixed_val, x2d, y2d, color2d):
            tol = 0.05 * (df[fixed_var].max() - df[fixed_var].min())
            subset = df[np.abs(df[fixed_var] - fixed_val) < tol]
            if len(subset) == 0:
                print("No hay datos en ese rango")
                return
            plt.figure(figsize=(8,6))
            sc = plt.scatter(subset[x2d], subset[y2d], c=subset[color2d],
                             cmap='viridis', s=50, edgecolor='k')
            plt.colorbar(sc, label=color2d)
            plt.xlabel(x2d); plt.ylabel(y2d)
            plt.title(f'{fixed_var} ≈ {fixed_val:.2f}')
            plt.grid(True)
            plt.show()

        interact_2d = widgets.interactive(update_2d,
                                          fixed_var=fixed_var,
                                          fixed_val=fixed_val,
                                          x2d=x2d,
                                          y2d=y2d,
                                          color2d=color2d)
        display(interact_2d)

    except ImportError as e:
        print("\nNota: Para visualización interactiva instala plotly, ipywidgets y pandas:")
        print("pip install plotly ipywidgets pandas")

    print("\n¡Proceso completado!")
