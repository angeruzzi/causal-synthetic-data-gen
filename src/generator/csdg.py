import numpy as np

def get_effect_function(nonlinear=False, seed=None):
    if not nonlinear:
        return lambda x: x, "linear"
    
    rng = np.random.default_rng(seed)
    options = [
        (lambda x: x ** 2, "quadratic"),
        (np.sin, "sine"),
        (lambda x: np.log1p(x ** 2), "logarithmic")
    ]
    f, name = rng.choice(options)
    return f, name

def min_max_scale(arr):
    min_val = arr.min(axis=1, keepdims=True)
    max_val = arr.max(axis=1, keepdims=True)
    return 2 * (arr - min_val) / (max_val - min_val + 1e-8) - 1

def generate_synthetic_data(n, t, structure_type='direct', nonlinear=False, seed=42, intervention_type='pontual', config=None):
    
    """
    Gera dados sintéticos com estrutura causal:
    - 'direct':     T → Y         (usa betas['ty'])
    - 'chain':      T → X → Y     (usa betas['tx'], betas['xy'])
    - 'confounder': X → T, X → Y, T → Y (usa betas['xt'], betas['xy'], betas['ty'])

    Suporte a relações não lineares, intervenções e normalização.
    """
    np.random.seed(seed)

    # Defaults
    config = config or {}
    phi_T = config.get("phi_T", np.round(np.random.uniform(0.5, 1.0), 2))
    phi_Y = config.get("phi_Y", np.round(np.random.uniform(0.5, 1.0), 2))
    betas = config.get("betas", {})
    t_interv = config.get("t_interv")
    delta_T = config.get("delta_T", np.round(np.random.uniform(0.1, 1.0), 2) if t_interv is not None else None)
    normalize = config.get("normalize", True)

    # Ruído
    noise_T_range = config.get("noise_T_range", (-0.1, 0.1))
    noise_Y_range = config.get("noise_Y_range", (-0.1, 0.1))
    noise_X_range = config.get("noise_X_range", (-0.1, 0.1))  # Usado apenas se houver X

    # Geração dos ruídos
    eps_T = np.random.uniform(*noise_T_range, size=(n, t))
    eps_Y = np.random.uniform(*noise_Y_range, size=(n, t))
    eps_X = np.random.uniform(*noise_X_range, size=(n, t)) if structure_type in ['chain', 'confounder'] else None

    # Funções causais
    f_effect, f_name = get_effect_function(nonlinear, seed)
    f_x1, fx1_name = get_effect_function(nonlinear, seed + 1)
    f_x2, fx2_name = get_effect_function(nonlinear, seed + 2)


    # Geração automática dos betas se não fornecido
    def get_beta(key):
        if key not in betas:
            betas[key] = np.round(np.random.uniform(0.5, 2.0), 2)
        return betas[key]

    T = np.zeros((n, t), dtype=np.float32)
    Y = np.zeros((n, t), dtype=np.float32)
    X = np.zeros((n, t), dtype=np.float32) if structure_type in ['chain', 'confounder'] else None

    T[:, 0] = np.random.uniform(-1, 1, n)
    Y[:, 0] = np.random.uniform(-1, 1, n)
    if structure_type == 'confounder':
        X[:, :] = np.random.uniform(-1, 1, size=(n, t))

    sigma = 0.1
    eps_T = np.random.normal(0, sigma, size=(n, t))
    eps_Y = np.random.normal(0, sigma, size=(n, t))
    eps_X = np.random.normal(0, sigma, size=(n, t)) if structure_type == 'chain' else None

    for j in range(1, t):
        if structure_type == 'direct':
            T[:, j] = phi_T * T[:, j - 1] + eps_T[:, j]
            Y[:, j] = phi_Y * Y[:, j - 1] + get_beta("ty") * f_effect(T[:, j]) + eps_Y[:, j]

        elif structure_type == 'chain':
            T[:, j] = phi_T * T[:, j - 1] + eps_T[:, j]
            X[:, j] = get_beta("tx") * f_effect(T[:, j]) + eps_X[:, j]
            Y[:, j] = phi_Y * Y[:, j - 1] + get_beta("xy") * f_x1(X[:, j]) + eps_Y[:, j]

        elif structure_type == 'confounder':
            T[:, j] = phi_T * T[:, j - 1] + get_beta("xt") * f_x1(X[:, j]) + eps_T[:, j]
            Y[:, j] = phi_Y * Y[:, j - 1] + get_beta("ty") * f_effect(T[:, j]) + get_beta("xy") * f_x2(X[:, j]) + eps_Y[:, j]

    results = {
        "treatments": T,
        "outcomes": Y,
        "phi_T": phi_T,
        "phi_Y": phi_Y,
        "betas": betas,
        "noise_T_range": noise_T_range,
        "noise_Y_range": noise_Y_range,
        "noise_X_range": noise_X_range,
        "structure_type":  structure_type,
        "intervention_type": intervention_type,
        "effect_function": f_name,
        "effect_fx1": fx1_name,
        "effect_fx2": fx2_name,
        "normalize": normalize,
        "nonlinear": nonlinear,
        "seed": seed
    }

    if X is not None:
        results["covariates"] = X
        results["effect_fx1"] = fx1_name
        if structure_type == "confounder":
            results["effect_fx2"] = fx2_name

    if t_interv is not None:
        T_cf = np.zeros((n, t), dtype=np.float32)
        Y_cf = np.zeros((n, t), dtype=np.float32)
        X_cf = np.copy(X) if X is not None else None

        T_cf[:, :t_interv] = T[:, :t_interv]
        Y_cf[:, :t_interv] = Y[:, :t_interv]

        if intervention_type == 'gradual':
            k = max(1, t - t_interv + 1)

        for j in range(t_interv, t):
            if intervention_type == 'pontual':
                delta_t = delta_T if j == t_interv else 0.0
            elif intervention_type == 'continua':
                delta_t = delta_T
            elif intervention_type == 'gradual':
                delta_t = delta_T * (j - t_interv + 1) / k
            else:
                raise ValueError("Tipo de intervenção inválido")

            if structure_type == 'direct':
                T_cf[:, j] = phi_T * T_cf[:, j - 1] + eps_T[:, j] + delta_t
                Y_cf[:, j] = phi_Y * Y_cf[:, j - 1] + get_beta("ty") * f_effect(T_cf[:, j]) + eps_Y[:, j]

            elif structure_type == 'chain':
                T_cf[:, j] = phi_T * T_cf[:, j - 1] + eps_T[:, j] + delta_t
                X_cf[:, j] = get_beta("tx") * f_effect(T_cf[:, j]) + eps_X[:, j]
                Y_cf[:, j] = phi_Y * Y_cf[:, j - 1] + get_beta("xy") * f_x1(X_cf[:, j]) + eps_Y[:, j]

            elif structure_type == 'confounder':
                T_cf[:, j] = phi_T * T_cf[:, j - 1] + get_beta("xt") * f_x1(X_cf[:, j]) + eps_T[:, j] + delta_t
                Y_cf[:, j] = phi_Y * Y_cf[:, j - 1] + get_beta("ty") * f_effect(T_cf[:, j]) + get_beta("xy") * f_x2(X_cf[:, j]) + eps_Y[:, j]

        results["treatments_cf"] = T_cf
        results["outcomes_cf"] = Y_cf
        if X_cf is not None:
            results["covariates_cf"] = X_cf
        results["delta_T"] = delta_T
        results["t_interv"] = t_interv

    if normalize:
        results["treatments"] = min_max_scale(results["treatments"])
        results["outcomes"] = min_max_scale(results["outcomes"])
        if "covariates" in results:
            results["covariates"] = min_max_scale(results["covariates"])
        if "treatments_cf" in results:
            results["treatments_cf"] = min_max_scale(results["treatments_cf"])
        if "outcomes_cf" in results:
            results["outcomes_cf"] = min_max_scale(results["outcomes_cf"])
        if "covariates_cf" in results:
            results["covariates_cf"] = min_max_scale(results["covariates_cf"])

    return results
