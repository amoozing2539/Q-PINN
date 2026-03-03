"""
Classical Physics-Informed Neural Network (PINN) for Alfvén Wave Reconstruction
================================================================================

Architecture: MLP taking (z, t) coordinates → predicting fields + measurements.
Outputs: φ, Bx, By, Bz, Ex, Ey, Ez, dBx/dt, dBy/dt, dBz/dt (10 total)

Uses PennyLane's autograd interface for automatic differentiation (compatible
with the qpinn conda environment).

Usage:
    conda run -n qpinn python classical_pinn.py
"""

import os
import sys
import time
import json

import autograd
import autograd.numpy as np
from autograd import grad
import h5py

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Architecture
    "hidden_layers": [64, 64, 64, 64],
    "activation": "tanh",       # tanh, relu, sigmoid, sin
    "n_outputs": 10,            # φ, Bx, By, Bz, Ex, Ey, Ez, dBx_dt, dBy_dt, dBz_dt

    # Training
    "learning_rate": 1e-3,
    "n_epochs": 5000,
    "batch_size": 256,
    "lr_decay_rate": 0.5,
    "lr_decay_every": 1000,

    # Loss weights
    "w_data_bdot": 1.0,         # Weight for Bdot measurement loss
    "w_data_phi": 1.0,          # Weight for phi measurement loss
    "w_pde": 0.0,               # Weight for PDE loss (set >0 once equations are added)

    # Reproducibility
    "seed": 42,
}

# Output channel indices (for readability)
IDX_PHI = 0
IDX_BX, IDX_BY, IDX_BZ = 1, 2, 3
IDX_EX, IDX_EY, IDX_EZ = 4, 5, 6
IDX_DBX_DT, IDX_DBY_DT, IDX_DBZ_DT = 7, 8, 9


# ===========================================================================
# 1. ACTIVATION FUNCTIONS
# ===========================================================================

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0.0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sin_activation(x):
    return np.sin(x)

ACTIVATIONS = {
    "tanh": tanh,
    "relu": relu,
    "sigmoid": sigmoid,
    "sin": sin_activation,
}


# ===========================================================================
# 2. MODEL ARCHITECTURE
# ===========================================================================

def init_params(layer_sizes, seed=42):
    """
    Initialize MLP parameters using Xavier initialization.

    Parameters
    ----------
    layer_sizes : list[int]
        [input_dim, hidden1, hidden2, ..., output_dim]

    Returns
    -------
    params : list of (W, b) tuples
    """
    rng = np.random.RandomState(seed)
    params = []
    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        std = np.sqrt(2.0 / (n_in + n_out))
        W = rng.randn(n_in, n_out) * std
        b = np.zeros(n_out)
        params.append((W, b))
    return params


def forward(params, x, activation_fn):
    """
    MLP forward pass.

    Parameters
    ----------
    params : list of (W, b) tuples
    x : array, shape (N, 2)  — columns are [z, t]
    activation_fn : callable

    Returns
    -------
    out : array, shape (N, 10)
        Columns: [φ, Bx, By, Bz, Ex, Ey, Ez, dBx_dt, dBy_dt, dBz_dt]
    """
    h = x
    for W, b in params[:-1]:
        h = activation_fn(np.dot(h, W) + b)
    W_last, b_last = params[-1]
    out = np.dot(h, W_last) + b_last
    return out


def predict(params, z, t, activation_fn):
    """
    Convenience wrapper: predict all fields at given (z, t) points.

    Parameters
    ----------
    z : array (N,) — spatial coordinates
    t : array (N,) — time coordinates

    Returns
    -------
    dict with keys: phi, Bx, By, Bz, Ex, Ey, Ez, dBx_dt, dBy_dt, dBz_dt
    """
    x = np.column_stack([z, t])
    out = forward(params, x, activation_fn)
    return {
        "phi":     out[:, IDX_PHI],
        "Bx":      out[:, IDX_BX],
        "By":      out[:, IDX_BY],
        "Bz":      out[:, IDX_BZ],
        "Ex":      out[:, IDX_EX],
        "Ey":      out[:, IDX_EY],
        "Ez":      out[:, IDX_EZ],
        "dBx_dt":  out[:, IDX_DBX_DT],
        "dBy_dt":  out[:, IDX_DBY_DT],
        "dBz_dt":  out[:, IDX_DBZ_DT],
    }


# ===========================================================================
# 3. DERIVATIVE COMPUTATION (via autograd)
# ===========================================================================

def compute_derivatives(params, z, t, activation_fn):
    """
    Compute spatial and temporal derivatives of all predicted fields
    using autograd automatic differentiation.

    Returns a dict with keys like 'dBx_dz', 'dBx_dt_auto', 'dphi_dz', etc.
    These can be used in PDE residual computations.
    """
    # Create a function that maps (z_scalar, t_scalar) → output vector
    def f(z_val, t_val):
        x = np.array([[z_val, t_val]])
        return forward(params, x, activation_fn)[0]  # shape (10,)

    # Gradients w.r.t. z (input index 0) and t (input index 1)
    df_dz = grad(lambda z_val, t_val: f(z_val, t_val), argnum=0)
    df_dt = grad(lambda z_val, t_val: f(z_val, t_val), argnum=1)

    # Vectorize over all (z, t) points
    N = len(z)
    derivs = {
        "dBx_dz": np.zeros(N), "dBy_dz": np.zeros(N), "dBz_dz": np.zeros(N),
        "dBx_dt_auto": np.zeros(N), "dBy_dt_auto": np.zeros(N),
        "dEx_dz": np.zeros(N), "dEy_dz": np.zeros(N), "dEz_dz": np.zeros(N),
        "dphi_dz": np.zeros(N), "dphi_dt": np.zeros(N),
    }

    for i in range(N):
        # This is a scalar-by-scalar gradient; for large N consider batched approach
        grad_z_i = df_dz(z[i], t[i])  # shape (10,)
        grad_t_i = df_dt(z[i], t[i])

        derivs["dphi_dz"][i] = grad_z_i[IDX_PHI]
        derivs["dphi_dt"][i] = grad_t_i[IDX_PHI]
        derivs["dBx_dz"][i] = grad_z_i[IDX_BX]
        derivs["dBy_dz"][i] = grad_z_i[IDX_BY]
        derivs["dBz_dz"][i] = grad_z_i[IDX_BZ]
        derivs["dBx_dt_auto"][i] = grad_t_i[IDX_BX]
        derivs["dBy_dt_auto"][i] = grad_t_i[IDX_BY]
        derivs["dEx_dz"][i] = grad_z_i[IDX_EX]
        derivs["dEy_dz"][i] = grad_z_i[IDX_EY]
        derivs["dEz_dz"][i] = grad_z_i[IDX_EZ]

    return derivs


# ===========================================================================
# 4. LOSS FUNCTIONS
# ===========================================================================

def data_loss_bdot(params, z_probes, t_probes, dBx_dt_meas, dBy_dt_meas,
                   activation_fn):
    """
    MSE loss fitting the predicted dBx/dt, dBy/dt to sparse Bdot measurements.

    Parameters
    ----------
    z_probes : array (N_meas,) — probe z-coordinates (repeated for each time)
    t_probes : array (N_meas,) — corresponding time coordinates
    dBx_dt_meas : array (N_meas,) — measured dBx/dt values
    dBy_dt_meas : array (N_meas,) — measured dBy/dt values
    """
    x = np.column_stack([z_probes, t_probes])
    pred = forward(params, x, activation_fn)

    err_bx = pred[:, IDX_DBX_DT] - dBx_dt_meas
    err_by = pred[:, IDX_DBY_DT] - dBy_dt_meas

    return np.mean(err_bx ** 2 + err_by ** 2)


def data_loss_phi(params, z_probes, t_probes, phi_meas, activation_fn):
    """
    MSE loss fitting the predicted φ to sparse potential measurements.
    """
    x = np.column_stack([z_probes, t_probes])
    pred = forward(params, x, activation_fn)

    err = pred[:, IDX_PHI] - phi_meas
    return np.mean(err ** 2)


def pde_loss(params, z_colloc, t_colloc, activation_fn):
    """
    Physics-informed PDE residual loss.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  TODO: FILL IN YOUR PDE EQUATIONS HERE                             ║
    ║                                                                    ║
    ║  Available fields from forward(params, x, activation_fn):          ║
    ║    [:, 0] = φ          (electrostatic potential)                    ║
    ║    [:, 1] = Bx         (magnetic field x)                          ║
    ║    [:, 2] = By         (magnetic field y)                          ║
    ║    [:, 3] = Bz         (magnetic field z)                          ║
    ║    [:, 4] = Ex         (electric field x)                          ║
    ║    [:, 5] = Ey         (electric field y)                          ║
    ║    [:, 6] = Ez         (electric field z)                          ║
    ║    [:, 7] = dBx/dt     (time derivative of Bx)                     ║
    ║    [:, 8] = dBy/dt     (time derivative of By)                     ║
    ║    [:, 9] = dBz/dt     (time derivative of Bz)                     ║
    ║                                                                    ║
    ║  Use compute_derivatives() for spatial/temporal gradients:          ║
    ║    derivs = compute_derivatives(params, z, t, activation_fn)       ║
    ║    derivs["dBx_dz"], derivs["dEx_dz"], derivs["dphi_dz"], etc.     ║
    ║                                                                    ║
    ║  Example PDE residuals (Faraday's law in 1D):                      ║
    ║    ∂Bx/∂t + ∂Ey/∂z = 0                                            ║
    ║    ∂By/∂t - ∂Ex/∂z = 0                                             ║
    ║    E_z = -∂φ/∂z                                                    ║
    ║                                                                    ║
    ║  Return: scalar MSE of residuals                                   ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    # ---- PLACEHOLDER: return 0 until PDE equations are added ----
    return 0.0

    # ---- EXAMPLE (uncomment and modify): ----
    # x = np.column_stack([z_colloc, t_colloc])
    # pred = forward(params, x, activation_fn)
    # derivs = compute_derivatives(params, z_colloc, t_colloc, activation_fn)
    #
    # # Faraday's law residuals (1D)
    # res1 = pred[:, IDX_DBX_DT] + derivs["dEy_dz"]   # ∂Bx/∂t + ∂Ey/∂z = 0
    # res2 = pred[:, IDX_DBY_DT] - derivs["dEx_dz"]   # ∂By/∂t - ∂Ex/∂z = 0
    #
    # # Potential relation
    # res3 = pred[:, IDX_EZ] + derivs["dphi_dz"]       # Ez = -∂φ/∂z
    #
    # return np.mean(res1**2 + res2**2 + res3**2)


def total_loss(params, z_bdot, t_bdot, dBx_dt_meas, dBy_dt_meas,
               z_phi, t_phi, phi_meas,
               z_colloc, t_colloc,
               activation_fn, config):
    """
    Weighted sum of data losses and PDE loss.
    """
    L_bdot = data_loss_bdot(params, z_bdot, t_bdot, dBx_dt_meas, dBy_dt_meas,
                            activation_fn)
    L_phi = data_loss_phi(params, z_phi, t_phi, phi_meas, activation_fn)
    L_pde = pde_loss(params, z_colloc, t_colloc, activation_fn)

    return (config["w_data_bdot"] * L_bdot +
            config["w_data_phi"] * L_phi +
            config["w_pde"] * L_pde)


# ===========================================================================
# 5. TRAINING LOOP
# ===========================================================================

def train(params, train_data, config=None, verbose=True):
    """
    Train the classical PINN using Adam optimizer.

    Parameters
    ----------
    params : list of (W, b) tuples
    train_data : dict with keys:
        z_bdot, t_bdot, dBx_dt_meas, dBy_dt_meas,
        z_phi, t_phi, phi_meas,
        z_colloc, t_colloc
    config : dict (uses DEFAULT_CONFIG if None)

    Returns
    -------
    params : trained parameters
    history : dict with loss arrays
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    activation_fn = ACTIVATIONS[config["activation"]]
    lr = config["learning_rate"]
    n_epochs = config["n_epochs"]

    # Flatten params for autograd-compatible Adam
    flat_params, unflatten = _flatten_params(params)

    # Adam state
    m = np.zeros_like(flat_params)
    v = np.zeros_like(flat_params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Loss gradient w.r.t. flat params
    def loss_fn(flat_p):
        p = unflatten(flat_p)
        return total_loss(
            p,
            train_data["z_bdot"], train_data["t_bdot"],
            train_data["dBx_dt_meas"], train_data["dBy_dt_meas"],
            train_data["z_phi"], train_data["t_phi"], train_data["phi_meas"],
            train_data["z_colloc"], train_data["t_colloc"],
            activation_fn, config,
        )

    grad_fn = grad(loss_fn)

    history = {"loss": [], "loss_bdot": [], "loss_phi": [], "loss_pde": [],
               "epoch": [], "time": []}
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # Learning rate decay
        current_lr = lr * (config["lr_decay_rate"] **
                           (epoch // config["lr_decay_every"]))

        # Compute gradient
        g = grad_fn(flat_params)

        # Adam update
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** epoch)
        v_hat = v / (1 - beta2 ** epoch)
        flat_params = flat_params - current_lr * m_hat / (np.sqrt(v_hat) + eps)

        # Record losses
        if epoch % 50 == 0 or epoch == 1:
            p = unflatten(flat_params)
            loss_val = loss_fn(flat_params)
            l_bdot = data_loss_bdot(p, train_data["z_bdot"], train_data["t_bdot"],
                                    train_data["dBx_dt_meas"], train_data["dBy_dt_meas"],
                                    activation_fn)
            l_phi = data_loss_phi(p, train_data["z_phi"], train_data["t_phi"],
                                  train_data["phi_meas"], activation_fn)
            l_pde = pde_loss(p, train_data["z_colloc"], train_data["t_colloc"],
                             activation_fn)

            history["epoch"].append(epoch)
            history["loss"].append(float(loss_val))
            history["loss_bdot"].append(float(l_bdot))
            history["loss_phi"].append(float(l_phi))
            history["loss_pde"].append(float(l_pde))
            history["time"].append(time.time() - t0)

            if verbose and (epoch % 500 == 0 or epoch == 1):
                print(f"Epoch {epoch:5d} | Loss: {loss_val:.6e} | "
                      f"Bdot: {l_bdot:.6e} | Phi: {l_phi:.6e} | "
                      f"PDE: {l_pde:.6e} | LR: {current_lr:.2e}")

    params = unflatten(flat_params)
    return params, history


# ===========================================================================
# 6. EVALUATION FUNCTIONS
# ===========================================================================

def evaluate(params, z_test, t_test, ground_truth, activation_fn):
    """
    Evaluate model predictions against ground truth on a test grid.

    Parameters
    ----------
    z_test, t_test : arrays (N,)
    ground_truth : dict with keys matching predict() output
    activation_fn : callable

    Returns
    -------
    metrics : dict with RMSE and relative L2 error for each field
    predictions : dict of predicted field arrays
    """
    predictions = predict(params, z_test, t_test, activation_fn)

    metrics = {}
    for key in predictions:
        if key in ground_truth and ground_truth[key] is not None:
            pred_vals = predictions[key]
            true_vals = ground_truth[key]

            rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
            norm = np.sqrt(np.mean(true_vals ** 2))
            rel_l2 = rmse / (norm + 1e-12)

            metrics[key] = {"rmse": float(rmse), "rel_l2": float(rel_l2)}

    return metrics, predictions


def evaluate_on_sparse(params, train_data, activation_fn):
    """
    Evaluate on the training sparse measurements (Bdot + phi).
    Returns dict with RMSE for each measurement type.
    """
    # Bdot
    x_bdot = np.column_stack([train_data["z_bdot"], train_data["t_bdot"]])
    pred_bdot = forward(params, x_bdot, activation_fn)
    rmse_dBx = np.sqrt(np.mean((pred_bdot[:, IDX_DBX_DT] -
                                 train_data["dBx_dt_meas"]) ** 2))
    rmse_dBy = np.sqrt(np.mean((pred_bdot[:, IDX_DBY_DT] -
                                 train_data["dBy_dt_meas"]) ** 2))

    # Phi
    x_phi = np.column_stack([train_data["z_phi"], train_data["t_phi"]])
    pred_phi = forward(params, x_phi, activation_fn)
    rmse_phi = np.sqrt(np.mean((pred_phi[:, IDX_PHI] -
                                 train_data["phi_meas"]) ** 2))

    return {
        "rmse_dBx_dt": float(rmse_dBx),
        "rmse_dBy_dt": float(rmse_dBy),
        "rmse_phi": float(rmse_phi),
    }


# ===========================================================================
# 7. SAVE / LOAD
# ===========================================================================

def save_model(params, history, config, filepath):
    """Save model parameters and training history to HDF5."""
    with h5py.File(filepath, "w") as f:
        for i, (W, b) in enumerate(params):
            f.create_dataset(f"layer_{i}/W", data=W)
            f.create_dataset(f"layer_{i}/b", data=b)
        f.attrs["n_layers"] = len(params)
        f.attrs["config_json"] = json.dumps(config)

        grp = f.create_group("history")
        for key, val in history.items():
            grp.create_dataset(key, data=np.array(val))


def load_model(filepath):
    """Load model parameters and training history from HDF5."""
    with h5py.File(filepath, "r") as f:
        n_layers = f.attrs["n_layers"]
        config = json.loads(f.attrs["config_json"])

        params = []
        for i in range(n_layers):
            W = f[f"layer_{i}/W"][:]
            b = f[f"layer_{i}/b"][:]
            params.append((W, b))

        history = {}
        if "history" in f:
            for key in f["history"]:
                history[key] = f["history"][key][:].tolist()

    return params, history, config


# ===========================================================================
# HELPERS
# ===========================================================================

def _flatten_params(params):
    """Flatten list of (W, b) tuples into a single 1D array."""
    shapes = []
    flat_list = []
    for W, b in params:
        shapes.append(("W", W.shape))
        shapes.append(("b", b.shape))
        flat_list.append(W.ravel())
        flat_list.append(b.ravel())
    flat = np.concatenate(flat_list)

    def unflatten(flat_array):
        result = []
        idx = 0
        for i in range(0, len(shapes), 2):
            w_shape = shapes[i][1]
            b_shape = shapes[i + 1][1]
            w_size = int(np.prod(w_shape))
            b_size = int(np.prod(b_shape))
            W = flat_array[idx:idx + w_size].reshape(w_shape)
            idx += w_size
            b = flat_array[idx:idx + b_size].reshape(b_shape)
            idx += b_size
            result.append((W, b))
        return result

    return flat, unflatten


def prepare_training_data(dataset_path):
    """
    Load the Alfvén wave dataset and prepare training arrays.

    Returns
    -------
    train_data : dict ready for train()
    norm : dict of normalization constants
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
    from load_dataset import load_full_dataset

    data = load_full_dataset(dataset_path)
    norm = data.get("normalization", {})

    # --- Sparse Bdot measurements ---
    bdot = data["sparse_bdot"]
    n_probes_b, n_times_b = bdot["dBx_dt"].shape
    z_probes_b = bdot["probe_positions"]  # (n_probes,)
    t_b = bdot["t_coords"]                # (n_times,)

    # Create meshgrid of (z, t) for all probe × time combinations
    Z_b, T_b = np.meshgrid(z_probes_b, t_b, indexing="ij")
    z_bdot_flat = Z_b.ravel()
    t_bdot_flat = T_b.ravel()
    dBx_dt_flat = bdot["dBx_dt"].ravel()
    dBy_dt_flat = bdot["dBy_dt"].ravel()

    # --- Sparse phi measurements ---
    phi = data["sparse_phi"]
    n_probes_p, n_times_p = phi["phi"].shape
    z_probes_p = phi["probe_positions"]
    t_p = phi["t_coords"]

    Z_p, T_p = np.meshgrid(z_probes_p, t_p, indexing="ij")
    z_phi_flat = Z_p.ravel()
    t_phi_flat = T_p.ravel()
    phi_flat = phi["phi"].ravel()

    # --- Collocation points for PDE loss ---
    # Random points in the (z, t) domain
    rng = np.random.RandomState(42)
    Lz = float(norm.get("Lz", z_probes_b.max()))
    T_max = float(t_b.max())
    n_colloc = 2000
    z_colloc = rng.uniform(0, Lz, n_colloc)
    t_colloc = rng.uniform(0, T_max, n_colloc)

    # --- Normalize inputs to [0, 1] for better training ---
    z_scale = Lz
    t_scale = T_max

    train_data = {
        "z_bdot": z_bdot_flat / z_scale,
        "t_bdot": t_bdot_flat / t_scale,
        "dBx_dt_meas": dBx_dt_flat,
        "dBy_dt_meas": dBy_dt_flat,
        "z_phi": z_phi_flat / z_scale,
        "t_phi": t_phi_flat / t_scale,
        "phi_meas": phi_flat,
        "z_colloc": z_colloc / z_scale,
        "t_colloc": t_colloc / t_scale,
        "z_scale": z_scale,
        "t_scale": t_scale,
    }

    return train_data, norm


# ===========================================================================
# 8. MAIN — Example usage
# ===========================================================================

if __name__ == "__main__":
    # Path to dataset
    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "simulation", "alfven_wave_dataset.h5"
    )

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Run the simulation first: cd simulation && conda run -n warpx_env python run_alfven_sim.py")
        sys.exit(1)

    print("=" * 60)
    print("Classical PINN — Alfvén Wave Field Reconstruction")
    print("=" * 60)

    # Load data
    print("\nLoading dataset ...")
    train_data, norm = prepare_training_data(dataset_path)
    print(f"  Bdot points:  {len(train_data['z_bdot'])}")
    print(f"  Phi points:   {len(train_data['z_phi'])}")
    print(f"  Collocation:  {len(train_data['z_colloc'])}")

    # Build model
    config = DEFAULT_CONFIG.copy()
    layers = [2] + config["hidden_layers"] + [config["n_outputs"]]
    params = init_params(layers, seed=config["seed"])
    n_params = sum(W.size + b.size for W, b in params)
    print(f"\nModel: {layers}")
    print(f"Total parameters: {n_params}")

    # Train
    print("\nTraining ...")
    params, history = train(params, train_data, config, verbose=True)

    # Evaluate on sparse measurements
    activation_fn = ACTIVATIONS[config["activation"]]
    sparse_metrics = evaluate_on_sparse(params, train_data, activation_fn)
    print(f"\nSparse measurement RMSE:")
    for k, v in sparse_metrics.items():
        print(f"  {k}: {v:.6e}")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "classical_pinn_trained.h5")
    save_model(params, history, config, save_path)
    print(f"\nModel saved to: {save_path}")
