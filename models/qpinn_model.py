"""
Quantum Physics-Informed Neural Network (Q-PINN) for Alfvén Wave Reconstruction
================================================================================

Hybrid quantum-classical architecture:
    Input (z, t) → Classical Pre-Net → PQC (Angle Embed + Ansatz) → Measurement → Classical Post-Net → Output

Outputs: φ, Bx, By, Bz, Ex, Ey, Ez, dBx/dt, dBy/dt, dBz/dt (10 total)

Uses PennyLane with lightning.gpu backend for GPU-accelerated quantum simulation.

Usage:
    conda run -n qpinn python qpinn_model.py
"""

import os
import sys
import time
import json

import autograd
import autograd.numpy as anp
import numpy as np
import pennylane as qml
from autograd import grad
import h5py

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Quantum circuit
    "n_qubits": 7,              # Number of qubits in the PQC
    "n_layers": 4,              # Number of ansatz layers
    "q_device": "lightning.qubit",  # "lightning.gpu" for GPU, "lightning.qubit" for CPU

    # Classical layers
    "pre_net_hidden": [16],     # Pre-net hidden layers (input → qubits)
    "post_net_hidden": [16],    # Post-net hidden layers (qubits → output)
    "n_outputs": 10,            # φ, Bx, By, Bz, Ex, Ey, Ez, dBx_dt, dBy_dt, dBz_dt
    "activation": "tanh",

    # Training
    "learning_rate": 1e-3,
    "n_epochs": 5000,
    "batch_size": 256,
    "lr_decay_rate": 0.5,
    "lr_decay_every": 1000,

    # Loss weights
    "w_data_bdot": 1.0,
    "w_data_phi": 1.0,
    "w_pde": 0.0,               # Set >0 once PDE equations are filled in

    # Reproducibility
    "seed": 42,
}

# Output channel indices
IDX_PHI = 0
IDX_BX, IDX_BY, IDX_BZ = 1, 2, 3
IDX_EX, IDX_EY, IDX_EZ = 4, 5, 6
IDX_DBX_DT, IDX_DBY_DT, IDX_DBZ_DT = 7, 8, 9


# ===========================================================================
# 1. ACTIVATION FUNCTIONS
# ===========================================================================

def tanh(x):
    return anp.tanh(x)

def relu(x):
    return anp.maximum(0.0, x)

def sigmoid(x):
    return 1.0 / (1.0 + anp.exp(-x))

ACTIVATIONS = {
    "tanh": tanh,
    "relu": relu,
    "sigmoid": sigmoid,
}


# ===========================================================================
# 2. QUANTUM CIRCUIT
# ===========================================================================

def create_qnode(n_qubits, n_layers, device_name="lightning.qubit"):
    """
    Create a PennyLane QNode for the parameterized quantum circuit.

    Architecture per layer:
        1. RY(θ) rotation on each qubit
        2. RZ(θ) rotation on each qubit
        3. Circular CNOT entanglement (qubit_i → qubit_{i+1})

    Measurement: Pauli-Z expectation on each qubit.
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="adjoint")
    def circuit(inputs, weights_ry, weights_rz):
        """
        Parameters
        ----------
        inputs : array (n_qubits,) — angle-embedded classical features
        weights_ry : array (n_layers, n_qubits) — RY rotation angles
        weights_rz : array (n_layers, n_qubits) — RZ rotation angles
        """
        # Angle embedding: encode classical input into qubit rotations
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational ansatz layers
        for layer in range(n_layers):
            # Parameterized rotations
            for i in range(n_qubits):
                qml.RY(weights_ry[layer, i], wires=i)
                qml.RZ(weights_rz[layer, i], wires=i)

            # Circular CNOT entanglement
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Measure Pauli-Z expectation on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ===========================================================================
# 3. MODEL ARCHITECTURE (Hybrid Quantum-Classical)
# ===========================================================================

def init_params(config, seed=42):
    """
    Initialize all parameters for the hybrid Q-PINN model.

    Returns
    -------
    params : dict with keys:
        "pre_net"   : list of (W, b) for pre-processing network
        "weights_ry": array (n_layers, n_qubits)
        "weights_rz": array (n_layers, n_qubits)
        "post_net"  : list of (W, b) for post-processing network
    """
    rng = np.random.RandomState(seed)
    n_qubits = config["n_qubits"]
    n_layers = config["n_layers"]
    n_outputs = config["n_outputs"]

    # --- Pre-net: 2 → ... → n_qubits ---
    pre_sizes = [2] + config["pre_net_hidden"] + [n_qubits]
    pre_net = _init_mlp(pre_sizes, rng)

    # --- Quantum circuit weights ---
    weights_ry = rng.randn(n_layers, n_qubits) * 0.1
    weights_rz = rng.randn(n_layers, n_qubits) * 0.1

    # --- Post-net: n_qubits → ... → n_outputs ---
    post_sizes = [n_qubits] + config["post_net_hidden"] + [n_outputs]
    post_net = _init_mlp(post_sizes, rng)

    return {
        "pre_net": pre_net,
        "weights_ry": weights_ry,
        "weights_rz": weights_rz,
        "post_net": post_net,
    }


def _init_mlp(layer_sizes, rng):
    """Xavier-initialized MLP parameters."""
    params = []
    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        std = np.sqrt(2.0 / (n_in + n_out))
        W = rng.randn(n_in, n_out) * std
        b = np.zeros(n_out)
        params.append((W, b))
    return params


def _mlp_forward(params, x, activation_fn):
    """Forward pass through a small classical MLP."""
    h = x
    for W, b in params[:-1]:
        h = activation_fn(anp.dot(h, W) + b)
    W_last, b_last = params[-1]
    return anp.dot(h, W_last) + b_last


def forward_single(params, x_single, circuit, activation_fn):
    """
    Forward pass for a SINGLE input point (z, t).

    Flow: (z,t) → pre_net → angle_embed → PQC → measurement → post_net → output

    Parameters
    ----------
    params : dict with pre_net, weights_ry, weights_rz, post_net
    x_single : array (2,) — [z, t]
    circuit : PennyLane QNode
    activation_fn : callable

    Returns
    -------
    out : array (10,) — [φ, Bx, By, Bz, Ex, Ey, Ez, dBx_dt, dBy_dt, dBz_dt]
    """
    # Pre-net: (2,) → (n_qubits,)
    x_2d = x_single.reshape(1, -1)
    pre_out = _mlp_forward(params["pre_net"], x_2d, activation_fn)[0]

    # Scale to [0, π] for angle embedding
    pre_out_scaled = anp.pi * sigmoid(pre_out)

    # Quantum circuit: (n_qubits,) → (n_qubits,) expectation values
    q_out = anp.array(circuit(pre_out_scaled,
                               params["weights_ry"],
                               params["weights_rz"]))

    # Post-net: (n_qubits,) → (n_outputs,)
    q_out_2d = q_out.reshape(1, -1)
    out = _mlp_forward(params["post_net"], q_out_2d, activation_fn)[0]

    return out


def forward(params, x, circuit, activation_fn):
    """
    Forward pass for a BATCH of input points.

    Parameters
    ----------
    x : array (N, 2) — columns [z, t]

    Returns
    -------
    out : array (N, 10)
    """
    N = x.shape[0]
    outputs = []
    for i in range(N):
        out_i = forward_single(params, x[i], circuit, activation_fn)
        outputs.append(out_i)
    return anp.stack(outputs)


def predict(params, z, t, circuit, activation_fn):
    """
    Convenience wrapper: predict all fields at given (z, t) points.

    Returns
    -------
    dict with keys: phi, Bx, By, Bz, Ex, Ey, Ez, dBx_dt, dBy_dt, dBz_dt
    """
    x = anp.column_stack([z, t])
    out = forward(params, x, circuit, activation_fn)
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
# 4. DERIVATIVE COMPUTATION (via autograd)
# ===========================================================================

def compute_derivatives(params, z, t, circuit, activation_fn):
    """
    Compute spatial and temporal derivatives of all predicted fields
    using autograd automatic differentiation.

    Returns a dict with keys like 'dBx_dz', 'dBx_dt_auto', 'dphi_dz', etc.
    """
    def f(z_val, t_val):
        x = anp.array([z_val, t_val])
        return forward_single(params, x, circuit, activation_fn)

    df_dz = grad(lambda z_val, t_val: f(z_val, t_val), argnum=0)
    df_dt = grad(lambda z_val, t_val: f(z_val, t_val), argnum=1)

    N = len(z)
    derivs = {
        "dBx_dz": np.zeros(N), "dBy_dz": np.zeros(N), "dBz_dz": np.zeros(N),
        "dBx_dt_auto": np.zeros(N), "dBy_dt_auto": np.zeros(N),
        "dEx_dz": np.zeros(N), "dEy_dz": np.zeros(N), "dEz_dz": np.zeros(N),
        "dphi_dz": np.zeros(N), "dphi_dt": np.zeros(N),
    }

    for i in range(N):
        grad_z_i = df_dz(z[i], t[i])
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
# 5. LOSS FUNCTIONS
# ===========================================================================

def data_loss_bdot(params, z_probes, t_probes, dBx_dt_meas, dBy_dt_meas,
                   circuit, activation_fn):
    """
    MSE loss fitting the predicted dBx/dt, dBy/dt to sparse Bdot measurements.
    """
    x = anp.column_stack([z_probes, t_probes])
    pred = forward(params, x, circuit, activation_fn)

    err_bx = pred[:, IDX_DBX_DT] - dBx_dt_meas
    err_by = pred[:, IDX_DBY_DT] - dBy_dt_meas

    return anp.mean(err_bx ** 2 + err_by ** 2)


def data_loss_phi(params, z_probes, t_probes, phi_meas, circuit, activation_fn):
    """
    MSE loss fitting the predicted φ to sparse potential measurements.
    """
    x = anp.column_stack([z_probes, t_probes])
    pred = forward(params, x, circuit, activation_fn)

    err = pred[:, IDX_PHI] - phi_meas
    return anp.mean(err ** 2)


def pde_loss(params, z_colloc, t_colloc, circuit, activation_fn):
    """
    Physics-informed PDE residual loss.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  TODO: FILL IN YOUR PDE EQUATIONS HERE                             ║
    ║                                                                    ║
    ║  Available fields from forward(params, x, circuit, activation_fn): ║
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
    ║    derivs = compute_derivatives(params, z, t, circuit, act_fn)     ║
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
    # x = anp.column_stack([z_colloc, t_colloc])
    # pred = forward(params, x, circuit, activation_fn)
    # derivs = compute_derivatives(params, z_colloc, t_colloc, circuit, activation_fn)
    #
    # # Faraday's law residuals (1D)
    # res1 = pred[:, IDX_DBX_DT] + derivs["dEy_dz"]
    # res2 = pred[:, IDX_DBY_DT] - derivs["dEx_dz"]
    #
    # # Potential relation
    # res3 = pred[:, IDX_EZ] + derivs["dphi_dz"]
    #
    # return anp.mean(res1**2 + res2**2 + res3**2)


def total_loss(params, z_bdot, t_bdot, dBx_dt_meas, dBy_dt_meas,
               z_phi, t_phi, phi_meas,
               z_colloc, t_colloc,
               circuit, activation_fn, config):
    """
    Weighted sum of data losses and PDE loss.
    """
    L_bdot = data_loss_bdot(params, z_bdot, t_bdot, dBx_dt_meas, dBy_dt_meas,
                            circuit, activation_fn)
    L_phi = data_loss_phi(params, z_phi, t_phi, phi_meas, circuit, activation_fn)
    L_pde = pde_loss(params, z_colloc, t_colloc, circuit, activation_fn)

    return (config["w_data_bdot"] * L_bdot +
            config["w_data_phi"] * L_phi +
            config["w_pde"] * L_pde)


# ===========================================================================
# 6. TRAINING LOOP
# ===========================================================================

def _flatten_params(params):
    """
    Flatten the hybrid params dict into a single 1D array for optimization.
    """
    flat_list = []
    shapes_info = []

    # Pre-net
    for i, (W, b) in enumerate(params["pre_net"]):
        shapes_info.append(("pre_net", i, "W", W.shape))
        shapes_info.append(("pre_net", i, "b", b.shape))
        flat_list.append(W.ravel())
        flat_list.append(b.ravel())

    # Quantum weights
    shapes_info.append(("weights_ry", None, None, params["weights_ry"].shape))
    flat_list.append(params["weights_ry"].ravel())
    shapes_info.append(("weights_rz", None, None, params["weights_rz"].shape))
    flat_list.append(params["weights_rz"].ravel())

    # Post-net
    for i, (W, b) in enumerate(params["post_net"]):
        shapes_info.append(("post_net", i, "W", W.shape))
        shapes_info.append(("post_net", i, "b", b.shape))
        flat_list.append(W.ravel())
        flat_list.append(b.ravel())

    flat = anp.concatenate(flat_list)

    def unflatten(flat_array):
        result = {"pre_net": [], "post_net": []}
        idx = 0
        for info in shapes_info:
            name, layer_idx, wb, shape = info
            size = int(np.prod(shape))
            arr = flat_array[idx:idx + size].reshape(shape)
            idx += size

            if name in ("pre_net", "post_net"):
                if wb == "W":
                    current_W = arr
                else:  # "b"
                    result[name].append((current_W, arr))
            elif name == "weights_ry":
                result["weights_ry"] = arr
            elif name == "weights_rz":
                result["weights_rz"] = arr

        return result

    return flat, unflatten


def train(params, train_data, circuit, config=None, verbose=True):
    """
    Train the Q-PINN using Adam optimizer.

    Parameters
    ----------
    params : dict (pre_net, weights_ry, weights_rz, post_net)
    train_data : dict (from prepare_training_data)
    circuit : PennyLane QNode
    config : dict

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

    # Flatten all params for Adam
    flat_params, unflatten = _flatten_params(params)

    # Adam state
    m = anp.zeros_like(flat_params)
    v = anp.zeros_like(flat_params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def loss_fn(flat_p):
        p = unflatten(flat_p)
        return total_loss(
            p,
            train_data["z_bdot"], train_data["t_bdot"],
            train_data["dBx_dt_meas"], train_data["dBy_dt_meas"],
            train_data["z_phi"], train_data["t_phi"], train_data["phi_meas"],
            train_data["z_colloc"], train_data["t_colloc"],
            circuit, activation_fn, config,
        )

    grad_fn = grad(loss_fn)

    history = {"loss": [], "loss_bdot": [], "loss_phi": [], "loss_pde": [],
               "epoch": [], "time": []}
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        current_lr = lr * (config["lr_decay_rate"] **
                           (epoch // config["lr_decay_every"]))

        g = grad_fn(flat_params)

        # Adam update
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** epoch)
        v_hat = v / (1 - beta2 ** epoch)
        flat_params = flat_params - current_lr * m_hat / (anp.sqrt(v_hat) + eps)

        # Log
        if epoch % 50 == 0 or epoch == 1:
            p = unflatten(flat_params)
            loss_val = loss_fn(flat_params)
            l_bdot = data_loss_bdot(p, train_data["z_bdot"], train_data["t_bdot"],
                                    train_data["dBx_dt_meas"], train_data["dBy_dt_meas"],
                                    circuit, activation_fn)
            l_phi = data_loss_phi(p, train_data["z_phi"], train_data["t_phi"],
                                  train_data["phi_meas"], circuit, activation_fn)
            l_pde = pde_loss(p, train_data["z_colloc"], train_data["t_colloc"],
                             circuit, activation_fn)

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
# 7. EVALUATION FUNCTIONS
# ===========================================================================

def evaluate(params, z_test, t_test, ground_truth, circuit, activation_fn):
    """
    Evaluate model predictions against ground truth on a test grid.

    Returns
    -------
    metrics : dict with RMSE and relative L2 error for each field
    predictions : dict of predicted field arrays
    """
    predictions = predict(params, z_test, t_test, circuit, activation_fn)

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


def evaluate_on_sparse(params, train_data, circuit, activation_fn):
    """
    Evaluate on the training sparse measurements.
    """
    x_bdot = anp.column_stack([train_data["z_bdot"], train_data["t_bdot"]])
    pred_bdot = forward(params, x_bdot, circuit, activation_fn)
    rmse_dBx = np.sqrt(np.mean((pred_bdot[:, IDX_DBX_DT] -
                                 train_data["dBx_dt_meas"]) ** 2))
    rmse_dBy = np.sqrt(np.mean((pred_bdot[:, IDX_DBY_DT] -
                                 train_data["dBy_dt_meas"]) ** 2))

    x_phi = anp.column_stack([train_data["z_phi"], train_data["t_phi"]])
    pred_phi = forward(params, x_phi, circuit, activation_fn)
    rmse_phi = np.sqrt(np.mean((pred_phi[:, IDX_PHI] -
                                 train_data["phi_meas"]) ** 2))

    return {
        "rmse_dBx_dt": float(rmse_dBx),
        "rmse_dBy_dt": float(rmse_dBy),
        "rmse_phi": float(rmse_phi),
    }


def count_parameters(params):
    """Count the total number of trainable parameters."""
    total = 0
    for W, b in params["pre_net"]:
        total += W.size + b.size
    total += params["weights_ry"].size
    total += params["weights_rz"].size
    for W, b in params["post_net"]:
        total += W.size + b.size
    return total


# ===========================================================================
# 8. SAVE / LOAD
# ===========================================================================

def save_model(params, history, config, filepath):
    """Save Q-PINN parameters and training history to HDF5."""
    with h5py.File(filepath, "w") as f:
        # Pre-net
        for i, (W, b) in enumerate(params["pre_net"]):
            f.create_dataset(f"pre_net/layer_{i}/W", data=W)
            f.create_dataset(f"pre_net/layer_{i}/b", data=b)
        f.attrs["n_pre_layers"] = len(params["pre_net"])

        # Quantum weights
        f.create_dataset("weights_ry", data=params["weights_ry"])
        f.create_dataset("weights_rz", data=params["weights_rz"])

        # Post-net
        for i, (W, b) in enumerate(params["post_net"]):
            f.create_dataset(f"post_net/layer_{i}/W", data=W)
            f.create_dataset(f"post_net/layer_{i}/b", data=b)
        f.attrs["n_post_layers"] = len(params["post_net"])

        f.attrs["config_json"] = json.dumps(config)

        grp = f.create_group("history")
        for key, val in history.items():
            grp.create_dataset(key, data=np.array(val))


def load_model(filepath):
    """Load Q-PINN parameters and training history from HDF5."""
    with h5py.File(filepath, "r") as f:
        config = json.loads(f.attrs["config_json"])

        params = {"pre_net": [], "post_net": []}

        for i in range(f.attrs["n_pre_layers"]):
            W = f[f"pre_net/layer_{i}/W"][:]
            b = f[f"pre_net/layer_{i}/b"][:]
            params["pre_net"].append((W, b))

        params["weights_ry"] = f["weights_ry"][:]
        params["weights_rz"] = f["weights_rz"][:]

        for i in range(f.attrs["n_post_layers"]):
            W = f[f"post_net/layer_{i}/W"][:]
            b = f[f"post_net/layer_{i}/b"][:]
            params["post_net"].append((W, b))

        history = {}
        if "history" in f:
            for key in f["history"]:
                history[key] = f["history"][key][:].tolist()

    return params, history, config


# ===========================================================================
# HELPERS
# ===========================================================================

def prepare_training_data(dataset_path):
    """
    Load the Alfvén wave dataset and prepare training arrays.
    (Identical to classical_pinn.prepare_training_data)
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
    from load_dataset import load_full_dataset

    data = load_full_dataset(dataset_path)
    norm = data.get("normalization", {})

    # Sparse Bdot
    bdot = data["sparse_bdot"]
    n_probes_b, n_times_b = bdot["dBx_dt"].shape
    z_probes_b = bdot["probe_positions"]
    t_b = bdot["t_coords"]
    Z_b, T_b = np.meshgrid(z_probes_b, t_b, indexing="ij")

    # Sparse phi
    phi = data["sparse_phi"]
    n_probes_p, n_times_p = phi["phi"].shape
    z_probes_p = phi["probe_positions"]
    t_p = phi["t_coords"]
    Z_p, T_p = np.meshgrid(z_probes_p, t_p, indexing="ij")

    # Collocation
    rng = np.random.RandomState(42)
    Lz = float(norm.get("Lz", z_probes_b.max()))
    T_max = float(t_b.max())
    n_colloc = 2000
    z_colloc = rng.uniform(0, Lz, n_colloc)
    t_colloc = rng.uniform(0, T_max, n_colloc)

    z_scale = Lz
    t_scale = T_max

    train_data = {
        "z_bdot": Z_b.ravel() / z_scale,
        "t_bdot": T_b.ravel() / t_scale,
        "dBx_dt_meas": bdot["dBx_dt"].ravel(),
        "dBy_dt_meas": bdot["dBy_dt"].ravel(),
        "z_phi": Z_p.ravel() / z_scale,
        "t_phi": T_p.ravel() / t_scale,
        "phi_meas": phi["phi"].ravel(),
        "z_colloc": z_colloc / z_scale,
        "t_colloc": t_colloc / t_scale,
        "z_scale": z_scale,
        "t_scale": t_scale,
    }

    return train_data, norm


# ===========================================================================
# 9. MAIN — Example usage
# ===========================================================================

if __name__ == "__main__":
    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "simulation", "alfven_wave_dataset.h5"
    )

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    print("=" * 60)
    print("Q-PINN — Alfvén Wave Field Reconstruction")
    print("=" * 60)

    # Load data
    print("\nLoading dataset ...")
    train_data, norm = prepare_training_data(dataset_path)
    print(f"  Bdot points:  {len(train_data['z_bdot'])}")
    print(f"  Phi points:   {len(train_data['z_phi'])}")
    print(f"  Collocation:  {len(train_data['z_colloc'])}")

    # Build model
    config = DEFAULT_CONFIG.copy()
    params = init_params(config, seed=config["seed"])
    n_params = count_parameters(params)
    print(f"\nQ-PINN config:")
    print(f"  Qubits: {config['n_qubits']},  Layers: {config['n_layers']}")
    print(f"  Pre-net: 2 → {config['pre_net_hidden']} → {config['n_qubits']}")
    print(f"  Post-net: {config['n_qubits']} → {config['post_net_hidden']} → {config['n_outputs']}")
    print(f"  Total parameters: {n_params}")

    # Create quantum circuit
    circuit = create_qnode(config["n_qubits"], config["n_layers"],
                           config["q_device"])

    # Train
    print("\nTraining ...")
    params, history = train(params, train_data, circuit, config, verbose=True)

    # Evaluate
    activation_fn = ACTIVATIONS[config["activation"]]
    sparse_metrics = evaluate_on_sparse(params, train_data, circuit, activation_fn)
    print(f"\nSparse measurement RMSE:")
    for k, v in sparse_metrics.items():
        print(f"  {k}: {v:.6e}")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "qpinn_trained.h5")
    save_model(params, history, config, save_path)
    print(f"\nModel saved to: {save_path}")
