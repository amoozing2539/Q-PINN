"""
Convergence Analysis: Classical PINN vs Q-PINN
===============================================

Trains both models on the same Alfvén wave dataset and compares:
  - Training loss convergence curves
  - Per-component RMSE on sparse measurements
  - Total trainable parameters
  - Wall-clock training time

Produces a summary JSON and optional matplotlib plots.

Usage:
    conda run -n qpinn python convergence_analysis.py [--epochs 2000] [--plot]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import h5py

# Import both models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classical_pinn as cpinn
import qpinn_model as qpinn


# ===========================================================================
# Run convergence comparison
# ===========================================================================

def run_comparison(dataset_path, n_epochs=2000, save_dir=None, make_plot=False):
    """
    Train both models and compare convergence.

    Parameters
    ----------
    dataset_path : str — path to alfven_wave_dataset.h5
    n_epochs : int — number of training epochs for each model
    save_dir : str — directory to save results (default: models/)
    make_plot : bool — whether to generate matplotlib convergence plot
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------
    # 1. Load data (shared between both models)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Convergence Analysis: Classical PINN vs Q-PINN")
    print("=" * 70)

    print("\nLoading dataset ...")
    train_data_c, norm = cpinn.prepare_training_data(dataset_path)
    train_data_q, _ = qpinn.prepare_training_data(dataset_path)

    n_bdot = len(train_data_c["z_bdot"])
    n_phi = len(train_data_c["z_phi"])
    print(f"  Bdot measurement points: {n_bdot}")
    print(f"  Phi measurement points:  {n_phi}")

    # ------------------------------------------------------------------
    # 2. Classical PINN
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Training Classical PINN ...")
    print("-" * 70)

    config_c = cpinn.DEFAULT_CONFIG.copy()
    config_c["n_epochs"] = n_epochs
    layers = [2] + config_c["hidden_layers"] + [config_c["n_outputs"]]
    params_c = cpinn.init_params(layers, seed=config_c["seed"])
    n_params_c = sum(W.size + b.size for W, b in params_c)
    print(f"  Architecture: {layers}")
    print(f"  Parameters: {n_params_c}")

    t0 = time.time()
    params_c, hist_c = cpinn.train(params_c, train_data_c, config_c, verbose=True)
    time_c = time.time() - t0

    act_c = cpinn.ACTIVATIONS[config_c["activation"]]
    metrics_c = cpinn.evaluate_on_sparse(params_c, train_data_c, act_c)
    print(f"\n  Training time: {time_c:.1f}s")
    print(f"  Final sparse RMSE: {metrics_c}")

    # ------------------------------------------------------------------
    # 3. Q-PINN
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Training Q-PINN ...")
    print("-" * 70)

    config_q = qpinn.DEFAULT_CONFIG.copy()
    config_q["n_epochs"] = n_epochs
    params_q = qpinn.init_params(config_q, seed=config_q["seed"])
    n_params_q = qpinn.count_parameters(params_q)
    print(f"  Qubits: {config_q['n_qubits']}, Layers: {config_q['n_layers']}")
    print(f"  Parameters: {n_params_q}")

    circuit = qpinn.create_qnode(config_q["n_qubits"], config_q["n_layers"],
                                  config_q["q_device"])

    t0 = time.time()
    params_q, hist_q = qpinn.train(params_q, train_data_q, circuit, config_q,
                                    verbose=True)
    time_q = time.time() - t0

    act_q = qpinn.ACTIVATIONS[config_q["activation"]]
    metrics_q = qpinn.evaluate_on_sparse(params_q, train_data_q, circuit, act_q)
    print(f"\n  Training time: {time_q:.1f}s")
    print(f"  Final sparse RMSE: {metrics_q}")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONVERGENCE COMPARISON SUMMARY")
    print("=" * 70)

    summary = {
        "classical_pinn": {
            "n_parameters": n_params_c,
            "architecture": str(layers),
            "training_time_s": round(time_c, 2),
            "final_loss": hist_c["loss"][-1] if hist_c["loss"] else None,
            "final_rmse": metrics_c,
            "loss_history": hist_c["loss"],
            "epoch_history": hist_c["epoch"],
            "time_history": hist_c["time"],
            "loss_bdot_history": hist_c["loss_bdot"],
            "loss_phi_history": hist_c["loss_phi"],
        },
        "qpinn": {
            "n_parameters": n_params_q,
            "architecture": f"qubits={config_q['n_qubits']}, layers={config_q['n_layers']}",
            "training_time_s": round(time_q, 2),
            "final_loss": hist_q["loss"][-1] if hist_q["loss"] else None,
            "final_rmse": metrics_q,
            "loss_history": hist_q["loss"],
            "epoch_history": hist_q["epoch"],
            "time_history": hist_q["time"],
            "loss_bdot_history": hist_q["loss_bdot"],
            "loss_phi_history": hist_q["loss_phi"],
        },
        "n_epochs": n_epochs,
    }

    # Print table
    print(f"\n{'Metric':<30s} {'Classical PINN':>18s} {'Q-PINN':>18s}")
    print("-" * 70)
    print(f"{'Parameters':<30s} {n_params_c:>18d} {n_params_q:>18d}")
    print(f"{'Training time (s)':<30s} {time_c:>18.1f} {time_q:>18.1f}")
    if hist_c["loss"] and hist_q["loss"]:
        print(f"{'Final total loss':<30s} {hist_c['loss'][-1]:>18.6e} {hist_q['loss'][-1]:>18.6e}")
    print(f"{'RMSE dBx/dt':<30s} {metrics_c['rmse_dBx_dt']:>18.6e} {metrics_q['rmse_dBx_dt']:>18.6e}")
    print(f"{'RMSE dBy/dt':<30s} {metrics_c['rmse_dBy_dt']:>18.6e} {metrics_q['rmse_dBy_dt']:>18.6e}")
    print(f"{'RMSE phi':<30s} {metrics_c['rmse_phi']:>18.6e} {metrics_q['rmse_phi']:>18.6e}")

    # Save summary
    summary_path = os.path.join(save_dir, "convergence_summary.json")
    # Convert for JSON serialization
    summary_json = json.loads(json.dumps(summary, default=lambda x: float(x) if isinstance(x, np.floating) else x))
    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Save models
    cpinn.save_model(params_c, hist_c, config_c,
                     os.path.join(save_dir, "classical_pinn_trained.h5"))
    qpinn.save_model(params_q, hist_q, config_q,
                     os.path.join(save_dir, "qpinn_trained.h5"))

    # ------------------------------------------------------------------
    # 5. Plot (optional)
    # ------------------------------------------------------------------
    if make_plot:
        plot_convergence(summary, save_dir)

    return summary


def plot_convergence(summary, save_dir):
    """Generate convergence comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    c = summary["classical_pinn"]
    q = summary["qpinn"]

    # Total loss
    axes[0].semilogy(c["epoch_history"], c["loss_history"], 'b-', label=f'PINN ({c["n_parameters"]} params)')
    axes[0].semilogy(q["epoch_history"], q["loss_history"], 'r-', label=f'Q-PINN ({q["n_parameters"]} params)')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Training Loss Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bdot loss
    axes[1].semilogy(c["epoch_history"], c["loss_bdot_history"], 'b-', label='PINN')
    axes[1].semilogy(q["epoch_history"], q["loss_bdot_history"], 'r-', label='Q-PINN')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Bdot Loss")
    axes[1].set_title("Bdot Measurement Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Phi loss
    axes[2].semilogy(c["epoch_history"], c["loss_phi_history"], 'b-', label='PINN')
    axes[2].semilogy(q["epoch_history"], q["loss_phi_history"], 'r-', label='Q-PINN')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Phi Loss")
    axes[2].set_title("Potential Measurement Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "convergence_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {plot_path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN vs Q-PINN convergence analysis")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Number of training epochs (default: 2000)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate convergence plots (requires matplotlib)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to alfven_wave_dataset.h5")
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = os.path.join(
            os.path.dirname(__file__), "..", "simulation", "alfven_wave_dataset.h5"
        )

    if not os.path.exists(args.dataset):
        print(f"Dataset not found at {args.dataset}")
        sys.exit(1)

    run_comparison(args.dataset, n_epochs=args.epochs, make_plot=args.plot)
