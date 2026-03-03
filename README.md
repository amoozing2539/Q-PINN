# Q-PINN: Quantum Physics-Informed Neural Networks for Alfvén Wave Reconstruction

Reconstruct electromagnetic field dynamics from sparse probe measurements using classical PINNs and hybrid quantum-classical PINNs (Q-PINNs).

## Project Overview

1. **Simulation** — WarpX particle-in-cell simulation of Alfvén waves generates a training dataset with sparse Bdot and electrostatic potential (φ) probe measurements
2. **Models** — Classical PINN (MLP) and Q-PINN (PennyLane PQC hybrid) both take `(z, t)` coordinates and predict `φ, Bx, By, Bz, Ex, Ey, Ez, dBx/dt, dBy/dt, dBz/dt`
3. **Convergence Analysis** — Head-to-head comparison of training convergence, accuracy, and parameter efficiency

## Quick Start

```bash
# Generate simulation data (~7 min on GPU)
cd simulation
conda run -n warpx_env python run_alfven_sim.py
conda run -n warpx_env python process_dataset.py

# Train and compare models
cd ../models
conda run -n qpinn python convergence_analysis.py --epochs 2000 --plot
```

## Loading the Dataset

```python
import sys; sys.path.insert(0, "simulation")
from load_dataset import load_full_dataset, load_sparse_measurements

data = load_full_dataset("simulation/alfven_wave_dataset.h5")
sparse = load_sparse_measurements("simulation/alfven_wave_dataset.h5")

# Sparse inputs for PINN training
bdot = sparse["bdot"]   # dBx_dt: (12, 6001), probe_positions: (12,)
phi  = sparse["phi"]    # phi: (16, 1201), probe_positions: (16,)

# Ground truth for evaluation
gt = data["ground_truth"]  # Bx, By, ...: (512, 601)
```

## Model Comparison

| | Classical PINN | Q-PINN |
|---|---|---|
| Architecture | MLP (4×64 hidden, tanh) | Pre-net → PQC (7 qubits, 4 layers) → Post-net |
| Parameters | 13,322 | 521 |
| Outputs | 10 fields | 10 fields |
| Environment | `qpinn` | `qpinn` (PennyLane + lightning.gpu) |

## Environments

| Env | Purpose | Key Packages |
|-----|---------|-------------|
| `warpx_env` | PIC simulation | pywarpx 26.1, openPMD, h5py |
| `qpinn` | ML models | PennyLane 0.44, pennylane-lightning-gpu, autograd |

## File Structure

```
Q-PINN/
├── simulation/
│   ├── run_alfven_sim.py        # WarpX Alfvén wave simulation
│   ├── process_dataset.py       # Post-process → HDF5
│   ├── load_dataset.py          # Data loading utilities
│   └── alfven_wave_dataset.h5   # Generated dataset (18.2 MB)
├── models/
│   ├── classical_pinn.py        # Classical PINN model
│   ├── qpinn_model.py           # Q-PINN model
│   └── convergence_analysis.py  # Comparison script
└── README.md
```

## References

1. Trahan, R., Loveland, J., & Dent, J. (2022). Quantum Physics-Informed Neural Networks. *Entropy*, 26(8), 649. DOI: 10.3390/e26080649
2. Hegde, S., & Markidis, S. (2024). Quantum Physics Informed Neural Networks. In *Proceedings of the 2024 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation* (pp. 1–10). ACM. DOI: 10.1145/3677333.3678272
3. Markidis, S. (2022). On Physics-Informed Neural Networks for Quantum Computers. *Frontiers in Applied Mathematics and Statistics*, 8, 1036711. DOI: 10.3389/fams.2022.1036711
4. Klement, J., Eyring, J., & Schwabe, T. (2026). Explaining the Advantage of Quantum-Enhanced Physics-Informed Neural Networks. *arXiv preprint arXiv:2601.15046*. (preprint, Jan 2026 — not yet peer-reviewed)
5. Ferreira, A. A., et al. (2025). QCPINN: Quantum Classical Physics-Informed Neural Networks for Solving PDEs. *arXiv preprint arXiv:2503.16678*. (preprint, Mar 2025 — not yet peer-reviewed)
