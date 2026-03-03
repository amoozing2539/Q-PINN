#!/usr/bin/env python3
"""
Alfvén Wave PIC Simulation using WarpX (HybridPIC Solver)
=========================================================

Generates training data for Q-PINNs by simulating Alfvén wave propagation
in a 1D magnetised plasma. Outputs:
  1. Full-field ground truth (B, E, J) via openPMD / HDF5
  2. Sparse Bdot probe data  (dB/dt at a few locations, high cadence)
  3. Sparse potential (phi) probe data (electrostatic potential at a few locations)

Uses the HybridPICSolver: kinetic ions + isothermal fluid electrons,
with Ohm's law for the electric field.

Run:
    conda run -n warpx_env python run_alfven_sim.py

Dependencies (all in warpx_env):
    pywarpx >=26, numpy, h5py, scipy
"""

import argparse
import os
import sys

import numpy as np
from pywarpx import callbacks, libwarpx, picmi

constants = picmi.constants

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

# Applied magnetic field
B0 = 0.25  # Tesla – strong enough for clear Alfvén modes

# Plasma beta (controls temperature)
BETA = 0.01

# Ion mass in units of electron masses
M_OVER_ME = 100.0

# Alfvén speed / speed of light ratio
VA_OVER_C = 1e-4

# Grid
NZ = 512              # cells along z (propagation direction)
DZ_OVER_DI = 0.1      # cell size in ion skin depths

# Time stepping
DT_OVER_TCI = 5e-3    # timestep in ion cyclotron periods
N_CYCLOTRON_PERIODS = 30.0  # total simulation time

# Particles
NPPC = 256  # macro-particles per cell

# Resistivity (damps highest-k modes, stabilises numerics)
ETA = 1e-7

# Ohm solver sub-steps
SUBSTEPS = 40

# ----- Diagnostic cadences -----
FIELD_DIAG_PERIOD = 10       # full-field dump every N steps
BDOT_PROBE_PERIOD = 1        # Bdot probes every step (high cadence)
PHI_PROBE_PERIOD  = 5        # potential probes every 5 steps

# ----- Sparse probe counts -----
N_BDOT_PROBES = 12           # number of sparse Bdot point probes
N_PHI_PROBES  = 16           # number of sparse phi point probes

# Reproducibility
RNG_SEED = 42


def compute_plasma_quantities():
    """Derive physical plasma parameters from the input knobs."""
    M = M_OVER_ME * constants.m_e  # ion mass (kg)

    # Cyclotron frequency & period
    w_ci = constants.q_e * B0 / M
    t_ci = 2.0 * np.pi / w_ci

    # Alfvén speed
    vA = VA_OVER_C * constants.c

    # Plasma density  (from vA = B/sqrt(mu0 * n * (M+me)))
    n_plasma = (B0 / vA) ** 2 / (constants.mu0 * (M + constants.m_e))

    # Ion skin depth
    w_pi = np.sqrt(constants.q_e ** 2 * n_plasma / (M * constants.ep0))
    l_i  = constants.c / w_pi

    # Thermal velocity  (beta = 2*(v_ti/vA)^2)
    v_ti = np.sqrt(BETA / 2.0) * vA

    # Temperature in eV
    T_eV = v_ti ** 2 * M / constants.q_e

    return dict(
        M=M, w_ci=w_ci, t_ci=t_ci, vA=vA,
        n_plasma=n_plasma, w_pi=w_pi, l_i=l_i,
        v_ti=v_ti, T_eV=T_eV,
    )


def build_simulation(test_mode=False):
    """Construct the full WarpX simulation object."""

    pq = compute_plasma_quantities()

    # Spatial / temporal quantities
    dz = DZ_OVER_DI * pq["l_i"]
    Lz = NZ * dz
    dt = DT_OVER_TCI * pq["t_ci"]

    if test_mode:
        total_steps = 100
    else:
        total_steps = int(N_CYCLOTRON_PERIODS / DT_OVER_TCI)

    print(f"=== Alfvén Wave PIC Simulation ===")
    print(f"  B0       = {B0:.3f} T")
    print(f"  n_plasma = {pq['n_plasma']:.3e} m^-3")
    print(f"  T_e=T_i  = {pq['T_eV']:.3f} eV")
    print(f"  vA       = {pq['vA']:.3e} m/s")
    print(f"  l_i      = {pq['l_i']:.3e} m")
    print(f"  t_ci     = {pq['t_ci']:.3e} s")
    print(f"  Lz       = {Lz:.3e} m  ({NZ} cells)")
    print(f"  dt       = {dt:.3e} s")
    print(f"  steps    = {total_steps}")
    print()

    # ---- Simulation object ----
    sim = picmi.Simulation(
        warpx_serialize_initial_conditions=True,
        verbose=1,
    )
    sim.time_step_size = dt
    sim.max_steps = total_steps
    sim.current_deposition_algo = "direct"
    sim.particle_shape = 1

    # ---- Grid ----
    grid = picmi.Cartesian1DGrid(
        number_of_cells=[NZ],
        lower_bound=[0.0],
        upper_bound=[Lz],
        lower_boundary_conditions=["periodic"],
        upper_boundary_conditions=["periodic"],
        warpx_max_grid_size=NZ,
    )

    # ---- Solver (Hybrid PIC: kinetic ions + fluid electrons) ----
    solver = picmi.HybridPICSolver(
        grid=grid,
        Te=pq["T_eV"],
        n0=pq["n_plasma"],
        plasma_resistivity=ETA,
        substeps=SUBSTEPS,
    )
    sim.solver = solver

    # ---- External B-field (along z – parallel propagation) ----
    B_ext = picmi.AnalyticInitialField(
        Bx_expression=0.0,
        By_expression=0.0,
        Bz_expression=B0,
    )
    sim.add_applied_field(B_ext)

    # ---- Ion species ----
    ions = picmi.Species(
        name="ions",
        charge="q_e",
        mass=pq["M"],
        initial_distribution=picmi.UniformDistribution(
            density=pq["n_plasma"],
            rms_velocity=[pq["v_ti"]] * 3,
        ),
    )
    sim.add_species(
        ions,
        layout=picmi.PseudoRandomLayout(
            grid=grid,
            n_macroparticles_per_cell=NPPC,
        ),
    )

    # ==================================================================
    # Diagnostics
    # ==================================================================

    # 1. Full-field openPMD diagnostic (ground truth) -----------------
    field_diag = picmi.FieldDiagnostic(
        name="full_field",
        grid=grid,
        period=FIELD_DIAG_PERIOD,
        data_list=["B", "E", "J"],
        write_dir="diags",
        warpx_format="openpmd",
        warpx_openpmd_backend="h5",
    )
    sim.add_diagnostic(field_diag)

    # 2. Line probe for full spatial data (used for phi computation) --
    #    Records E-field along z at higher cadence than full 3D dumps
    line_diag = picmi.ReducedDiagnostic(
        diag_type="FieldProbe",
        probe_geometry="Line",
        z_probe=0.0,
        z1_probe=Lz,
        resolution=NZ - 1,
        name="line_field",
        period=PHI_PROBE_PERIOD,
        path="diags/",
    )
    sim.add_diagnostic(line_diag)

    # 3. Sparse Bdot point probes ------------------------------------
    rng = np.random.RandomState(RNG_SEED)
    bdot_positions = np.sort(rng.uniform(0.0, Lz, size=N_BDOT_PROBES))

    for i, zp in enumerate(bdot_positions):
        probe = picmi.ReducedDiagnostic(
            diag_type="FieldProbe",
            probe_geometry="Point",
            z_probe=zp,
            name=f"bdot_probe_{i:02d}",
            period=BDOT_PROBE_PERIOD,
            path="diags/",
        )
        sim.add_diagnostic(probe)

    # 4. Sparse phi point probes (record E_z, integrate for phi) -----
    phi_positions = np.sort(rng.uniform(0.0, Lz, size=N_PHI_PROBES))

    for i, zp in enumerate(phi_positions):
        probe = picmi.ReducedDiagnostic(
            diag_type="FieldProbe",
            probe_geometry="Point",
            z_probe=zp,
            name=f"phi_probe_{i:02d}",
            period=PHI_PROBE_PERIOD,
            path="diags/",
        )
        sim.add_diagnostic(probe)

    # ==================================================================
    # Save simulation metadata for post-processing
    # ==================================================================
    meta = {
        **{k: float(v) for k, v in pq.items()},
        "B0": B0,
        "beta": BETA,
        "NZ": NZ,
        "Lz": Lz,
        "dz": dz,
        "dt": dt,
        "total_steps": total_steps,
        "NPPC": NPPC,
        "eta": ETA,
        "field_diag_period": FIELD_DIAG_PERIOD,
        "bdot_probe_period": BDOT_PROBE_PERIOD,
        "phi_probe_period": PHI_PROBE_PERIOD,
        "n_bdot_probes": N_BDOT_PROBES,
        "n_phi_probes": N_PHI_PROBES,
        "bdot_positions": bdot_positions.tolist(),
        "phi_positions": phi_positions.tolist(),
    }

    os.makedirs("diags", exist_ok=True)
    import json
    with open("diags/sim_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Metadata saved to diags/sim_metadata.json")
    print(f"Bdot probe positions: {bdot_positions}")
    print(f"Phi probe positions:  {phi_positions}")

    return sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alfvén wave PIC simulation")
    parser.add_argument(
        "-t", "--test", action="store_true",
        help="Run a short test (100 steps) instead of full simulation",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining

    sim = build_simulation(test_mode=args.test)
    sim.step()
    print("\n=== Simulation complete ===")
