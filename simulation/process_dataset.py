#!/usr/bin/env python3
"""
Post-processing: WarpX Alfvén wave output → ML-ready HDF5 dataset
=================================================================

Reads openPMD field dumps and reduced-diagnostic probe files produced by
``run_alfven_sim.py`` and packages everything into a single
``alfven_wave_dataset.h5`` suitable for Q-PINN training.

Run (from the simulation/ directory, after the simulation finishes):
    conda run -n warpx_env python process_dataset.py

Outputs:
    alfven_wave_dataset.h5
"""

import glob
import json
import os
import re
import sys

import h5py
import numpy as np

# Allow running from the simulation/ directory or one level up
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIAGS_DIR = os.path.join(SCRIPT_DIR, "diags")

# ---------------------------------------------------------------------------
# Helpers to read WarpX reduced-diagnostic text files
# ---------------------------------------------------------------------------

def read_probe_file(filepath):
    """
    Read a WarpX FieldProbe reduced-diagnostic text file.

    Returns
    -------
    header : list[str]   – column names
    data   : np.ndarray   – (N_records, N_columns)
    """
    with open(filepath, "r") as f:
        header_line = f.readline().strip()
    # Parse header: "[0]step() [1]time(s) [2]z_coord(m) [3]Ex ..."
    cols = re.findall(r'\[\d+\](\S+)', header_line)
    data = np.loadtxt(filepath, skiprows=1)
    return cols, data


def read_openpmd_fields(diags_dir):
    """
    Read full-field data from openPMD/HDF5 diagnostics.

    Returns dict with keys: Bx, By, Bz, Ex, Ey, Ez, z_coords, t_coords
    """
    try:
        from openpmd_viewer import OpenPMDTimeSeries
    except ImportError:
        print("WARNING: openpmd_viewer not available, trying manual HDF5 read")
        return _read_openpmd_manual(diags_dir)

    openpmd_dir = os.path.join(diags_dir, "full_field")
    ts = OpenPMDTimeSeries(openpmd_dir)

    iterations = ts.iterations
    t_coords = np.array(ts.t)

    # Read first iteration to get grid shape
    Bx0, info = ts.get_field("B", "x", iteration=iterations[0])
    z_coords = info.z

    Nt = len(iterations)
    Nz = len(z_coords)

    fields = {
        "Bx": np.zeros((Nz, Nt)),
        "By": np.zeros((Nz, Nt)),
        "Bz": np.zeros((Nz, Nt)),
        "Ex": np.zeros((Nz, Nt)),
        "Ey": np.zeros((Nz, Nt)),
        "Ez": np.zeros((Nz, Nt)),
    }

    for i, it in enumerate(iterations):
        for comp in ["x", "y", "z"]:
            B, _ = ts.get_field("B", comp, iteration=it)
            E, _ = ts.get_field("E", comp, iteration=it)
            fields[f"B{comp}"][:, i] = B.flatten()
            fields[f"E{comp}"][:, i] = E.flatten()

    fields["z_coords"] = z_coords
    fields["t_coords"] = t_coords
    return fields


def _read_openpmd_manual(diags_dir):
    """Fallback: read openPMD HDF5 files directly with h5py."""
    openpmd_dir = os.path.join(diags_dir, "full_field")
    h5_files = sorted(glob.glob(os.path.join(openpmd_dir, "*.h5")))

    if not h5_files:
        # Try the openpmd directory structure
        h5_files = sorted(glob.glob(os.path.join(openpmd_dir, "openpmd_*.h5")))

    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {openpmd_dir}")

    fields = {"Bx": [], "By": [], "Bz": [], "Ex": [], "Ey": [], "Ez": []}
    t_coords = []
    z_coords = None

    for fpath in h5_files:
        with h5py.File(fpath, "r") as hf:
            # Navigate openPMD structure
            base = hf["data"]
            for iteration_key in sorted(base.keys(), key=int):
                it_grp = base[iteration_key]
                t_coords.append(float(it_grp.attrs.get("time", 0)))

                fld = it_grp["fields"]
                for comp in ["x", "y", "z"]:
                    B = fld["B"][comp][:]
                    E = fld["E"][comp][:]
                    fields[f"B{comp}"].append(B.flatten())
                    fields[f"E{comp}"].append(E.flatten())

                if z_coords is None:
                    # Try to extract grid coords from attributes
                    Bx_ds = fld["B"]["x"]
                    nz = Bx_ds.shape[-1]
                    # Get grid spacing and offset
                    dx = fld["B"].attrs.get("gridSpacing", [1.0])[-1]
                    offset = fld["B"].attrs.get("gridGlobalOffset", [0.0])[-1]
                    z_coords = offset + np.arange(nz) * dx

    for k in fields:
        fields[k] = np.array(fields[k]).T  # (Nz, Nt)

    fields["z_coords"] = np.array(z_coords) if z_coords is not None else np.arange(fields["Bx"].shape[0])
    fields["t_coords"] = np.array(t_coords)
    return fields


# ---------------------------------------------------------------------------
# Process sparse probes
# ---------------------------------------------------------------------------

def process_bdot_probes(diags_dir, n_probes):
    """
    Read Bdot point-probe files, compute dB/dt via finite differences.

    Returns
    -------
    dict with keys: dBx_dt, dBy_dt, dBz_dt, Bx, By, Bz, t_coords, probe_positions
    """
    all_Bx, all_By, all_Bz = [], [], []
    probe_positions = []
    t_coords = None

    for i in range(n_probes):
        fname = os.path.join(diags_dir, f"bdot_probe_{i:02d}.txt")
        if not os.path.exists(fname):
            print(f"  WARNING: {fname} not found, skipping")
            continue

        cols, data = read_probe_file(fname)

        # Columns: step, time, part_x, part_y, part_z, Ex, Ey, Ez, Bx, By, Bz, S
        t = data[:, 1]  # time

        # Find z-coordinate column (part_z_lev0)
        z_idx = next((j for j, c in enumerate(cols) if 'part_z' in c.lower()), 4)
        z = data[0, z_idx]  # probe position (constant for point probe)

        # Find B-field columns
        bx_idx = next((j for j, c in enumerate(cols) if 'Bx' in c or 'bx' in c.lower()), 8)
        by_idx = next((j for j, c in enumerate(cols) if 'By' in c or 'by' in c.lower()), 9)
        bz_idx = next((j for j, c in enumerate(cols) if 'Bz' in c or 'bz' in c.lower()), 10)

        all_Bx.append(data[:, bx_idx])
        all_By.append(data[:, by_idx])
        all_Bz.append(data[:, bz_idx])
        probe_positions.append(z)

        if t_coords is None:
            t_coords = t

    all_Bx = np.array(all_Bx)  # (N_probes, Nt)
    all_By = np.array(all_By)
    all_Bz = np.array(all_Bz)

    # Compute dB/dt via central finite differences
    dt = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 1.0
    dBx_dt = np.gradient(all_Bx, dt, axis=1)
    dBy_dt = np.gradient(all_By, dt, axis=1)
    dBz_dt = np.gradient(all_Bz, dt, axis=1)

    return {
        "Bx": all_Bx,
        "By": all_By,
        "Bz": all_Bz,
        "dBx_dt": dBx_dt,
        "dBy_dt": dBy_dt,
        "dBz_dt": dBz_dt,
        "t_coords": t_coords,
        "probe_positions": np.array(probe_positions),
    }


def process_phi_probes(diags_dir, n_probes, dz, Lz):
    """
    Read phi point-probe files.  Since the HybridPICSolver computes E from
    Ohm's law, we derive the electrostatic potential phi from E_z at each
    probe location by using phi = -integral(E_z dz).  For point probes we
    simply record the local E_z and compute a running integral approximation.

    For a true phi we also read the line probe (full Ez along z) and compute
    phi(z) = -cumulative_trapz(Ez, z), then sample at probe positions.

    Returns
    -------
    dict with keys: phi, Ez_local, t_coords, probe_positions
    """
    # --- Strategy: use the line probe to get full Ez(z,t), then compute
    #     phi(z,t) = -cumtrapz(Ez, z) and sample at probe positions ---

    line_file = os.path.join(diags_dir, "line_field.txt")
    if os.path.exists(line_file):
        return _phi_from_line_probe(line_file, diags_dir, n_probes, dz, Lz)
    else:
        # Fallback: use point probes directly (phi ~ -Ez * local_dz)
        return _phi_from_point_probes(diags_dir, n_probes)


def _phi_from_line_probe(line_file, diags_dir, n_probes, dz, Lz):
    """Compute phi from the line probe Ez data."""
    cols, data = read_probe_file(line_file)

    # Columns: step, time, part_x, part_y, part_z, Ex, Ey, Ez, Bx, By, Bz, S
    steps = data[:, 0].astype(int)
    times = data[:, 1]

    # Find z and Ez columns
    z_idx = next((j for j, c in enumerate(cols) if 'part_z' in c.lower()), 4)
    ez_idx = next((j for j, c in enumerate(cols) if 'Ez' in c), 7)
    z_vals = data[:, z_idx]

    # Group by timestep
    unique_steps = np.unique(steps)
    Nt = len(unique_steps)
    # Get z-grid from first timestep
    mask0 = steps == unique_steps[0]
    z_grid = z_vals[mask0]
    Nz_probe = len(z_grid)

    Ez_full = np.zeros((Nz_probe, Nt))
    t_coords = np.zeros(Nt)

    for i, s in enumerate(unique_steps):
        mask = steps == s
        Ez_full[:, i] = data[mask, ez_idx]
        t_coords[i] = times[mask][0]

    # Compute phi(z,t) = -cumulative integral of Ez along z
    from scipy.integrate import cumulative_trapezoid
    # phi(z=0) = 0 (reference), phi(z) = -int_0^z Ez dz'
    phi_full = np.zeros_like(Ez_full)
    phi_full[1:, :] = -cumulative_trapezoid(Ez_full, z_grid, axis=0)

    # Read probe positions from metadata
    meta_file = os.path.join(diags_dir, "sim_metadata.json")
    with open(meta_file, "r") as f:
        meta = json.load(f)
    probe_positions = np.array(meta["phi_positions"])

    # Interpolate phi at probe positions
    phi_sparse = np.zeros((n_probes, Nt))
    for t_i in range(Nt):
        phi_sparse[:, t_i] = np.interp(probe_positions, z_grid, phi_full[:, t_i])

    return {
        "phi": phi_sparse,
        "phi_full": phi_full,  # bonus: full phi(z,t) for ground truth
        "z_grid_full": z_grid,
        "t_coords": t_coords,
        "probe_positions": probe_positions,
    }


def _phi_from_point_probes(diags_dir, n_probes):
    """Fallback: read point probes and record local Ez (approximate phi)."""
    all_Ez = []
    probe_positions = []
    t_coords = None

    for i in range(n_probes):
        fname = os.path.join(diags_dir, f"phi_probe_{i:02d}.txt")
        if not os.path.exists(fname):
            continue
        cols, data = read_probe_file(fname)
        t = data[:, 1]
        z_idx = next((j for j, c in enumerate(cols) if 'part_z' in c.lower()), 4)
        z = data[0, z_idx]
        ez_idx = next((j for j, c in enumerate(cols) if 'Ez' in c), 7)
        all_Ez.append(data[:, ez_idx])
        probe_positions.append(z)
        if t_coords is None:
            t_coords = t

    return {
        "Ez_local": np.array(all_Ez),
        "phi": None,
        "t_coords": t_coords,
        "probe_positions": np.array(probe_positions),
    }


# ---------------------------------------------------------------------------
# Build final HDF5 dataset
# ---------------------------------------------------------------------------

def build_dataset(output_path="alfven_wave_dataset.h5"):
    """Main entry point: read all diagnostics and write HDF5."""

    # Load metadata
    meta_file = os.path.join(DIAGS_DIR, "sim_metadata.json")
    if not os.path.exists(meta_file):
        print(f"ERROR: {meta_file} not found. Run the simulation first.")
        sys.exit(1)

    with open(meta_file, "r") as f:
        meta = json.load(f)

    print("=" * 60)
    print("Post-processing Alfvén wave simulation data")
    print("=" * 60)

    # 1. Full-field ground truth
    print("\n[1/3] Reading full-field openPMD data ...")
    try:
        gt = read_openpmd_fields(DIAGS_DIR)
        have_gt = True
        print(f"  Ground truth shape: Bx = {gt['Bx'].shape}")
    except Exception as e:
        print(f"  WARNING: Could not read openPMD fields: {e}")
        print(f"  Will skip ground truth fields in dataset.")
        have_gt = False

    # 2. Sparse Bdot probes
    print("\n[2/3] Processing sparse Bdot probes ...")
    bdot = process_bdot_probes(DIAGS_DIR, meta["n_bdot_probes"])
    print(f"  Bdot probes: {len(bdot['probe_positions'])} probes, "
          f"{bdot['dBx_dt'].shape[1]} time steps")

    # 3. Sparse phi probes
    print("\n[3/3] Processing sparse potential (phi) probes ...")
    phi_data = process_phi_probes(
        DIAGS_DIR, meta["n_phi_probes"],
        dz=meta["dz"], Lz=meta["Lz"],
    )
    if phi_data["phi"] is not None:
        print(f"  Phi probes: {len(phi_data['probe_positions'])} probes, "
              f"{phi_data['phi'].shape[1]} time steps")
    else:
        print(f"  Phi probes: using Ez fallback (no integration)")

    # ---------- Write HDF5 ----------
    output_full_path = os.path.join(SCRIPT_DIR, output_path)
    print(f"\nWriting dataset to: {output_full_path}")

    with h5py.File(output_full_path, "w") as hf:
        # -- Ground truth --
        if have_gt:
            grp = hf.create_group("ground_truth")
            for key in ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]:
                grp.create_dataset(key, data=gt[key], compression="gzip")
            grp.create_dataset("z_coords", data=gt["z_coords"])
            grp.create_dataset("t_coords", data=gt["t_coords"])

        # Also store full phi if available
        if phi_data.get("phi_full") is not None:
            grp_phi_gt = hf.require_group("ground_truth")
            grp_phi_gt.create_dataset("phi", data=phi_data["phi_full"],
                                       compression="gzip")
            grp_phi_gt.create_dataset("z_coords_phi",
                                       data=phi_data["z_grid_full"])
            grp_phi_gt.create_dataset("t_coords_phi",
                                       data=phi_data["t_coords"])

        # -- Sparse Bdot --
        grp = hf.create_group("sparse_bdot")
        grp.create_dataset("dBx_dt", data=bdot["dBx_dt"], compression="gzip")
        grp.create_dataset("dBy_dt", data=bdot["dBy_dt"], compression="gzip")
        grp.create_dataset("dBz_dt", data=bdot["dBz_dt"], compression="gzip")
        grp.create_dataset("Bx", data=bdot["Bx"], compression="gzip")
        grp.create_dataset("By", data=bdot["By"], compression="gzip")
        grp.create_dataset("Bz", data=bdot["Bz"], compression="gzip")
        grp.create_dataset("probe_positions", data=bdot["probe_positions"])
        grp.create_dataset("t_coords", data=bdot["t_coords"])

        # -- Sparse potential (phi) --
        grp = hf.create_group("sparse_potential")
        if phi_data["phi"] is not None:
            grp.create_dataset("phi", data=phi_data["phi"], compression="gzip")
        if phi_data.get("Ez_local") is not None:
            grp.create_dataset("Ez_local", data=phi_data["Ez_local"],
                               compression="gzip")
        grp.create_dataset("probe_positions", data=phi_data["probe_positions"])
        grp.create_dataset("t_coords", data=phi_data["t_coords"])

        # -- Normalization constants --
        grp = hf.create_group("normalization")
        for key in ["B0", "n_plasma", "vA", "l_i", "t_ci", "w_ci", "w_pi",
                     "v_ti", "T_eV", "M", "beta", "dz", "Lz", "dt"]:
            if key in meta:
                grp.attrs[key] = meta[key]

        # -- Metadata --
        grp = hf.create_group("metadata")
        grp.attrs["description"] = (
            "Alfvén wave PIC simulation dataset for Q-PINN training. "
            "Contains full-field ground truth (B, E, phi) and sparse "
            "measurements (Bdot probes, potential probes)."
        )
        grp.attrs["NZ"] = meta["NZ"]
        grp.attrs["total_steps"] = meta["total_steps"]
        grp.attrs["n_bdot_probes"] = meta["n_bdot_probes"]
        grp.attrs["n_phi_probes"] = meta["n_phi_probes"]
        grp.attrs["field_diag_period"] = meta["field_diag_period"]
        grp.attrs["bdot_probe_period"] = meta["bdot_probe_period"]
        grp.attrs["phi_probe_period"] = meta["phi_probe_period"]
        # Store as JSON string for complex types
        grp.attrs["sim_params_json"] = json.dumps(meta)

    file_size_mb = os.path.getsize(output_full_path) / (1024 ** 2)
    print(f"\nDataset written: {output_full_path}  ({file_size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    build_dataset()
