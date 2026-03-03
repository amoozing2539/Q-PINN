#!/usr/bin/env python3
"""
Data Loading Utilities for Alfvén Wave Q-PINN Dataset
=====================================================

Provides simple functions to load the HDF5 dataset produced by
``process_dataset.py``.  Designed to be imported directly into your
Q-PINN training pipeline.

Usage
-----
>>> from load_dataset import load_full_dataset, load_sparse_measurements
>>>
>>> # Load everything
>>> data = load_full_dataset("alfven_wave_dataset.h5")
>>> print(data.keys())
>>>
>>> # Load only sparse measurements (probes)
>>> sparse = load_sparse_measurements("alfven_wave_dataset.h5")
>>> bdot  = sparse["bdot"]    # dict with dBx_dt, dBy_dt, probe_positions, ...
>>> phi   = sparse["phi"]     # dict with phi, probe_positions, ...
>>>
>>> # Load only ground truth fields
>>> gt = load_ground_truth("alfven_wave_dataset.h5")
>>> Bx = gt["Bx"]             # shape (Nz, Nt)
"""

import json
from typing import Any, Dict, Optional

import h5py
import numpy as np


def load_full_dataset(filepath: str) -> Dict[str, Any]:
    """
    Load the complete Alfvén wave dataset.

    Returns
    -------
    dict with keys:
        "ground_truth" : dict – Bx, By, Bz, Ex, Ey, Ez, phi, z_coords, t_coords
        "sparse_bdot"  : dict – dBx_dt, dBy_dt, dBz_dt, Bx, By, Bz,
                                probe_positions, t_coords
        "sparse_phi"   : dict – phi, probe_positions, t_coords
        "normalization": dict – B0, n_plasma, vA, l_i, t_ci, etc.
        "metadata"     : dict – sim_params, description
    """
    result = {}

    with h5py.File(filepath, "r") as hf:
        # Ground truth
        if "ground_truth" in hf:
            result["ground_truth"] = _read_group(hf["ground_truth"])

        # Sparse Bdot
        if "sparse_bdot" in hf:
            result["sparse_bdot"] = _read_group(hf["sparse_bdot"])

        # Sparse potential
        if "sparse_potential" in hf:
            result["sparse_phi"] = _read_group(hf["sparse_potential"])

        # Normalization
        if "normalization" in hf:
            result["normalization"] = dict(hf["normalization"].attrs)

        # Metadata
        if "metadata" in hf:
            meta = dict(hf["metadata"].attrs)
            if "sim_params_json" in meta:
                meta["sim_params"] = json.loads(meta["sim_params_json"])
            result["metadata"] = meta

    return result


def load_sparse_measurements(filepath: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load only the sparse probe measurements (Bdot + potential).

    Returns
    -------
    dict with keys:
        "bdot": dict
            - dBx_dt      : (N_probes, Nt)  time derivative of Bx
            - dBy_dt      : (N_probes, Nt)
            - dBz_dt      : (N_probes, Nt)
            - Bx, By, Bz  : (N_probes, Nt) raw B-field at probes
            - probe_positions : (N_probes,)  z-coordinates
            - t_coords    : (Nt,)
        "phi": dict
            - phi          : (N_phi, Nt_phi) electrostatic potential
            - probe_positions : (N_phi,)
            - t_coords    : (Nt_phi,)
    """
    result = {}

    with h5py.File(filepath, "r") as hf:
        if "sparse_bdot" in hf:
            result["bdot"] = _read_group(hf["sparse_bdot"])
        if "sparse_potential" in hf:
            result["phi"] = _read_group(hf["sparse_potential"])

    return result


def load_ground_truth(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load only the full-field ground truth data.

    Returns
    -------
    dict with keys:
        Bx, By, Bz  : (Nz, Nt) magnetic field components
        Ex, Ey, Ez   : (Nz, Nt) electric field components
        phi          : (Nz, Nt) electrostatic potential (if available)
        z_coords     : (Nz,)    spatial coordinates
        t_coords     : (Nt,)    time coordinates
    """
    with h5py.File(filepath, "r") as hf:
        if "ground_truth" not in hf:
            raise KeyError("Dataset does not contain 'ground_truth' group")
        return _read_group(hf["ground_truth"])


def load_normalization(filepath: str) -> Dict[str, float]:
    """
    Load normalization / physical constants.

    Returns
    -------
    dict with keys like B0, n_plasma, vA, l_i, t_ci, w_ci, etc.
    """
    with h5py.File(filepath, "r") as hf:
        if "normalization" not in hf:
            raise KeyError("Dataset does not contain 'normalization' group")
        return dict(hf["normalization"].attrs)


def print_dataset_summary(filepath: str) -> None:
    """Print a human-readable summary of the dataset contents."""
    print(f"Dataset: {filepath}")
    print("=" * 60)

    with h5py.File(filepath, "r") as hf:
        for group_name in hf:
            grp = hf[group_name]
            print(f"\n  [{group_name}]")
            # Print datasets
            for key in grp:
                ds = grp[key]
                if isinstance(ds, h5py.Dataset):
                    print(f"    {key:25s}  shape={str(ds.shape):20s}  dtype={ds.dtype}")
            # Print attributes
            for key, val in grp.attrs.items():
                val_str = str(val)
                if len(val_str) > 60:
                    val_str = val_str[:57] + "..."
                print(f"    @{key:24s}  = {val_str}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_group(grp: h5py.Group) -> Dict[str, Any]:
    """Read all datasets from an HDF5 group into a dict of numpy arrays."""
    result = {}
    for key in grp:
        ds = grp[key]
        if isinstance(ds, h5py.Dataset):
            result[key] = ds[:]
    # Also include attributes
    for key, val in grp.attrs.items():
        result[f"attr_{key}"] = val
    return result


# ---------------------------------------------------------------------------
# CLI: print summary if run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        fpath = "alfven_wave_dataset.h5"
    else:
        fpath = sys.argv[1]

    print_dataset_summary(fpath)

    print("\n--- Quick Load Test ---")
    data = load_full_dataset(fpath)
    for section, content in data.items():
        print(f"\n{section}:")
        if isinstance(content, dict):
            for k, v in content.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                          f"range=[{v.min():.4e}, {v.max():.4e}]")
                else:
                    vstr = str(v)
                    if len(vstr) > 80:
                        vstr = vstr[:77] + "..."
                    print(f"  {k}: {vstr}")
