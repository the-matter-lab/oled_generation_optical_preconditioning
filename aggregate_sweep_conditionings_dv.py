#!/usr/bin/env python3
import os
import csv
import argparse
from pathlib import Path
import re
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

# still parse folders like: strength_4_absorption_0_solvent_0
FOLDER_RE = re.compile(
    r"^strength_(?P<strength>\d+)_absorption_(?P<absorption>\d+)_solvent_(?P<solvent>\d+)$"
)

def read_smiles_csv(p: Path):
    rows = []
    with p.open(newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)  # skip header if present
        for row in r:
            if not row:
                continue
            s = (row[0] or "").strip()
            if s:
                rows.append(s)
    return rows

def save_global_smiles(out_csv: Path, smiles_to_meta):
    """Write global deduplicated SMILES with only strength (f_osc) and absorption (E_abs)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "strength", "absorption"])  # solvent removed
        for smi, (s, a) in smiles_to_meta.items():
            w.writerow([smi, s, a])

def jaccard_percent(a_set, b_set):
    if not a_set and not b_set:
        return 0.0
    inter = len(a_set.intersection(b_set))
    union = len(a_set.union(b_set))
    return 100.0 * inter / union if union else 0.0

def plot_upper_triangle(M, labels, out_pdf, cmap="viridis"):
    """
    Plot upper-triangular matrix M (NxN), masking the diagonal and lower half.
    Labels use LaTeX (e.g., f_osc, E_abs). Colorbar fixed to 0..10 (%).
    """
    N = M.shape[0]
    mask = np.tri(N, N, k=0, dtype=bool)  # lower triangle + diag masked
    M_plot = np.ma.array(M, mask=mask)

    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,  # keep Matplotlib's mathtext (no LaTeX dependency)
    })

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    # Colorbar range fixed to 0..10 as requested
    im = ax.imshow(M_plot, cmap=cmap, vmin=0, vmax=10)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Overlap (%)")  # label text updated

    # annotate only upper triangle (exclude diag)
    for i in range(N):
        for j in range(N):
            if j > i and not np.ma.is_masked(M_plot[i, j]) and not np.isnan(M_plot[i, j]):
                ax.text(j, i, f"{M_plot[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="white")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

def save_matrix_csv(M, labels, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, row in enumerate(M):
            w.writerow([labels[i]] + [f"{x:.4f}" if not np.isnan(x) else "" for x in row])

def main():
    ap = argparse.ArgumentParser(
        description="Aggregate per-condition SMILES globally and compute pairwise overlap (Jaccard %), plotting the upper triangle."
    )
    ap.add_argument("--input_root", default="sweep_conditionings_dv",
                    help="Root with strength_*_absorption_*_solvent_* folders")
    ap.add_argument("--output_root", default="sweep_conditionings_global",
                    help="Output folder for aggregated smiles.csv and overlap figure")
    args = ap.parse_args()

    in_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Load each condition's deduped smiles
    condition_sets = {}                 # (strength, absorption) -> set(smiles)   (solvent ignored in key)
    all_smiles_first_meta = OrderedDict()  # smi -> (strength, absorption)

    for name in sorted(os.listdir(in_root)):
        m = FOLDER_RE.match(name)
        if not m:
            continue

        strength = int(m.group("strength"))
        absorption = int(m.group("absorption"))
        # solvent parsed but ignored thereafter
        # solvent = int(m.group("solvent"))

        smiles_path = in_root / name / "smiles.csv"
        if not smiles_path.is_file():
            continue

        smi_list = read_smiles_csv(smiles_path)
        key = (strength, absorption)
        condition_sets.setdefault(key, set()).update(smi_list)

        # global dedup preserving first (strength, absorption)
        for s in smi_list:
            if s not in all_smiles_first_meta:
                all_smiles_first_meta[s] = (strength, absorption)

    # Save global dedup smiles.csv (no solvent column)
    global_csv = out_root / "smiles.csv"
    save_global_smiles(global_csv, all_smiles_first_meta)
    print(f"Global deduplicated smiles → {global_csv} (n={len(all_smiles_first_meta)})")

    # Condition list & labels (sorted by strength then absorption)
    keys = sorted(condition_sets.keys(), key=lambda k: (k[0], k[1]))

    # LaTeX-like labels for plot: f_osc, E_abs
    labels_plot = [rf"$f_{{\mathrm{{osc}}}}\, {s},\ E_{{\mathrm{{abs}}}} \, {a}$" for (s, a) in keys]

    # Also prepare simple labels for CSV header (no LaTeX)
    labels_csv = [f"S{s}-A{a}" for (s, a) in keys]

    # Compute pairwise Jaccard % overlap matrix
    N = len(keys)
    M = np.full((N, N), np.nan, dtype=float)
    for i in range(N):
        A = condition_sets[keys[i]]
        for j in range(N):
            if i == j:
                continue
            B = condition_sets[keys[j]]
            M[i, j] = jaccard_percent(A, B)

    # Save numeric matrix and triangular plot
    overlap_csv = out_root / "overlap.csv"
    save_matrix_csv(M, labels_csv, overlap_csv)
    print(f"Pairwise overlap matrix → {overlap_csv}")

    overlap_pdf = out_root / "overlap.pdf"
    plot_upper_triangle(M, labels_plot, overlap_pdf, cmap="viridis")
    print(f"Triangular overlap plot → {overlap_pdf}")

if __name__ == "__main__":
    """
    Usage example:
        python aggregate_sweep_conditionings_dv.py \
            --input_root sweep_conditionings_dv \
            --output_root sweep_conditionings_global
    """
    main()
