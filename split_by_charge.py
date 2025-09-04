#!/usr/bin/env python3
import os
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem


def read_smiles_csv(p: Path):
    with p.open(newline="") as f:
        r = csv.DictReader(f)
        if "smiles" not in r.fieldnames:
            raise ValueError(f"{p} must have a 'smiles' column")
        for row in r:
            s = (row.get("smiles") or "").strip()
            if s:
                yield s


def total_formal_charge_from_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))


def save_smiles_list(out_csv: Path, smiles_list):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for s in smiles_list:
            w.writerow([s])


def save_counts_csv(out_csv: Path, counts: Counter):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    charges_sorted = sorted(counts.keys())
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["charge", "count"])
        for c in charges_sorted:
            w.writerow([c, counts[c]])


def plot_charge_hist(charges, counts: Counter, out_pdf: Path):
    if not charges:
        return
    uniq_sorted = sorted(set(charges))
    x = np.array(uniq_sorted, dtype=int)
    y = np.array([counts[c] for c in uniq_sorted], dtype=int)

    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x, y, width=0.8, align="center")
    ax.set_xlabel("formal charge")
    ax.set_ylabel("molecules")
    ax.set_title("distribution of formal charge (counts)")
    ax.set_xticks(x)

    for xi, yi in zip(x, y):
        ax.text(xi, yi, str(int(yi)), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Split molecules by formal charge from a SMILES CSV and plot distribution."
    )
    ap.add_argument("--input_csv", required=True, help="Input CSV with a 'smiles' column")
    ap.add_argument("--output_root", required=True, help="Output root directory")
    args = ap.parse_args()

    input_csv = Path(args.input_csv).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    charge_to_smiles = defaultdict(list)
    invalid = []

    for smi in read_smiles_csv(input_csv):
        ch = total_formal_charge_from_smiles(smi)
        if ch is None:
            invalid.append(smi)
            continue
        charge_to_smiles[ch].append(smi)

    for ch, smiles_list in charge_to_smiles.items():
        sub = out_root / f"charge_{ch:+d}"
        save_smiles_list(sub / "smiles.csv", smiles_list)

    counts = Counter({ch: len(v) for ch, v in charge_to_smiles.items()})
    save_counts_csv(out_root / "summary_counts.csv", counts)
    plot_charge_hist(
        [c for c, v in charge_to_smiles.items() for _ in range(len(v))],
        counts,
        out_root / "charge_histogram.pdf",
    )

    if invalid:
        with (out_root / "invalid_smiles.txt").open("w") as f:
            for s in invalid:
                f.write(s + "\n")

    print(f"done. charges found: {sorted(charge_to_smiles.keys())}")
    print(f"wrote {out_root/'summary_counts.csv'} and {out_root/'charge_histogram.pdf'}")
    if invalid:
        print(f"skipped {len(invalid)} invalid SMILES → {out_root/'invalid_smiles.txt'}")


if __name__ == "__main__":
    """Example usage:
        python split_by_charge.py\
        --input_csv sweep_conditionings_global/smiles.csv\
        --output_root sweep_conditionings_global/charge_separated
    """
    main()

