#!/usr/bin/env python3
import os
import csv
import json
import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

RDLogger.DisableLog("rdApp.*")

# Folder format: strength_4_absorption_0_solvent_0
FOLDER_RE = re.compile(
    r"^strength_(?P<strength>\d+)_absorption_(?P<absorption>\d+)_solvent_(?P<solvent>\d+)$"
)

# ---------------------------- I/O helpers ----------------------------

def read_smiles_csv(p: Path) -> List[str]:
    rows: List[str] = []
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

def write_smiles_csv(p: Path, smiles: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for s in smiles:
            w.writerow([s])

def copy_metadata(src_dir: Path, dst_dir: Path) -> None:
    src = src_dir / "metadata.json"
    if src.is_file():
        dst = dst_dir / "metadata.json"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())

# ---------------------------- SMILES ops ----------------------------

def canonicalize(smiles_list: List[str]) -> List[str]:
    """Return list of canonical SMILES for VALID molecules only."""
    out: List[str] = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        out.append(can)
    return out

def deduplicate_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ---------------------------- Scaffold + Tanimoto ----------------------------

def get_murcko_scaffold_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return None
    return Chem.MolToSmiles(scaf, isomericSmiles=False)

def fp_from_scaffold_smiles(scaf_smi: Optional[str], radius: int, nbits: int):
    if not scaf_smi:
        return None
    mol = Chem.MolFromSmiles(scaf_smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

def mean_pairwise_tanimoto(fps: List) -> Tuple[float, int]:
    """Mean Tanimoto similarity over all i<j where both fps are not None."""
    valid = [fp for fp in fps if fp is not None]
    n = len(valid)
    if n < 2:
        return 0.0, 0
    total = 0.0
    pairs = 0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(valid[i], valid[i + 1:])
        total += sum(sims)
        pairs += len(sims)
    return (total / pairs) if pairs else 0.0, pairs

# ---------------------------- Folder summary ----------------------------

def write_summary_json(
    dst_dir: Path,
    total_rows: int,
    n_valid: int,
    n_unique: int,
    mean_scaf_tani: Optional[float] = None,
    n_pairs: Optional[int] = None,
) -> dict:
    pct_valid_total = 100.0 * n_valid / total_rows if total_rows else 0.0
    pct_unique_within_valid = 100.0 * n_unique / n_valid if n_valid else 0.0
    pct_unique_total = 100.0 * n_unique / total_rows if total_rows else 0.0
    summary = {
        "total_rows": total_rows,
        "valid": n_valid,
        "unique_valid": n_unique,
        "pct_valid_total": round(pct_valid_total, 2),
        "pct_unique_within_valid": round(pct_unique_within_valid, 2),
        "pct_unique_total": round(pct_unique_total, 2),
    }
    if mean_scaf_tani is not None:
        diversity = 1.0 - float(mean_scaf_tani)
        summary["mean_scaffold_tanimoto"] = round(float(mean_scaf_tani), 4)
        summary["scaffold_diversity"] = round(diversity, 4)
        summary["n_scaffold_pairs"] = int(n_pairs or 0)
    (dst_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

# ---------------------------- Plot helpers (Matplotlib only) ----------------------------

def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def _matrix_from_rows(rows, value_key, strengths, absorptions):
    M = np.full((len(strengths), len(absorptions)), np.nan, dtype=float)
    index_map = {(r["strength"], r["absorption"]): r for r in rows}
    for i, s in enumerate(strengths):
        for j, a in enumerate(absorptions):
            r = index_map.get((s, a))
            if r is not None:
                v = r.get(value_key, None)
                if v == "" or v is None:
                    continue
                M[i, j] = float(v)
    return M

def _heatmap_pdf(M, xvals, yvals, xlabel, ylabel, cbar_label, outpath, cmap="viridis", fmt=".1f"):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(M, aspect="auto", origin="upper", cmap=cmap)

    # ticks/labels
    ax.set_xticks(range(len(xvals)))
    ax.set_xticklabels([str(x) for x in xvals])
    ax.set_yticks(range(len(yvals)))
    ax.set_yticklabels([str(y) for y in yvals])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label)

    # ALWAYS annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(
                    j, i, format(M[i, j], fmt),
                    ha="center", va="center", fontsize=8, color="white" if M[i, j] < np.nanmax(M)*0.5 else "black"
                )

    fig.tight_layout()
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Analyze conditioning sweep: canonicalize/dedup SMILES, per-folder summary.json, global summary.csv, and (optionally) publication-ready matrices."
    )
    ap.add_argument("--input_root", default="sweep_conditionings",
                    help="source root with strength_*_absorption_*_solvent_* folders")
    ap.add_argument("--output_root", default="sweep_conditionings_dv",
                    help="destination root (mirrors structure)")
    ap.add_argument("--compute_tanimoto", action="store_true",
                    help="Compute Murcko-scaffold mean Tanimoto and diversity (on deduplicated set).")
    ap.add_argument("--fp_radius", type=int, default=2, help="ECFP/Morgan radius for scaffold fingerprints")
    ap.add_argument("--fp_nbits", type=int, default=2048, help="Fingerprint size (bits) for scaffold fingerprints")

    # NEW: create publication-ready matrix PDFs
    ap.add_argument("--make_matrices", action="store_true",
                    help="Save validity, uniqueness, and (if available) scaffold diversity matrices as vector PDFs.")
    ap.add_argument("--annotate_cells", action="store_true",
                    help="Write numeric values inside matrix cells (nice for supplementary).")

    args = ap.parse_args()

    src_root = Path(args.input_root).resolve()
    dst_root = Path(args.output_root).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    # Global CSV rows
    rows_out = []

    for name in tqdm(sorted(os.listdir(src_root))):
        m = FOLDER_RE.match(name)
        if not m:
            continue

        strength = int(m.group("strength"))
        absorption = int(m.group("absorption"))
        splitting = absorption  # per your generation script
        solvent = int(m.group("solvent"))

        src_dir = src_root / name
        smiles_path = src_dir / "smiles.csv"
        if not smiles_path.is_file():
            continue

        # Read raw, canonicalize valid, deduplicate, and write to destination
        raw = read_smiles_csv(smiles_path)
        total_rows = len(raw)
        valid_can = canonicalize(raw)
        unique_can = deduplicate_preserve_order(valid_can)

        dst_dir = dst_root / name
        write_smiles_csv(dst_dir / "smiles.csv", unique_can)
        copy_metadata(src_dir, dst_dir)

        # Optional scaffold stats on deduplicated set
        mean_scaf_tani = None
        n_pairs = None
        if args.compute_tanimoto:
            scaf_smis = [get_murcko_scaffold_smiles(s) for s in unique_can]
            fps = [fp_from_scaffold_smiles(smi, args.fp_radius, args.fp_nbits) for smi in scaf_smis]
            mean_scaf_tani, n_pairs = mean_pairwise_tanimoto(fps)

        summary = write_summary_json(
            dst_dir,
            total_rows,
            len(valid_can),
            len(unique_can),
            mean_scaf_tani=mean_scaf_tani,
            n_pairs=n_pairs,
        )

        row = {
            "strength": strength,
            "absorption": absorption,
            "splitting": splitting,
            "solvent": solvent,
            "pct_valid_total": summary["pct_valid_total"],
            "pct_unique_total": summary["pct_unique_total"],
            "pct_unique_within_valid": summary["pct_unique_within_valid"],
            "mean_scaffold_tanimoto": summary.get("mean_scaffold_tanimoto", ""),
            "scaffold_diversity": summary.get("scaffold_diversity", ""),
        }
        rows_out.append(row)

    # Write global summary.csv
    out_csv = dst_root / "summary.csv"
    with out_csv.open("w", newline="") as f:
        fieldnames = [
            "strength", "absorption", "splitting", "solvent",
            "pct_valid_total", "pct_unique_total", "pct_unique_within_valid",
            "mean_scaffold_tanimoto", "scaffold_diversity",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows_out, key=lambda x: (x["strength"], x["absorption"], x["solvent"])):
            w.writerow(r)

    print(f"Processed {len(rows_out)} folders → {dst_root}")
    print(f"Wrote global summary → {out_csv}")

    # ---------------------------- Publication-ready matrices ----------------------------
    if args.make_matrices:
        # Build matrices over the (strength, absorption) grid present
        strengths = sorted({int(r["strength"]) for r in rows_out})
        absorptions = sorted({int(r["absorption"]) for r in rows_out})

        # validity (%)
        M_valid = _matrix_from_rows(rows_out, "pct_valid_total", strengths, absorptions)
        _heatmap_pdf(
            M_valid, absorptions, strengths,
            xlabel="absorption", ylabel="strength",
            cbar_label="% valid",
            outpath=dst_root / "validity_matrix.pdf",
            cmap="viridis",
            fmt=".1f"
        )

        # uniqueness (%)
        M_unique = _matrix_from_rows(rows_out, "pct_unique_total", strengths, absorptions)
        _heatmap_pdf(
            M_unique, absorptions, strengths,
            xlabel="absorption", ylabel="strength",
            cbar_label="% unique (total)",
            outpath=dst_root / "uniqueness_matrix.pdf",
            cmap="magma",
            fmt=".1f"
        )

        # check if scaffold diversity data exists
        has_div = any(str(r.get("scaffold_diversity", "")) not in ("", "nan") for r in rows_out)

        if has_div:
            M_div = _matrix_from_rows(rows_out, "scaffold_diversity", strengths, absorptions)
            _heatmap_pdf(
                M_div, absorptions, strengths,
                xlabel="absorption", ylabel="strength",
                cbar_label="scaffold diversity",
                outpath=dst_root / "diversity_matrix.pdf",
                cmap="plasma",
                fmt=".2f"
            )

if __name__ == "__main__":
    """Example usage:
        python analyze_sweep_conditionings.py \
            --input_root sweep_conditionings \
            --output_root sweep_conditionings_dv \
            --compute_tanimoto \
            --make_matrices
    """

    main()
