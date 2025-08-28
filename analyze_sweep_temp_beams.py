#!/usr/bin/env python3
import os
import csv
import json
import argparse
import re
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")

FOLDER_RE = re.compile(r"^temp_(?P<temp>-?\d+(?:\.\d+)?)_beams_(?P<beams>\d+)$")

# ---------------------------- I/O helpers ----------------------------

def read_smiles_csv(p: Path):
    rows = []
    with p.open(newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)  # skip header
        for row in r:
            if not row:
                continue
            s = (row[0] or "").strip()
            if s:
                rows.append(s)
    return rows

def write_smiles_csv(p: Path, smiles):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for s in smiles:
            w.writerow([s])

def copy_metadata(src_dir: Path, dst_dir: Path):
    src = src_dir / "metadata.json"
    if src.is_file():
        dst = dst_dir / "metadata.json"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())

# ---------------------------- SMILES ops ----------------------------

def canonicalize(smiles_list):
    """Return list of canonical SMILES (only valid ones)."""
    valid_can = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        valid_can.append(can)
    return valid_can

def deduplicate_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ---------------------------- Scaffold + Tanimoto ----------------------------

def get_murcko_scaffold_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return None
    return Chem.MolToSmiles(scaf, isomericSmiles=False)

def fp_from_scaffold_smiles(scaf_smi: str, radius: int, nbits: int):
    if scaf_smi is None:
        return None
    mol = Chem.MolFromSmiles(scaf_smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

def mean_pairwise_tanimoto(fps):
    """Mean Tanimoto similarity over all i<j where both fps are not None.
       Returns (mean, n_pairs_used). If n_pairs_used==0 -> (0.0, 0)."""
    valid = [fp for fp in fps if fp is not None]
    n = len(valid)
    if n < 2:
        return 0.0, 0
    total = 0.0
    pairs = 0
    for i in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(valid[i], valid[i+1:])
        total += sum(sims)
        pairs += len(sims)
    return (total / pairs) if pairs else 0.0, pairs

# ---------------------------- Folder summary ----------------------------

def write_summary_json(dst_dir: Path, total_rows, n_valid, n_unique,
                       mean_scaf_tani=None, n_pairs=None):
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

# ---------------------------- Plot helpers ----------------------------

def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def collect_xy(rows, series_key):
    """Return (x1,y1,x250,y250) for beams 1 and 250 keyed by temperature."""
    temps = sorted({r["temperature"] for r in rows})
    b1 = {r["temperature"]: r[series_key] for r in rows if r["beams"] == 1 and series_key in r}
    b250 = {r["temperature"]: r[series_key] for r in rows if r["beams"] == 250 and series_key in r}
    x1 = [t for t in temps if t in b1]
    y1 = [b1[t] for t in x1]
    x250 = [t for t in temps if t in b250]
    y250 = [b250[t] for t in x250]
    return (x1, y1, x250, y250)

def add_series(ax, x1, y1, x250, y250, ylabel, xlabel=None, show_legend=False):
    line1 = ax.plot(x1, y1, marker="o", color="black", label="beams=1")[0] if x1 else None
    line2 = ax.plot(x250, y250, marker="o", color="magenta", label="beams=250")[0] if x250 else None
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if show_legend:
        handles = [h for h in (line1, line2) if h is not None]
        labels = [h.get_label() for h in handles]
        if handles:
            ax.legend(handles, labels, frameon=False, handlelength=2.5, ncols=len(handles))

# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Deduplicate canonical SMILES per sweep folder, copy metadata, write summaries, and generate panel PDFs."
    )
    ap.add_argument("--input_root", default="sweep_temp_beams", help="source root with temp_*_beams_* folders")
    ap.add_argument("--output_root", default="sweep_temp_beams_dv", help="destination root")
    ap.add_argument("--metrics_pdf_name", default="metrics_panels.pdf",
                    help="3x1 PDF for validity/uniqueness panels (saved under output_root)")
    ap.add_argument("--diversity_pdf_name", default="diversity_panels.pdf",
                    help="2x1 PDF for mean Tanimoto & diversity (saved under output_root)")

    # optional scaffold similarity computation
    ap.add_argument("--compute_tanimoto", action="store_true",
                    help="Compute Murcko scaffold mean Tanimoto (ECFP) and report diversity = 1 - mean.")
    ap.add_argument("--fp_radius", type=int, default=2, help="ECFP/Morgan radius for scaffold fingerprints")
    ap.add_argument("--fp_nbits", type=int, default=2048, help="Fingerprint size (bits) for scaffold fingerprints")

    args = ap.parse_args()

    src_root = Path(args.input_root).resolve()
    dst_root = Path(args.output_root).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for name in tqdm(sorted(os.listdir(src_root))):
        m = FOLDER_RE.match(name)
        if not m:
            continue
        temp = m.group("temp")
        beams = int(m.group("beams"))

        src_dir = src_root / name
        smiles_path = src_dir / "smiles.csv"
        if not smiles_path.is_file():
            continue

        # read raw, derive canonical + dedup, write to destination
        raw = read_smiles_csv(smiles_path)
        total_rows = len(raw)
        valid_can = canonicalize(raw)
        unique_can = deduplicate_preserve_order(valid_can)

        dst_dir = dst_root / name
        write_smiles_csv(dst_dir / "smiles.csv", unique_can)
        copy_metadata(src_dir, dst_dir)

        # optional: compute scaffold stats on the DEDUPED set (unique_can)
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
            "temperature": float(temp),
            "beams": beams,
            "pct_valid_total": summary["pct_valid_total"],
            "pct_unique_total": summary["pct_unique_total"],
            "pct_unique_within_valid": summary["pct_unique_within_valid"],
        }
        if args.compute_tanimoto:
            row["mean_scaffold_tanimoto"] = summary.get("mean_scaffold_tanimoto", 0.0)
            row["scaffold_diversity"] = summary.get("scaffold_diversity", 0.0)
        all_rows.append(row)

    # write aggregated summary.csv
    out_csv = dst_root / "summary.csv"
    with out_csv.open("w", newline="") as f:
        fieldnames = ["temperature", "beams", "pct_valid_total", "pct_unique_total", "pct_unique_within_valid"]
        if args.compute_tanimoto:
            fieldnames += ["mean_scaffold_tanimoto", "scaffold_diversity"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(all_rows, key=lambda x: (x["temperature"], x["beams"])):
            w.writerow(r)

    # ---- metrics PDF: 3x1 panels (valid / unique / unique within valid) ----
    set_pub_style()
    fig_m, axes_m = plt.subplots(3, 1, figsize=(5.0, 7.5), sharex=True)

    x1, y1, x250, y250 = collect_xy(all_rows, "pct_valid_total")
    add_series(axes_m[0], x1, y1, x250, y250, ylabel="% valid (total)", show_legend=True)

    x1, y1, x250, y250 = collect_xy(all_rows, "pct_unique_total")
    add_series(axes_m[1], x1, y1, x250, y250, ylabel="% unique (total)")

    x1, y1, x250, y250 = collect_xy(all_rows, "pct_unique_within_valid")
    add_series(axes_m[2], x1, y1, x250, y250, ylabel="% unique (within valid)", xlabel="temperature")

    fig_m.tight_layout()
    metrics_pdf = dst_root / args.metrics_pdf_name
    fig_m.savefig(metrics_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig_m)

    # ---- diversity PDF: 2x1 panels (mean Tanimoto / diversity) ----
    if args.compute_tanimoto:
        set_pub_style()
        fig_d, axes_d = plt.subplots(2, 1, figsize=(5.0, 5.0), sharex=True)

        x1, y1, x250, y250 = collect_xy(all_rows, "mean_scaffold_tanimoto")
        add_series(axes_d[0], x1, y1, x250, y250, ylabel="mean scaffold Tanimoto", show_legend=True)

        x1, y1, x250, y250 = collect_xy(all_rows, "scaffold_diversity")
        add_series(axes_d[1], x1, y1, x250, y250, ylabel="scaffold diversity", xlabel="temperature")

        fig_d.tight_layout()
        diversity_pdf = dst_root / args.diversity_pdf_name
        fig_d.savefig(diversity_pdf, format="pdf", bbox_inches="tight")
        plt.close(fig_d)

    print(f"Processed {len(all_rows)} folders → {dst_root}")
    print(f"Saved metrics panels → {metrics_pdf}")
    if args.compute_tanimoto:
        print(f"Saved diversity panels → {diversity_pdf}")

if __name__ == "__main__":

    """Example usage:
        python analyze_sweep_temp_beams.py \
        --input_root sweep_temp_beams \
        --output_root sweep_temp_beams_dv \
        --compute_tanimoto
    """
    main()
