#!/usr/bin/env python3
import os
import csv
import json
import argparse
import re
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")

FOLDER_RE = re.compile(r"^temp_(?P<temp>-?\d+(?:\.\d+)?)_beams_(?P<beams>\d+)$")

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

def canonicalize(smiles_list):
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

def write_summary_json(dst_dir: Path, total_rows, n_valid, n_unique):
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
    (dst_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

def main():
    ap = argparse.ArgumentParser(
        description="Deduplicate canonical SMILES per sweep folder, copy metadata, write summaries, and generate a 3x1 PDF."
    )
    ap.add_argument("--input_root", default="sweep_temp_beams", help="source root with temp_*_beams_* folders")
    ap.add_argument("--output_root", default="sweep_temp_beams_dv", help="destination root")
    ap.add_argument("--pdf_name", default="summary_plots.pdf", help="output PDF filename (saved under output_root)")
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

        raw = read_smiles_csv(smiles_path)
        total_rows = len(raw)
        valid_can = canonicalize(raw)
        unique_can = deduplicate_preserve_order(valid_can)

        dst_dir = dst_root / name
        write_smiles_csv(dst_dir / "smiles.csv", unique_can)
        copy_metadata(src_dir, dst_dir)
        summary = write_summary_json(dst_dir, total_rows, len(valid_can), len(unique_can))

        all_rows.append({
            "temperature": float(temp),
            "beams": beams,
            "pct_valid_total": summary["pct_valid_total"],
            "pct_unique_total": summary["pct_unique_total"],
            "pct_unique_within_valid": summary["pct_unique_within_valid"],
        })

    # write aggregated summary.csv
    out_csv = dst_root / "summary.csv"
    with out_csv.open("w", newline="") as f:
        fieldnames = ["temperature", "beams", "pct_valid_total", "pct_unique_total", "pct_unique_within_valid"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(all_rows, key=lambda x: (x["temperature"], x["beams"])):
            w.writerow(r)

    # ---- publication-ready 3x1 PDF (no titles), beams=1 in black, beams=250 in magenta ----
    # organize data
    # only plot beams 1 and 250; silently skip if one is missing
    def collect(series_key):
        # returns x (temps), y1 (beams=1), y250 (beams=250)
        temps = sorted({r["temperature"] for r in all_rows})
        b1 = {r["temperature"]: r[series_key] for r in all_rows if r["beams"] == 1}
        b250 = {r["temperature"]: r[series_key] for r in all_rows if r["beams"] == 250}
        x1 = [t for t in temps if t in b1]
        y1 = [b1[t] for t in x1]
        x250 = [t for t in temps if t in b250]
        y250 = [b250[t] for t in x250]
        return (x1, y1, x250, y250)

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

    fig, axes = plt.subplots(3, 1, figsize=(5.0, 7.5), sharex=True)

    # panel 1: % valid (of total)
    x1, y1, x250, y250 = collect("pct_valid_total")
    axes[0].plot(x1, y1, marker="o", color="black", label="beams=1")
    axes[0].plot(x250, y250, marker="o", color="magenta", label="beams=250")
    axes[0].set_ylabel("% valid (total)")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].legend(frameon=False, handlelength=2.5)

    # panel 2: % unique (of total)
    x1, y1, x250, y250 = collect("pct_unique_total")
    axes[1].plot(x1, y1, marker="o", color="black")
    axes[1].plot(x250, y250, marker="o", color="magenta")
    axes[1].set_ylabel("% unique (total)")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # panel 3: % unique (within valid)
    x1, y1, x250, y250 = collect("pct_unique_within_valid")
    axes[2].plot(x1, y1, marker="o", color="black")
    axes[2].plot(x250, y250, marker="o", color="magenta")
    axes[2].set_ylabel("% unique (within valid)")
    axes[2].set_xlabel("temperature")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    pdf_path = dst_root / args.pdf_name
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"processed {len(all_rows)} folders → {dst_root}")
    print(f"saved plot → {pdf_path}")

if __name__ == "__main__":
    main()
