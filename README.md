# De novo molecular generation with optical property preconditioning

This repository contains the GPT-2 sampling code for generating OLED candidate molecules and a browser for inspecting generated SMILES. The trained checkpoint and study data are available in the [Zenodo record](https://zenodo.org/records/23045084) (DOI: 10.5281/zenodo.23045084). The [TD-DFT evaluation workflow](https://github.com/the-matter-lab/OLED_accelerated_screening) is in a separate repository.

## Installation and model

Set up the environment with `bash installation.sh` (requires mamba). Download the trained model directly from Zenodo:

```bash
bash download_model.sh
```

The script saves `model_checkpoints/finetuned_coldstart_v1.ckpt` and verifies its SHA-256 checksum. It can be run from any directory and skips an existing checkpoint only when its checksum matches. The checkpoint is not stored in Git. The [model file on Zenodo](https://zenodo.org/records/23045084/files/finetuned_coldstart_v1.ckpt?download=1) can also be downloaded separately.

## Generate molecules

Run this command from the repository root after downloading the model:

```bash
python generate_smiles.py \
  --model_path model_checkpoints/finetuned_coldstart_v1.ckpt \
  --datapath generated_data/example_toluene \
  --strength 4 --absorption 0 --splitting 0 --rate 0 \
  --solvent toluene --num_return_sequences 250 --temperature 1.0
```

The script creates `smiles.csv` and `metadata.json` in the chosen `--datapath`. It uses a GPU when available and falls back to CPU after a CUDA out-of-memory error. `generated_data/` is ignored by Git.

| Flag | Meaning | Default |
| --- | --- | ---: |
| `--strength` | Excited-state strength index, 0–4 | 4 |
| `--absorption` | Absorption energy index, 0–4 | 0 |
| `--splitting` | Energy splitting index, 0–4 | 0 |
| `--rate` | Rate index, 0–3 | 0 |
| `--solvent` | Index 0–9, solvent name, or a SMILES from the table below | 0 |
| `--num_return_sequences` | Molecules to generate | 180 |
| `--num_beams` | Beam search width | 1 |
| `--temperature` | Sampling temperature | 0.8 |
| `--do_sample` / `--no-do_sample` | Enable or disable sampling | enabled |

Prompts are built automatically from `<strength0>`–`<strength4>`, `<absorption0>`–`<absorption4>`, `<splitting0>`–`<splitting4>`, `<rate0>`–`<rate3>`, and `<solvent0>`–`<solvent9>` tokens.

| Token | SMILES | Solvent |
| --- | --- | --- |
| `<solvent0>` | `ClCCl` | Dichloromethane |
| `<solvent1>` | `CC#N` | Acetonitrile |
| `<solvent2>` | `Cc1ccccc1` | Toluene |
| `<solvent3>` | `ClC(Cl)Cl` | Chloroform |
| `<solvent4>` | `C1COC1` | Oxetane |
| `<solvent5>` | `CO` | Methanol |
| `<solvent6>` | `CCO` | Ethanol |
| `<solvent7>` | `CS(C)=O` | DMSO |
| `<solvent8>` | `C1CCCCC1` | Cyclohexane |
| `<solvent9>` | `CN(C)C=O` | DMF |

## Preserved experiment data

Earlier generated results and paper figures remain tracked in `experiments/`, with their original files and internal folder structure preserved:

| Folder | Contents |
| --- | --- |
| `experiments/runs/` | Individual saved generation runs |
| `experiments/sweep_conditionings/` | Raw conditioning sweep |
| `experiments/sweep_conditionings_dv/` | Deduplicated conditioning sweep and metrics |
| `experiments/sweep_conditionings_global/` | Aggregated molecules, overlap, and charge analysis |
| `experiments/sweep_temp_beams/` | Raw temperature and beam-width sweep |
| `experiments/sweep_temp_beams_dv/` | Deduplicated temperature and beam-width sweep |
| `experiments/paper_figures/` | Figure source data, plotting code, and outputs |

New sweep runs and sweep analyses write under the ignored `generated_data/` directory by default, keeping the preserved results intact. The sweep analysis scripts accept `--input_root` and `--output_root` when you want to read an earlier sweep or choose another output location.
