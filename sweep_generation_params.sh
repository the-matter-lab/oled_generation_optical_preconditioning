#!/usr/bin/env bash
set -euo pipefail

MODEL=./model_checkpoints/finetuned_coldstart_v1.ckpt
OUTROOT=./sweep_params

STRENGTH=4
ABSORPTION=0
SPLITTING=0
SOLVENT=0
NSEQ=250   # number of sequences to generate

mkdir -p "$OUTROOT"
export LC_NUMERIC=C   # ensure decimal point

for T in $(seq 0.5 0.1 2.0); do
  T_FMT=$(printf "%.1f" "$T")
  for BEAMS in 1 250; do
    DPATH="${OUTROOT}/temp_${T_FMT}_beams_${BEAMS}"
    mkdir -p "$DPATH"
    echo ">> T=${T_FMT}, beams=${BEAMS}, nseq=${NSEQ} -> ${DPATH}"
    python generate_smiles.py \
      --model_path "$MODEL" \
      --datapath "$DPATH" \
      --strength "$STRENGTH" \
      --absorption "$ABSORPTION" \
      --splitting "$SPLITTING" \
      --solvent "$SOLVENT" \
      --num_return_sequences "$NSEQ" \
      --num_beams "$BEAMS" \
      --temperature "$T_FMT" \
      --do_sample
  done
done

echo "sweep complete."
