#!/usr/bin/env bash
set -euo pipefail

PY=./generate_smiles.py          # path to your Python script
MODEL=./model_checkpoints/finetuned_coldstart_v1.ckpt
OUTROOT=./sweep_conditionings  # output root directory
RATE=0                           # keep rate fixed (change if needed)
SEQS=250                         # num_return_sequences

mkdir -p "$OUTROOT"

for S in {0..4}; do
  for A in {0..4}; do
    DPATH="${OUTROOT}/strength_${S}_absorption_${A}_solvent_0"
    mkdir -p "$DPATH"
    echo ">> strength=${S}, absorption=${A}, splitting=${A}, solvent=0 -> ${DPATH}"
    python "$PY" \
      --model_path "$MODEL" \
      --datapath "$DPATH" \
      --strength "$S" \
      --absorption "$A" \
      --splitting "$A" \
      --rate "$RATE" \
      --solvent 0 \
      --num_return_sequences "$SEQS" \
      --num_beams 1 \
      --do_sample \
      --temperature 1
  done
done

echo "done."

