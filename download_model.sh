#!/bin/bash

# Create target directory
mkdir -p model_checkpoints

# Download model from Google Drive
echo "Downloading model checkpoint..."
gdown --id 1ssRT_iReTCoJf3UCGPJprxSVIvpPulLy -O model_checkpoints/finetuned_coldstart_v1.ckpt

echo "Download complete: model_checkpoints/finetuned_coldstart_v1.ckpt"

