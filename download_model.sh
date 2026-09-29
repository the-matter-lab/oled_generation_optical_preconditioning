#!/usr/bin/env bash
set -euo pipefail

url='https://zenodo.org/records/23045084/files/finetuned_coldstart_v1.ckpt?download=1'
expected_sha256='4852012a84efa2a9719f045d52732a1f03238963b43a78b708c843aabcafc860'
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
model_dir="$script_dir/model_checkpoints"
model_path="$model_dir/finetuned_coldstart_v1.ckpt"
partial_path="$model_path.part"

for program in curl sha256sum; do
    if ! command -v "$program" >/dev/null 2>&1; then
        printf 'Required command is missing: %s\n' "$program" >&2
        exit 1
    fi
done

check_model() {
    printf '%s  %s\n' "$expected_sha256" "$1" | sha256sum --check --status
}

if [[ -e "$model_path" ]]; then
    if check_model "$model_path"; then
        printf 'Verified existing model: %s\n' "$model_path"
        exit 0
    fi
    printf 'Existing model does not match the Zenodo checkpoint: %s\n' "$model_path" >&2
    exit 1
fi

mkdir -p "$model_dir"
printf 'Downloading model from Zenodo...\n'
curl --fail --location --progress-bar --show-error --retry 3 --retry-delay 2 \
    --connect-timeout 20 --continue-at - --output "$partial_path" "$url"

if ! check_model "$partial_path"; then
    printf 'Downloaded checkpoint failed SHA-256 verification: %s\n' "$partial_path" >&2
    exit 1
fi

mv -- "$partial_path" "$model_path"
printf 'Verified model: %s\n' "$model_path"
