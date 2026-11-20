#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/raw}"
mkdir -p "$DATA_ROOT"

usage() {
  cat <<'EOF'
Usage: data/download_datasets.sh [dataset ...]

Datasets:
  mrl        MRL Eye Dataset (manual/Kaggle mirror)
  yawdd      YawDD (manual IEEE DataPort sign-up)
  fer2013    FER2013 via Kaggle
  dmd        DMD (manual Vicomtech access request)
  age_faces  Age Detection - Human Faces 18-60 (via Kaggle)
  syndd1     SynDD1 / PMC9730022 (manual Mendeley download)
  all        Run the Kaggle commands and print manual instructions

Environment:
  DATA_ROOT  Destination root (default: data/raw)
  KAGGLE_CONFIG_DIR  Kaggle CLI credential directory
EOF
}

need_kaggle() {
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "Missing Kaggle CLI. Install: python3 -m pip install kaggle" >&2
    return 1
  fi
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]] && [[ ! -f "${KAGGLE_CONFIG_DIR:-$HOME/.config/kaggle}/kaggle.json" ]]; then
    echo "Missing Kaggle credentials. Configure ~/.config/kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY." >&2
    return 1
  fi
}

download_age_faces() {
  need_kaggle
  local dest="$DATA_ROOT/age_detection_faces"
  mkdir -p "$dest"
  kaggle datasets download -d trainingdatapro/age-detection-human-faces-18-60-years -p "$dest" --unzip
}

download_fer2013() {
  need_kaggle
  local dest="$DATA_ROOT/fer2013"
  mkdir -p "$dest"
  kaggle datasets download -d msambare/fer2013 -p "$dest" --unzip
}

manual_mrl() {
  cat <<EOF
[MRL Eye Dataset] place an approved copy in $DATA_ROOT/mrl_eye
Reference: http://mrl.cs.vsb.cz/eyedataset
The original page requires a manual form/download. Do not commit the dataset.
EOF
}

manual_yawdd() {
  cat <<EOF
[YawDD] place the downloaded dataset in $DATA_ROOT/yawdd
Manual step: create a free IEEE DataPort account and request/download YawDD.
Reference: https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset
Then inspect the folder structure before training; layouts may vary.
EOF
}

manual_dmd() {
  cat <<EOF
[DMD] place approved RGB clips in $DATA_ROOT/dmd
Manual step: submit the access request at https://dmd.vicomtech.org/#Download
Access is subject to the DMD terms and the Creative Commons Attribution-
NonCommercial-NoDerivatives 4.0 License. Only RGB material may be available.
EOF
}

manual_syndd1() {
  cat <<EOF
[SynDD1] place the Mendeley download in $DATA_ROOT/syndd1
Manual step: download the DATASET LICENSE AGREEMENT and data from
https://data.mendeley.com/datasets/ptcp7rp3wb/4
Source article: https://pmc.ncbi.nlm.nih.gov/articles/PMC9730022/
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

for dataset in "$@"; do
  case "$dataset" in
    age_faces) download_age_faces ;;
    fer2013) download_fer2013 ;;
    mrl) manual_mrl ;;
    yawdd) manual_yawdd ;;
    dmd) manual_dmd ;;
    syndd1) manual_syndd1 ;;
    all)
      for item in age_faces fer2013 mrl yawdd dmd syndd1; do
        if [[ "$item" == mrl || "$item" == yawdd || "$item" == dmd || "$item" == syndd1 ]]; then
          "manual_$item"
        else
          "download_$item"
        fi
      done
      ;;
    -h|--help) usage ;;
    *) echo "Unknown dataset: $dataset" >&2; usage >&2; exit 2 ;;
  esac
done
