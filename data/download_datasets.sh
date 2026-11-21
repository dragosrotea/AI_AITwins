#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT/data/raw"
MODEL_DIR="$ROOT/models"
MEDIAPIPE_MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

usage() {
  cat <<'EOF'
Usage: data/download_datasets.sh <dataset>

Datasets:
  mrl         Print manual MRL Eye Dataset download steps
  yawdd       Print IEEE DataPort sign-up and download steps
  fer2013     Download FER2013 using the configured Kaggle CLI
  dmd         Print DMD access steps and report existing local data
  mediapipe   Download the MediaPipe Face Landmarker task model
  all         Download FER2013 and MediaPipe; print manual-source steps
  status      Show which dataset folders currently contain files
EOF
}

manual_mrl() {
  cat <<EOF
MRL Eye Dataset is not downloaded automatically.
1. Review the dataset terms and current download instructions at:
   http://mrl.cs.vsb.cz/eyedataset
2. Download the dataset from the official source.
3. Extract it under: $RAW_DIR/mrl_eye
4. Inspect the extracted subject and eye-state naming before training.
EOF
}

manual_yawdd() {
  cat <<EOF
YawDD requires a free IEEE DataPort account; this script does not automate
account access or bypass DataPort terms.
1. Sign in and review the current dataset terms at:
   https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset
2. Download the permitted research files.
3. Extract them under: $RAW_DIR/yawdd
4. Review the participant information before sharing any screenshots.
EOF
}

manual_dmd() {
  cat <<EOF
DMD access must be requested manually; this script does not automate access.
1. Request access through: https://dmd.vicomtech.org/#Download
2. Review the supplied licence and access terms.
3. Extract the approved files under: $RAW_DIR/dmd
EOF
}

download_fer2013() {
  if [[ "${ACCEPT_FER2013_TERMS:-}" != "yes" ]]; then
    printf 'Review FER2013 terms, then rerun with ACCEPT_FER2013_TERMS=yes.\n' >&2
    return 1
  fi
  if ! command -v kaggle >/dev/null 2>&1; then
    printf 'Kaggle CLI is required. Install it and configure your account first.\n' >&2
    return 1
  fi
  cat <<EOF
Before continuing, verify that the FER2013 dataset licence and Kaggle terms
permit your intended research use:
https://www.kaggle.com/datasets/msambare/fer2013
EOF
  mkdir -p "$RAW_DIR/fer2013"
  kaggle datasets download \
    --dataset msambare/fer2013 \
    --path "$RAW_DIR/fer2013" \
    --unzip
}

download_mediapipe() {
  local destination="$MODEL_DIR/face_landmarker.task"
  local temporary="$destination.part"
  mkdir -p "$MODEL_DIR"
  if [[ -s "$destination" ]]; then
    printf 'MediaPipe model already exists: %s\n' "$destination"
    return
  fi
  if ! command -v curl >/dev/null 2>&1; then
    printf 'curl is required to download the MediaPipe model.\n' >&2
    return 1
  fi
  curl --fail --location --retry 3 "$MEDIAPIPE_MODEL_URL" --output "$temporary"
  mv -- "$temporary" "$destination"
  printf 'Downloaded MediaPipe model to %s\n' "$destination"
}

show_status() {
  local dataset
  for dataset in mrl_eye yawdd fer2013 dmd; do
    local directory="$RAW_DIR/$dataset"
    if [[ -d "$directory" ]] && find "$directory" -type f -print -quit | grep -q .; then
      printf '%-10s present (%s)\n' "$dataset" "$(du -sh "$directory" | cut -f1)"
    else
      printf '%-10s missing\n' "$dataset"
    fi
  done
  if [[ -s "$MODEL_DIR/face_landmarker.task" ]]; then
    printf 'mediapipe  present (%s)\n' "$(du -h "$MODEL_DIR/face_landmarker.task" | cut -f1)"
  else
    printf 'mediapipe  missing\n'
  fi
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  mrl) manual_mrl ;;
  yawdd) manual_yawdd ;;
  fer2013) download_fer2013 ;;
  dmd) manual_dmd ;;
  mediapipe) download_mediapipe ;;
  all)
    manual_mrl
    manual_yawdd
    manual_dmd
    download_fer2013
    download_mediapipe
    ;;
  status) show_status ;;
  -h|--help) usage ;;
  *)
    printf 'Unknown dataset: %s\n' "$1" >&2
    usage >&2
    exit 2
    ;;
esac
