#!/usr/bin/env bash
# Download the online retail dataset with the Kaggle CLI.
# Requires KAGGLE_API_TOKEN in the environment; the token is never written to disk.
set -euo pipefail

SLUG="thedevastator/online-retail-transaction-data"
DEST="data/raw"

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "Set KAGGLE_API_TOKEN first." >&2
  exit 1
fi

mkdir -p "$DEST"
kaggle datasets download -d "$SLUG" -p "$DEST" --unzip
ls -l "$DEST"
