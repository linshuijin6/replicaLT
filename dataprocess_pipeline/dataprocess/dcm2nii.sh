#!/usr/bin/env bash
set -euo pipefail

# Get script directory and add bin to PATH
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PATH="$DIR/bin:$PATH"

# Check for dcm2niix
if ! command -v dcm2niix &> /dev/null; then
    echo "Error: dcm2niix not found. Please install it or ensure it is in your PATH."
    exit 1
fi

IN="/mnt/nfsdata/nfsdata/ADNI/ADNI0103/ADNI"
OUT="/mnt/nfsdata/nfsdata/ADNI/ADNI0103/NIFTI"
find "$IN" -type f -name '*.dcm' -printf '%h\0' | sort -zu | \
  while IFS= read -r -d '' d; do
    rel="${d#"$IN"/}"
    out="$OUT/$rel"
    mkdir -p "$out"
    echo "Converting $d"
    dcm2niix -z y -b y -m y -f "%p_%s" -o "$out" "$d" || echo "Failed: $d"
  done