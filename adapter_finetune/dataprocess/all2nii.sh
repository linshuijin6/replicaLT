#!/usr/bin/env bash
set -euo pipefail

IN="/mnt/nfsdata/nfsdata/ADNI/ADNI0103/ADNI"
OUT="/mnt/nfsdata/nfsdata/ADNI/ADNI0103/NIFTI"

mkdir -p "$OUT"

# trova tutte le directory che contengono almeno un file DICOM
# (molti ADNI non hanno estensione .dcm, quindi NON cerchiamo *.dcm)
mapfile -t DIRS < <(find "$IN" -type f \
  \( -iname "*.dcm" -o -iname "*.ima" -o -iname "*.dicom" -o -true \) \
  -print0 | xargs -0 -n1 dirname | sort -u)

echo "[INFO] Candidate dirs: ${#DIRS[@]}"

for d in "${DIRS[@]}"; do
  # controlla se la cartella sembra davvero DICOM: basta che dcm2niix veda qualcosa
  # creiamo una cartella output “specchio” rispetto a IN
  rel="${d#"$IN"/}"
  outdir="$OUT/$rel"
  mkdir -p "$outdir"

  echo "==> Converting: $d"
  # -z y  -> .nii.gz
  # -b y  -> salva anche JSON (metadati) (non fa male, utile dopo)
  # -m y  -> merge di slice 2D se necessario
  # -f ...-> nome basato su series desc + series number (abbastanza stabile)
  dcm2niix -z y -b y -m y -o "$outdir" -f "%p_%s" "$d" >/dev/null 2>&1 || true
done

echo "[DONE] Output in: $OUT"
