"""
Check PET AV1451 dose information from DICOM headers across all subjects.
Structure: ADNI0103/ADNI/{subj}/{series_name*AV1451*}/{date}/{image_id}/*.dcm
Uses stop_before_pixels=True for speed.
"""
import os
import csv
import pydicom
import collections
from pathlib import Path

BASE_DIR = "/mnt/nfsdata/nfsdata/ADNI/ADNI0103/ADNI"
OUTPUT_CSV = "/mnt/nfsdata/nfsdata/lsj.14/replicaLT/av1451_dose_report.csv"

def get_rp_field(ds, field):
    try:
        return getattr(ds.RadiopharmaceuticalInformationSequence[0], field, None)
    except (AttributeError, IndexError):
        return None

def process_series_dir(series_dir):
    dcm_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
    if not dcm_files:
        return None
    dcm_path = os.path.join(series_dir, sorted(dcm_files)[0])
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    except Exception as e:
        return {"error": str(e), "dcm_path": dcm_path}
    return {
        "units":              str(getattr(ds, 'Units', '')),
        "rescale_slope":      str(getattr(ds, 'RescaleSlope', '')),
        "rescale_type":       str(getattr(ds, 'RescaleType', '')),
        "decay_correction":   str(getattr(ds, 'DecayCorrection', '')),
        "decay_factor":       str(getattr(ds, 'DecayFactor', '')),
        "dose_cal_factor":    str(getattr(ds, 'DoseCalibrationFactor', '')),
        "patient_weight_kg":  str(getattr(ds, 'PatientWeight', '')),
        "radiopharmaceutical":    str(get_rp_field(ds, 'Radiopharmaceutical') or ''),
        "total_dose_bq":          str(get_rp_field(ds, 'RadionuclideTotalDose') or ''),
        "injection_start_time":   str(get_rp_field(ds, 'RadiopharmaceuticalStartTime') or ''),
        "half_life_s":            str(get_rp_field(ds, 'RadionuclideHalfLife') or ''),
        "positron_fraction":      str(get_rp_field(ds, 'RadionuclidePositronFraction') or ''),
        "acquisition_date":       str(getattr(ds, 'AcquisitionDate', '')),
        "series_description":     str(getattr(ds, 'SeriesDescription', '')),
        "study_description":      str(getattr(ds, 'StudyDescription', '')),
        "num_dcm_files":          len(dcm_files),
        "error":                  "",
        "dcm_path":               dcm_path,
    }

rows = []
subjects = sorted(os.listdir(BASE_DIR))
print(f"Total subjects: {len(subjects)}", flush=True)

for subj_idx, subj in enumerate(subjects):
    subj_dir = os.path.join(BASE_DIR, subj)
    if not os.path.isdir(subj_dir):
        continue
    for series_name in os.listdir(subj_dir):
        if 'AV1451' not in series_name:
            continue
        series_base = os.path.join(subj_dir, series_name)
        if not os.path.isdir(series_base):
            continue
        for date_entry in os.listdir(series_base):
            date_dir = os.path.join(series_base, date_entry)
            if not os.path.isdir(date_dir):
                continue
            for img_id in os.listdir(date_dir):
                img_dir = os.path.join(date_dir, img_id)
                if not os.path.isdir(img_dir):
                    continue
                info = process_series_dir(img_dir)
                if info is None:
                    continue
                row = {"subject": subj, "series": series_name, "date": date_entry, "image_id": img_id}
                row.update(info)
                rows.append(row)

    if (subj_idx + 1) % 200 == 0:
        print(f"  [{subj_idx+1}/{len(subjects)}] {len(rows)} series so far...", flush=True)

# Write CSV
fieldnames = [
    "subject", "series", "date", "image_id",
    "units", "rescale_slope", "rescale_type",
    "decay_correction", "decay_factor", "dose_cal_factor",
    "patient_weight_kg", "radiopharmaceutical",
    "total_dose_bq", "injection_start_time", "half_life_s", "positron_fraction",
    "acquisition_date", "series_description", "study_description",
    "num_dcm_files", "error", "dcm_path",
]
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone. {len(rows)} series written to {OUTPUT_CSV}", flush=True)

# Summary stats
print("\n=== Summary ===")
print(f"Total series: {len(rows)}")

units_ctr = collections.Counter(r.get('units', '') for r in rows)
print(f"\nUnits distribution:")
for k, v in sorted(units_ctr.items(), key=lambda x: -x[1]):
    print(f"  '{k}': {v}")

dose_present = sum(1 for r in rows if r.get('total_dose_bq') not in ('', 'None', None))
print(f"\nTotal dose present: {dose_present}/{len(rows)}")

inj_present = sum(1 for r in rows if r.get('injection_start_time') not in ('', 'None', None))
print(f"Injection time present: {inj_present}/{len(rows)}")

weight_present = sum(1 for r in rows if r.get('patient_weight_kg') not in ('', 'None', None))
print(f"Patient weight present: {weight_present}/{len(rows)}")

decay_zero = sum(1 for r in rows if r.get('decay_factor', '') in ('0.00000', '0', '0.0'))
print(f"DecayFactor = 0: {decay_zero}/{len(rows)}")

study_ctr = collections.Counter(r.get('study_description', '') for r in rows)
print(f"\nStudy description distribution:")
for k, v in sorted(study_ctr.items(), key=lambda x: -x[1]):
    print(f"  '{k}': {v}")

errors = [r for r in rows if r.get('error')]
print(f"\nErrors: {len(errors)}")
if errors:
    for r in errors[:5]:
        print(f"  {r['subject']}: {r['error']}")
