"""
PET injection dose analysis for AV1451 (TAU), AV45, and FDG.

CSV dose column mapping (confirmed via ADNI data dictionary / AMYMETA table):
  DOSETAU      → AV1451 / Flortaucipir (TAU PET)
  DOSEFLRBTPR  → Florbetapir / AV-45 / Amyvid (Amyloid PET, from AMYMETA table)
                 Covers 2016–2026, N=1460. PRIMARY field for AV45 Florbetapir.
  PMFDGDOS     → Florbetapir / AV-45 (from AV45META table, older ADNI1/2 scans)
                 Covers 2009–2017, N=812. COMPLEMENTARY to DOSEFLRBTPR (mutually
                 exclusive: same row never has both filled).
  DOSEFLRBTBN  → Florbetaben / NeuraCeq (different amyloid tracer, NOT AV45)
                 Covers 2016–2026, N=861.
  FDGDOS       → FDG (Glucose metabolism PET)

  AV45_FP dose = DOSEFLRBTPR if non-empty, else PMFDGDOS (merged Florbetapir)
  AV45_FBN dose = DOSEFLRBTBN (Florbetaben)

Strategy:
  Primary table: pairs_180d_dx_plasma_90d_matched.csv
    - 1360 rows with PTID, EXAMDATE (real scan date), id_fdg/id_av45/id_av1451
  Dose table:  Images_PET__ADNI3_4___PLASMA_with_T1_My_Table_15Mar2026.csv
    - per-(subject, visit) dose records
  Matching: for each pairs row, find the dose row with the same PTID
            and estimated_scan_date (entry_date + visit_months) closest
            to EXAMDATE, within ±180 days.
  Output: pairs CSV enriched with dose columns + low-dose flags.
"""
import os
import csv
import re
import collections
from datetime import datetime, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR  = "/mnt/nfsdata/nfsdata/lsj.14/replicaLT"

PAIRS_CSV = os.path.join(WORK_DIR, "adapter_v2", "data_csv",
            "pairs_180d_dx_plasma_90d_matched.csv")
DOSE_CSV  = os.path.join(WORK_DIR, "analysis",
            "Images_PET__ADNI3_4___PLASMA_with_T1_My_Table_15Mar2026.csv")
OUTPUT_CSV = os.path.join(WORK_DIR, "analysis", "pet_dose_analysis_report.csv")

# Tolerance for date matching: dose row estimated_date vs pairs EXAMDATE
DATE_TOLERANCE_DAYS = 180

# Low-dose thresholds (mCi) per tracer
LOW_DOSE_THRESH = {
    "TAU":      9.0,   # AV1451/Flortaucipir; ADNI protocol ~10 mCi
    "AV45_FBN": 7.5,   # Florbetaben (NeuraCeq, DOSEFLRBTBN); standard ~8-9 mCi
    "AV45_FP":  7.5,   # Florbetapir (Amyvid/AV-45, PMFDGDOS); standard ~8-10 mCi
    "FDG":      4.5,   # FDG (FDGDOS); ADNI4 protocol ~5 mCi
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def visit_to_months(visit):
    """Convert visit code to integer months from baseline. Returns None if unknown."""
    v = visit.strip().lower()
    if v in ('bl', 'sc', 'nv', 'scmri', 'aut'):
        return 0
    m = re.match(r'^m(\d+)$', v)
    return int(m.group(1)) if m else None


def estimate_scan_date(entry_date_str, visit):
    """Estimate scan date from entry_date (YYYY-MM-DD) + visit offset."""
    months = visit_to_months(visit)
    if months is None:
        return None
    try:
        ed = datetime.strptime(entry_date_str.strip(), '%Y-%m-%d')
    except ValueError:
        return None
    return ed + timedelta(days=months * 30.44)


def load_dose_csv(path):
    """Load dose CSV, handling the duplicate SCANQLTY column."""
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        seen = {}
        clean_header = []
        for col in header:
            if col in seen:
                seen[col] += 1
                clean_header.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                clean_header.append(col)
        for row_vals in reader:
            rows.append(dict(zip(clean_header, row_vals)))
    return rows


def build_dose_index(dose_rows):
    """
    Index dose records by subject_id.
    Each entry: {estimated_dt, visit, entry_date, dose_tau, dose_fbn, dose_av45fp, dose_fdg}
    AV45_FP = DOSEFLRBTPR (primary, 2016–2026) or PMFDGDOS (fallback, 2009–2017).
    Only rows with at least one non-empty dose field are included.
    """
    idx = collections.defaultdict(list)
    for r in dose_rows:
        subj  = r.get('subject_id', '').strip()
        visit = r.get('visit', '').strip()
        entry = r.get('entry_date', '').strip()
        dose_tau    = r.get('DOSETAU', '').strip()
        dose_fbn    = r.get('DOSEFLRBTBN', '').strip()
        # Merge Florbetapir sources: DOSEFLRBTPR (primary) or PMFDGDOS (fallback)
        dose_av45fp = r.get('DOSEFLRBTPR', '').strip() or r.get('PMFDGDOS', '').strip()
        dose_fdg    = r.get('FDGDOS', '').strip()
        if not any([dose_tau, dose_fbn, dose_av45fp, dose_fdg]):
            continue
        est_dt = estimate_scan_date(entry, visit)
        if est_dt is None:
            continue
        idx[subj].append({
            'est_dt':      est_dt,
            'visit':       visit,
            'entry_date':  entry,
            'dose_tau':    dose_tau,
            'dose_fbn':    dose_fbn,
            'dose_av45fp': dose_av45fp,
            'dose_fdg':    dose_fdg,
        })
    return idx


def find_best_dose(subj, exam_dt, dose_index):
    """
    Find the dose record for subj whose estimated_scan_date is closest
    to exam_dt, within DATE_TOLERANCE_DAYS. Returns (record, day_diff) or (None, None).
    """
    candidates = dose_index.get(subj, [])
    if not candidates or exam_dt is None:
        return None, None
    best = min(candidates, key=lambda c: abs((c['est_dt'] - exam_dt).days))
    diff = abs((best['est_dt'] - exam_dt).days)
    if diff <= DATE_TOLERANCE_DAYS:
        return best, diff
    return None, diff   # return diff even when outside tolerance, for reporting


def safe_float(s):
    try:
        return float(s) if s else None
    except ValueError:
        return None


def low_dose_flag(val, tracer):
    if val is None:
        return ''
    return '1' if val < LOW_DOSE_THRESH.get(tracer, 0) else '0'


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.join(WORK_DIR, "analysis"), exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("[1/3] Loading input tables ...")
    with open(PAIRS_CSV, newline='') as f:
        pairs_rows = list(csv.DictReader(f))
    print(f"  Pairs CSV: {len(pairs_rows)} rows from {PAIRS_CSV}")

    dose_rows = load_dose_csv(DOSE_CSV)
    print(f"  Dose CSV:  {len(dose_rows)} rows from {DOSE_CSV}")

    # ── Build dose index ──────────────────────────────────────────────────────
    print("\n[2/3] Building dose index and matching ...")
    dose_idx = build_dose_index(dose_rows)
    subjects_with_dose = len(dose_idx)
    print(f"  Subjects with ≥1 dose record: {subjects_with_dose}")

    # ── Match each pairs row to a dose record ─────────────────────────────────
    output_rows = []
    dose_found = 0
    outside_tol = 0
    no_dose_subj = 0

    for pr in pairs_rows:
        ptid     = pr['PTID'].strip()
        examdate = pr.get('EXAMDATE', '').strip()
        id_tau   = pr.get('id_av1451', '').strip()
        id_av45  = pr.get('id_av45', '').strip()
        id_fdg   = pr.get('id_fdg', '').strip()

        try:
            exam_dt = datetime.strptime(examdate, '%Y-%m-%d')
        except ValueError:
            exam_dt = None

        best, day_diff = find_best_dose(ptid, exam_dt, dose_idx)

        if best is None and day_diff is None:
            no_dose_subj += 1
        elif best is None:
            outside_tol += 1
        else:
            dose_found += 1

        # Dose values (None if no match or field empty)
        d_tau    = safe_float(best['dose_tau'])    if best else None
        d_fbn    = safe_float(best['dose_fbn'])    if best else None
        d_av45fp = safe_float(best['dose_av45fp']) if best else None
        d_fdg    = safe_float(best['dose_fdg'])    if best else None

        row = {
            # ── Original pairs fields ──────────────────────────────────────
            'PTID':            ptid,
            'EXAMDATE':        examdate,
            'DIAGNOSIS':       pr.get('DIAGNOSIS', ''),
            'id_mri':          pr.get('id_mri', ''),
            'id_fdg':          id_fdg,
            'id_av45':         id_av45,
            'id_av1451':       id_tau,
            # plasma fields pass-through
            'pT217_F':         pr.get('pT217_F', ''),
            'AB42_F':          pr.get('AB42_F', ''),
            'AB40_F':          pr.get('AB40_F', ''),
            'AB42_AB40_F':     pr.get('AB42_AB40_F', ''),
            'pT217_AB42_F':    pr.get('pT217_AB42_F', ''),
            'NfL_Q':           pr.get('NfL_Q', ''),
            'GFAP_Q':          pr.get('GFAP_Q', ''),
            'plasma_source':   pr.get('plasma_source', ''),
            'plasma_date':     pr.get('plasma_date', ''),
            'plasma_date_diff':pr.get('plasma_date_diff', ''),
            # ── Dose match metadata ────────────────────────────────────────
            'dose_match_visit':    best['visit']                       if best else '',
            'dose_match_est_date': best['est_dt'].strftime('%Y-%m-%d') if best else '',
            'dose_match_day_diff': str(day_diff)                       if day_diff is not None else '',
            # ── TAU dose ──────────────────────────────────────────────────
            'dose_tau_mci':        str(d_tau)  if d_tau  is not None else '',
            'low_dose_tau':        low_dose_flag(d_tau,    'TAU')      if id_tau  else '',
            # ── AV45 Florbetaben (DOSEFLRBTBN) ────────────────────────────
            'dose_av45fbn_mci':    str(d_fbn)  if d_fbn  is not None else '',
            'low_dose_av45fbn':    low_dose_flag(d_fbn,   'AV45_FBN') if id_av45 else '',
            # ── AV45 Florbetapir (PMFDGDOS) ───────────────────────────────
            'dose_av45fp_mci':     str(d_av45fp) if d_av45fp is not None else '',
            'low_dose_av45fp':     low_dose_flag(d_av45fp, 'AV45_FP') if id_av45 else '',
            # ── FDG dose ──────────────────────────────────────────────────
            'dose_fdg_mci':        str(d_fdg)  if d_fdg  is not None else '',
            'low_dose_fdg':        low_dose_flag(d_fdg,   'FDG')      if id_fdg  else '',
        }
        output_rows.append(row)

    print(f"  Dose matched (within {DATE_TOLERANCE_DAYS}d): {dose_found}/{len(pairs_rows)}")
    print(f"  Outside tolerance: {outside_tol}")
    print(f"  No dose data for subject: {no_dose_subj}")

    # ── Write output ──────────────────────────────────────────────────────────
    print(f"\n[3/3] Writing output ...")
    fieldnames = [
        'PTID', 'EXAMDATE', 'DIAGNOSIS', 'id_mri', 'id_fdg', 'id_av45', 'id_av1451',
        'pT217_F', 'AB42_F', 'AB40_F', 'AB42_AB40_F', 'pT217_AB42_F',
        'NfL_Q', 'GFAP_Q', 'plasma_source', 'plasma_date', 'plasma_date_diff',
        'dose_match_visit', 'dose_match_est_date', 'dose_match_day_diff',
        'dose_tau_mci',    'low_dose_tau',
        'dose_av45fbn_mci','low_dose_av45fbn',
        'dose_av45fp_mci', 'low_dose_av45fp',
        'dose_fdg_mci',    'low_dose_fdg',
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(output_rows)
    print(f"  -> {OUTPUT_CSV}")

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n=== Dose Statistics (pairs CSV scope) ===")

    def stats(vals, tracer):
        vals = [v for v in vals if v is not None]
        if not vals:
            return
        low = sum(1 for v in vals if v < LOW_DOSE_THRESH[tracer])
        print(f"\n  {tracer}  (N with dose={len(vals)}):")
        print(f"    range  : {min(vals):.2f} – {max(vals):.2f} mCi")
        print(f"    mean   : {sum(vals)/len(vals):.2f} mCi")
        import statistics
        print(f"    median : {statistics.median(vals):.2f} mCi")
        print(f"    std    : {statistics.stdev(vals):.2f} mCi" if len(vals) > 1 else "")
        print(f"    low-dose (< {LOW_DOSE_THRESH[tracer]} mCi): {low}/{len(vals)} ({100*low/len(vals):.1f}%)")
        if low:
            low_ids = sorted({r['PTID'] for r in output_rows
                              if safe_float(r.get(f'dose_{tracer.lower().replace("45","45")}_mci',''))
                              and safe_float(r.get(f'dose_{tracer.lower().replace("45","45")}_mci','')) < LOW_DOSE_THRESH[tracer]})
            print(f"    low-dose PTIDs: {', '.join(low_ids[:20])}"
                  + (' ...' if len(low_ids) > 20 else ''))

    # Use consistent key names to print stats
    tau_vals    = [safe_float(r['dose_tau_mci'])     for r in output_rows if r['id_av1451']]
    fbn_vals    = [safe_float(r['dose_av45fbn_mci']) for r in output_rows if r['id_av45']]
    fp_vals     = [safe_float(r['dose_av45fp_mci'])  for r in output_rows if r['id_av45']]
    fdg_vals    = [safe_float(r['dose_fdg_mci'])     for r in output_rows if r['id_fdg']]

    print(f"\n  Pairs with TAU image: {sum(1 for r in output_rows if r['id_av1451'])}")
    print(f"  Pairs with AV45 image: {sum(1 for r in output_rows if r['id_av45'])}")
    print(f"  Pairs with FDG image: {sum(1 for r in output_rows if r['id_fdg'])}")

    import statistics as _stats

    for tracer, vals in [('TAU', tau_vals), ('AV45_FBN', fbn_vals),
                         ('AV45_FP', fp_vals), ('FDG', fdg_vals)]:
        vals_clean = [v for v in vals if v is not None]
        if not vals_clean:
            print(f"\n  {tracer}: no dose data")
            continue
        low = sum(1 for v in vals_clean if v < LOW_DOSE_THRESH[tracer])
        print(f"\n  {tracer}  (N with dose={len(vals_clean)} / {len(vals)} with image):")
        print(f"    range  : {min(vals_clean):.2f} – {max(vals_clean):.2f} mCi")
        print(f"    mean±sd: {sum(vals_clean)/len(vals_clean):.2f} ± "
              f"{(_stats.stdev(vals_clean) if len(vals_clean)>1 else 0):.2f} mCi")
        print(f"    median : {_stats.median(vals_clean):.2f} mCi")
        print(f"    low-dose (< {LOW_DOSE_THRESH[tracer]} mCi): "
              f"{low}/{len(vals_clean)} ({100*low/len(vals_clean):.1f}%)")

    print("\nDone.")


if __name__ == '__main__':
    main()
