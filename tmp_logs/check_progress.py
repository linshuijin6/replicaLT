import csv
from pathlib import Path
root = Path('/mnt/nfsdata/nfsdata/ADNI/ADNI0103')
chunks = Path('/home/ssddata/linshuijin/replicaLT/adapter_finetune/data_csv/chunks0106')

def read_pairs(path: Path):
    rows = []
    exp_pet = 0
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row.get('PTID','').strip(), row.get('id_mri','').strip()))
            for col in ('id_fdg','id_av45','id_av1451'):
                v = (row.get(col,'') or '').strip()
                if v:
                    exp_pet += 1
    return rows, exp_pet

def main():
    for i in range(1,7):
        base = root / f'logs_pipeline0106_part{i}'
        pairs_csv = chunks / f'pairs0106_part{i}.csv'
        print('checking', i, pairs_csv.exists(), base.exists())
        if not pairs_csv.exists():
            print(f'== shard {i} ==\nmissing pairs csv\n')
            continue
        pairs, exp_pet = read_pairs(pairs_csv)
        total_rows = len(pairs)

        mri_ok = mri_run_ok = mri_reuse_ok = 0
        mri_fail_only = 0
        mri_log = base / 'pipeline_mri.csv'
        if mri_log.exists():
            state = {}
            with open(mri_log, newline='', encoding='utf-8') as f:
                r = csv.reader(f); _ = next(r, None)
                for row in r:
                    subj = row[0].strip() if len(row)>0 else ''
                    mid = row[1].strip() if len(row)>1 else ''
                    act = row[2].strip() if len(row)>2 else ''
                    status = row[4].strip() if len(row)>4 else ''
                    key = (subj, mid)
                    st = state.get(key, {'ok':False,'fail':False,'run':False,'reuse':False})
                    if status == 'OK':
                        st['ok'] = True
                        if act == 'run': st['run'] = True
                        if ('copy' in act) or ('reuse_target' in act): st['reuse'] = True
                    if status == 'FAIL':
                        st['fail'] = True
                    state[key] = st
            for st in state.values():
                if st['ok']:
                    mri_ok += 1
                    if st['run']: mri_run_ok += 1
                    if st['reuse']: mri_reuse_ok += 1
                elif st['fail']:
                    mri_fail_only += 1

        pet_ok = pet_fail = pet_skip = 0
        pet_log = base / 'pipeline_pet.csv'
        if pet_log.exists():
            with open(pet_log, newline='', encoding='utf-8') as f:
                r = csv.reader(f); _ = next(r, None)
                for row in r:
                    status = row[5].strip() if len(row)>5 else ''
                    if status == 'OK': pet_ok += 1
                    elif status == 'FAIL': pet_fail += 1
                    elif status == 'SKIP': pet_skip += 1

        print(f'== shard {i} ==')
        print(f'rows_total: {total_rows}')
        print(f'MRI OK pairs: {mri_ok} (run {mri_run_ok}, reuse {mri_reuse_ok}), FAIL-only: {mri_fail_only}')
        print(f'PET OK entries: {pet_ok} / expected: {exp_pet} (skip {pet_skip}, fail {pet_fail})')
        if total_rows:
            print(f'MRI done %: {round(100.0*mri_ok/total_rows,2)}')
        if exp_pet:
            print(f'PET done % by entries: {round(100.0*pet_ok/exp_pet,2)}')
        print()

if __name__ == '__main__':
    main()
