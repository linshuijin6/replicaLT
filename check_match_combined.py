import pandas as pd
import os
from datetime import datetime

pairs_path = 'adapter_finetune/pairs_mri_with_pet_types_p1.csv'
upenn_path = 'adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv'
c2n_path = 'adapter_finetune/C2N_PRECIVITYAD2_PLASMA_29Dec2025.csv'
mri_pet_path = 'adapter_finetune/MRI_PET_IDs.csv'

pairs_df = pd.read_csv(pairs_path)
upenn_df = pd.read_csv(upenn_path)
c2n_df = pd.read_csv(c2n_path)
mri_pet_df = pd.read_csv(mri_pet_path)

# Normalize columns
pairs_df.columns = [c.lower() for c in pairs_df.columns]
upenn_df.columns = [c.lower() for c in upenn_df.columns]
c2n_df.columns = [c.lower() for c in c2n_df.columns]
mri_pet_df.columns = [c.strip() for c in mri_pet_df.columns]

# Create a lookup for MRI_PET_IDs: Image ID -> Study Date
mri_pet_df['Image ID'] = mri_pet_df['Image ID'].astype(str)
image_id_to_date = dict(zip(mri_pet_df['Image ID'], mri_pet_df['Study Date']))

matches = []
mismatches = []

TIME_WINDOW = 90

def convert_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

def find_closest_date_in_df(ptid, target_date_str, df, source_name):
    if ptid not in df['ptid'].values:
        return None, float('inf'), source_name
    
    subject_rows = df[df['ptid'] == ptid]
    if subject_rows.empty:
        return None, float('inf'), source_name
        
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        return None, float('inf'), source_name
        
    closest_date = None
    min_diff = float('inf')
    
    for date_str in subject_rows['examdate']:
        try:
            current_date = datetime.strptime(date_str, "%Y-%m-%d")
            diff = abs((current_date - target_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_date = date_str
        except ValueError:
            continue
            
    return closest_date, min_diff, source_name

print(f"Total pairs to check: {len(pairs_df)}")

for idx, row in pairs_df.iterrows():
    subject_id = row['subject_id']
    id_mri_raw = str(row['id_mri'])
    
    # Strip 'I' from id_mri
    if id_mri_raw.startswith('I'):
        image_id = id_mri_raw[1:]
    else:
        image_id = id_mri_raw
        
    # Find Study Date
    study_date_raw = image_id_to_date.get(image_id)
    
    if not study_date_raw:
        mismatches.append({
            'subject_id': subject_id,
            'id_mri': id_mri_raw,
            'reason': 'Image ID not found in MRI_PET_IDs'
        })
        continue
        
    study_date_formatted = convert_date(study_date_raw)
    if not study_date_formatted:
        mismatches.append({
            'subject_id': subject_id,
            'id_mri': id_mri_raw,
            'reason': f'Invalid date format in MRI_PET_IDs: {study_date_raw}'
        })
        continue
        
    # Find closest date in UPENN
    upenn_date, upenn_diff, _ = find_closest_date_in_df(subject_id, study_date_formatted, upenn_df, 'UPENN')
    
    # Find closest date in C2N
    c2n_date, c2n_diff, _ = find_closest_date_in_df(subject_id, study_date_formatted, c2n_df, 'C2N')
    
    # Compare and pick the best one
    best_date = None
    best_diff = float('inf')
    best_source = None
    
    # Logic: Pick the one with smaller diff. If equal, prefer UPENN (arbitrary choice, or could be C2N)
    if upenn_diff <= c2n_diff:
        best_date = upenn_date
        best_diff = upenn_diff
        best_source = 'UPENN'
    else:
        best_date = c2n_date
        best_diff = c2n_diff
        best_source = 'C2N'
        
    if best_date and best_diff <= TIME_WINDOW:
        matches.append({
            'subject_id': subject_id,
            'id_mri': id_mri_raw,
            'mri_study_date': study_date_formatted,
            'plasma_examdate': best_date,
            'days_diff': best_diff,
            'source': best_source
        })
    else:
        mismatches.append({
            'subject_id': subject_id,
            'id_mri': id_mri_raw,
            'study_date': study_date_formatted,
            'closest_date': best_date,
            'days_diff': best_diff if best_diff != float('inf') else None,
            'best_source_attempted': best_source,
            'reason': 'No match within time window' if best_date else 'Subject not found in Plasma tables'
        })

print(f"Matches found (within {TIME_WINDOW} days): {len(matches)}")
print(f"Mismatches found: {len(mismatches)}")

# Save results
matches_df = pd.DataFrame(matches)
mismatches_df = pd.DataFrame(mismatches)

matches_df.to_csv('matches_combined_90_days.csv', index=False)
mismatches_df.to_csv('mismatches_combined_90_days.csv', index=False)

if len(matches) > 0:
    print("\nSample matches:")
    print(matches_df.head())
    print("\nSource distribution:")
    print(matches_df['source'].value_counts())

if len(mismatches) > 0:
    print("\nSample mismatches:")
    print(mismatches_df.head())
