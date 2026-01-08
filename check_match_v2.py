import pandas as pd
import os
from datetime import datetime

pairs_path = 'adapter_finetune/pairs_mri_with_pet_types_p1.csv'
upenn_path = 'adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv'
mri_pet_path = 'adapter_finetune/MRI_PET_IDs.csv'

pairs_df = pd.read_csv(pairs_path)
upenn_df = pd.read_csv(upenn_path)
mri_pet_df = pd.read_csv(mri_pet_path)

# Normalize columns
pairs_df.columns = [c.lower() for c in pairs_df.columns]
upenn_df.columns = [c.lower() for c in upenn_df.columns]
# MRI_PET_IDs columns are "Subject ID","Sex","Research Group","Study Date","Age","Modality","Description","Image ID"
# We can normalize them too or just access by name.
mri_pet_df.columns = [c.strip() for c in mri_pet_df.columns]

# Create a lookup for MRI_PET_IDs: Image ID -> Study Date
# Image ID in MRI_PET_IDs is int or string. Let's ensure string.
mri_pet_df['Image ID'] = mri_pet_df['Image ID'].astype(str)
image_id_to_date = dict(zip(mri_pet_df['Image ID'], mri_pet_df['Study Date']))

# Create a set of (ptid, examdate) from UPENN for fast lookup
upenn_keys = set(zip(upenn_df['ptid'], upenn_df['examdate']))

matches = []
mismatches = []

def convert_date(date_str):
    # MRI_PET_IDs date format: M/D/YYYY (e.g. 6/21/2017)
    # UPENN date format: YYYY-MM-DD (e.g. 2005-09-08)
    try:
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

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
        
    # Check against UPENN
    if (subject_id, study_date_formatted) in upenn_keys:
        matches.append({
            'subject_id': subject_id,
            'id_mri': id_mri_raw,
            'examdate': study_date_formatted
        })
    else:
        # Check if subject exists in UPENN
        subject_in_upenn = subject_id in upenn_df['ptid'].values
        mismatches.append({
            'subject_id': subject_id,
            'id_mri': id_mri_raw,
            'study_date': study_date_formatted,
            'found_in_upenn_by_id': subject_in_upenn,
            'reason': 'Date mismatch' if subject_in_upenn else 'Subject not found in UPENN'
        })

print(f"Matches found: {len(matches)}")
print(f"Mismatches found: {len(mismatches)}")

# Save results
matches_df = pd.DataFrame(matches)
mismatches_df = pd.DataFrame(mismatches)

matches_df.to_csv('matches_v2.csv', index=False)
mismatches_df.to_csv('mismatches_v2.csv', index=False)

if len(mismatches) > 0:
    print("\nSample mismatches (Date mismatch):")
    print(mismatches_df[mismatches_df['reason'] == 'Date mismatch'].head())
    
    print("\nSample mismatches (Image ID not found):")
    print(mismatches_df[mismatches_df['reason'] == 'Image ID not found in MRI_PET_IDs'].head())

# Add closest date logic
def find_closest_date(ptid, target_date_str, upenn_df):
    if ptid not in upenn_df['ptid'].values:
        return None, None
    
    subject_rows = upenn_df[upenn_df['ptid'] == ptid]
    if subject_rows.empty:
        return None, None
        
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        return None, None
        
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
            
    return closest_date, min_diff

# Update mismatches with closest date info
print("Calculating closest dates for mismatches...")
for m in mismatches:
    if m.get('reason') == 'Date mismatch':
        closest, diff = find_closest_date(m['subject_id'], m['study_date'], upenn_df)
        m['closest_upenn_date'] = closest
        m['days_diff'] = diff
    else:
        m['closest_upenn_date'] = None
        m['days_diff'] = None

# Save updated mismatches
mismatches_df = pd.DataFrame(mismatches)
mismatches_df.to_csv('mismatches_v2_with_diff.csv', index=False)
print("Saved detailed mismatch report to mismatches_v2_with_diff.csv")
print(mismatches_df[['subject_id', 'study_date', 'closest_upenn_date', 'days_diff']].head(10))
