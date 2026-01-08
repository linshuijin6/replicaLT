import pandas as pd
import os

pairs_path = 'adapter_finetune/pairs_mri_with_pet_types_p1.csv'
upenn_path = 'adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv'

pairs_df = pd.read_csv(pairs_path)
upenn_df = pd.read_csv(upenn_path)

# Normalize columns
pairs_df.columns = [c.lower() for c in pairs_df.columns]
upenn_df.columns = [c.lower() for c in upenn_df.columns]

# Create a set of (ptid, examdate) from UPENN
upenn_keys = set(zip(upenn_df['ptid'], upenn_df['examdate']))

mismatches = []

for idx, row in pairs_df.iterrows():
    subject_id = row['subject_id']
    age_mri = row['age_mri']
    
    # Check for exact match as requested
    if (subject_id, age_mri) not in upenn_keys:
        # Check if subject exists in UPENN at all
        subject_in_upenn = subject_id in upenn_df['ptid'].values
        
        mismatches.append({
            'subject_id': subject_id,
            'age_mri': age_mri,
            'found_in_upenn_by_id': subject_in_upenn,
            'reason': 'Type mismatch (float vs date)' if subject_in_upenn else 'Subject not found'
        })

mismatches_df = pd.DataFrame(mismatches)
output_path = 'mismatches_report.csv'
mismatches_df.to_csv(output_path, index=False)

print(f"Generated report with {len(mismatches_df)} mismatches at {output_path}")
print("Sample of mismatches:")
print(mismatches_df.head())
