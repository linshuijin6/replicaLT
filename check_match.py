import pandas as pd
import os

pairs_path = 'adapter_finetune/pairs_mri_with_pet_types_p1.csv'
upenn_path = 'adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv'

if not os.path.exists(pairs_path):
    print(f"File not found: {pairs_path}")
    exit()
if not os.path.exists(upenn_path):
    print(f"File not found: {upenn_path}")
    exit()

pairs_df = pd.read_csv(pairs_path)
upenn_df = pd.read_csv(upenn_path)

# Normalize columns
pairs_df.columns = [c.lower() for c in pairs_df.columns]
upenn_df.columns = [c.lower() for c in upenn_df.columns]

print("Pairs columns:", pairs_df.columns.tolist())
print("UPENN columns:", upenn_df.columns.tolist())

# Check sample values
print("\nSample pairs data:")
print(pairs_df[['subject_id', 'age_mri']].head())
print("\nSample UPENN data:")
print(upenn_df[['ptid', 'examdate']].head())

# Perform check
mismatches = []
matches = []

# Create a set of (ptid, examdate) from UPENN for fast lookup
# Note: examdate in UPENN is string YYYY-MM-DD. age_mri in pairs is float.
# They will likely not match.
upenn_keys = set(zip(upenn_df['ptid'], upenn_df['examdate']))

for idx, row in pairs_df.iterrows():
    subject_id = row['subject_id']
    age_mri = row['age_mri']
    
    # The user asked to match subject_id + age_mri with PTID + EXAMDATE
    # We check if (subject_id, age_mri) exists in upenn_keys
    # We need to be careful about types.
    
    # Try direct match
    if (subject_id, age_mri) in upenn_keys:
        matches.append(row)
    else:
        # Check if subject exists at least
        subject_exists = subject_id in upenn_df['ptid'].values
        mismatches.append({
            'subject_id': subject_id,
            'age_mri': age_mri,
            'subject_found_in_upenn': subject_exists
        })

print(f"\nTotal pairs: {len(pairs_df)}")
print(f"Matches found: {len(matches)}")
print(f"Mismatches found: {len(mismatches)}")

if len(mismatches) > 0:
    print("\nFirst 10 mismatches:")
    for m in mismatches[:10]:
        print(m)

# Check if any age_mri looks like a date string?
print("\nChecking age_mri types:")
print(pairs_df['age_mri'].apply(type).value_counts())

# Check specific subject
subject = '941_S_7085'
print(f"\nEntries for {subject} in UPENN:")
print(upenn_df[upenn_df['ptid'] == subject][['ptid', 'examdate']])
