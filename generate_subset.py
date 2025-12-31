import pandas as pd
import os

pairs_path = 'adapter_finetune/pairs_mri_with_pet_types_p1.csv'
matches_path = 'matches_combined_90_days.csv'
output_path = 'adapter_finetune/pairs_mri_with_pet_types_p1_subset.csv'

if not os.path.exists(pairs_path):
    print(f"File not found: {pairs_path}")
    exit(1)

if not os.path.exists(matches_path):
    print(f"File not found: {matches_path}")
    exit(1)

# Read files
pairs_df = pd.read_csv(pairs_path)
matches_df = pd.read_csv(matches_path)

print(f"Original pairs count: {len(pairs_df)}")
print(f"Matches count: {len(matches_df)}")

# Create a set of keys for filtering
# Using subject_id and id_mri to ensure uniqueness
# Ensure id_mri are strings
pairs_df['id_mri'] = pairs_df['id_mri'].astype(str)
matches_df['id_mri'] = matches_df['id_mri'].astype(str)

# Create a set of (subject_id, id_mri) from matches
match_keys = set(zip(matches_df['subject_id'], matches_df['id_mri']))

# Filter pairs
subset_df = pairs_df[pairs_df.apply(lambda row: (row['subject_id'], row['id_mri']) in match_keys, axis=1)]

print(f"Subset pairs count: {len(subset_df)}")

# Save subset
subset_df.to_csv(output_path, index=False)
print(f"Subset saved to {output_path}")

# Verify
print("\nFirst 5 rows of subset:")
print(subset_df.head())
