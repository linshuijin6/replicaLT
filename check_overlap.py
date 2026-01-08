import pandas as pd

pairs_path = 'adapter_finetune/pairs_mri_with_pet_types_p1.csv'
mri_pet_path = 'adapter_finetune/MRI_PET_IDs.csv'

pairs_df = pd.read_csv(pairs_path)
mri_pet_df = pd.read_csv(mri_pet_path)

pairs_subjects = set(pairs_df['subject_id'])
mri_pet_subjects = set(mri_pet_df['Subject ID'])

intersection = pairs_subjects.intersection(mri_pet_subjects)

print(f"Pairs subjects: {len(pairs_subjects)}")
print(f"MRI_PET subjects: {len(mri_pet_subjects)}")
print(f"Intersection: {len(intersection)}")

if len(intersection) > 0:
    print("Sample intersection:", list(intersection)[:5])
else:
    print("No intersection found.")
    
# Also check Image IDs just in case
# Remove 'I' from pairs id_mri
pairs_ids = set(pairs_df['id_mri'].astype(str).str.replace('I', ''))
mri_pet_ids = set(mri_pet_df['Image ID'].astype(str))

id_intersection = pairs_ids.intersection(mri_pet_ids)
print(f"ID Intersection (stripping I): {len(id_intersection)}")
