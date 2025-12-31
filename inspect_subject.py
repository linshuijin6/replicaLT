import pandas as pd
upenn_path = 'adapter_finetune/UPENN_PLASMA_FUJIREBIO_QUANTERIX_21Dec2025.csv'
upenn_df = pd.read_csv(upenn_path)
upenn_df.columns = [c.lower() for c in upenn_df.columns]
subject = '941_S_7085'
print(f"Entries for {subject} in UPENN:")
print(upenn_df[upenn_df['ptid'] == subject][['ptid', 'examdate']])
