import pandas as pd

df = pd.read_csv('../output/EA/TOP100.csv').drop_duplicates(subset='AA_sequence')
df['multi_tar'] = df['Target_LY6C1_Enri']*0.5 + df['Target_LY6A_Enri']*0.5
df.to_csv('../output/EA/TOP100.csv')

