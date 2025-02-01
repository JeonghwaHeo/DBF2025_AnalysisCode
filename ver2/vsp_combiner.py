import pandas as pd
import glob
import os

csv_files = glob.glob(r"data/aircraft_*.csv")
csv_files = [f for f in csv_files if "merged" not in f]
csv_files.sort() 

df_list = []

for i, csv_file in enumerate(csv_files):
    if i == 0:
        df_temp = pd.read_csv(csv_file, sep='|', encoding='utf-8')
    else:
        df_temp = pd.read_csv(csv_file, sep='|', skiprows=1, header=None, encoding='utf-8')
        df_temp.columns = df_list[0].columns
    
    df_list.append(df_temp)

df_merged = pd.concat(df_list, ignore_index=True)
df_merged.to_csv(r"data/aircraft.csv", sep='|', index=False, encoding='utf-8')
