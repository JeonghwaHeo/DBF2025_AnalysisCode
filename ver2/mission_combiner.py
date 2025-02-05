import pandas as pd
import glob
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python3 mission_combiner.py <server number>")
    sys.exit(1)

server = sys.argv[1]

for j in range(2, 4): # Mission 2/3

    df_list = []

    csv_files = glob.glob(f"data/mission{j}_results_*.csv")
    csv_files.sort()

    for k, csv_file in enumerate(csv_files):
        if os.path.getsize(csv_file) == 0:  # 빈 파일이면 건너뛰기
            print(f"Skipping empty file: {csv_file}")
            continue

        if len(df_list) == 0:
            df_temp = pd.read_csv(csv_file, sep='|', encoding='utf-8')
        else:
            df_temp = pd.read_csv(csv_file, sep='|', header=0, encoding='utf-8')

        df_list.append(df_temp)

    if df_list:
        output_file = f"data/mission{j}_server{server}_results.csv"
        df_merged = pd.concat(df_list, ignore_index=True)
        df_merged.to_csv(output_file, sep='|', index=False, encoding='utf-8')
        print(f"Merged CSV file saved: {output_file}")
    else:
        print("No valid CSV files found. Merging skipped.")