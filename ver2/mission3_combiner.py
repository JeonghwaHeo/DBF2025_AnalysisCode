import pandas as pd
import glob
import os

csv_files = glob.glob(r"data/mission3_results_*.csv")
csv_files.sort()

df_list = []

for i, csv_file in enumerate(csv_files):
    if os.path.getsize(csv_file) == 0:  # 빈 파일이면 건너뛰기
        print(f"Skipping empty file: {csv_file}")
        continue

    if i == 0:
        df_temp = pd.read_csv(csv_file, sep='|', encoding='utf-8')
    else:
        df_temp = pd.read_csv(csv_file, sep='|', skiprows=1, header=None, encoding='utf-8')
        df_temp.columns = df_list[0].columns  # 첫 번째 파일의 컬럼명을 유지

    df_list.append(df_temp)

# 데이터가 있을 때만 병합 및 저장
if df_list:
    df_merged = pd.concat(df_list, ignore_index=True)
    df_merged.to_csv(r"data/mission3_results.csv", sep='|', index=False, encoding='utf-8')
    print("Merged CSV file saved.")
else:
    print("No valid CSV files found. Merging skipped.")