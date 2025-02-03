import pandas as pd
import glob
import os
import sys

# 명령줄 인자로 mission과 server 값 받기
if len(sys.argv) != 3:
    print("Usage: python3 mission_combiner.py <s1> <m2>     (server1의 mission2일 때)")
    sys.exit(1)

server = sys.argv[1]   # 예: "s1", "s2"
mission = sys.argv[2]  # 예: "m2", "m3"

# 파일 경로 설정
csv_files = glob.glob(f"data/server{server[-1]}/mission{mission[-1]}_results_*.csv")
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
    output_file = f"data/server{server[-1]}/mission{mission[-1]}_results.csv"
    df_merged = pd.concat(df_list, ignore_index=True)
    df_merged.to_csv(output_file, sep='|', index=False, encoding='utf-8')
    print(f"Merged CSV file saved: {output_file}")
else:
    print("No valid CSV files found. Merging skipped.")