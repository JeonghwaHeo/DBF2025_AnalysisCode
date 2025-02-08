import pandas as pd
from src.vsp_grid import runVSPGridAnalysis
from src.mission_grid import runMissionGridSearch
from config import get_config
from src.internal_dataclass import *
from src.setup_dataclass import *
import argparse
import shutil
import os

def run_vsp_analysis(server_id: int, total_servers: int):
    (presetValues, _, aircraftParamConstraints, aerodynamicSetup, baseAircraft, _) = get_config()
    
    output_path = f"data/aircraft_{server_id}.csv"
    vsp_path = f"aircraft_{server_id}.vsp3"
    if os.path.exists(output_path):
        os.remove(output_path)
        
    runVSPGridAnalysis(aircraftParamConstraints, aerodynamicSetup, presetValues, 
                      baseAircraft, server_id, total_servers, csvPath=output_path,vspPath=vsp_path)

def run_mission_analysis(server_id: int, total_servers: int):
    (presetValues, propulsionSpecs, _, _, _, missionParamConstraints) = get_config()
    
    if os.path.exists(f"data/mission2_results_{server_id}.csv"):
        os.remove(f"data/mission2_results_{server_id}.csv")
    if os.path.exists(f"data/mission3_results_{server_id}.csv"):
        os.remove(f"data/mission3_results_{server_id}.csv")
    
    
    
    df_saved = pd.read_csv(f"./data/aircraft.csv", sep='|', header=0, encoding='utf-8')
    results = df_saved

    all_hashes = results["hash"].tolist()
    worker_hashes = all_hashes[server_id-1::total_servers]
    
    output2_path = f"data/mission2_results_{server_id}.csv"
    output3_path = f"data/mission3_results_{server_id}.csv"

    for hashVal in worker_hashes:
        print(f"\nWorker {server_id} analyzing hash {hashVal}")
        runMissionGridSearch(hashVal, presetValues, missionParamConstraints, 
                             propulsionSpecs, 
                             mission2Out=output2_path,
                             mission3Out=output3_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, default = 1, help="current server ID")
    parser.add_argument("--total_server", type=int, default = 1, help="total server number")
    parser.add_argument("--mode", choices=['vsp', 'mission','all'], default = 'all', 
                      help="Operation mode: 'vsp' for VSP analysis or 'mission' for mission analysis or 'all' for executing in 1 total server.")
    args = parser.parse_args()
    
    print(f"Starting worker {args.server_id} of {args.total_server} in {args.mode} mode")

    if args.mode == 'vsp':
        run_vsp_analysis(args.server_id, args.total_server)
    elif args.mode == 'mission':
        run_mission_analysis(args.server_id, args.total_server)
    elif args.mode == "all":
        run_vsp_analysis(args.server_id, args.total_server)
        shutil.copyfile('data/aircraft_1.csv','data/aircraft.csv')
        run_mission_analysis(args.server_id, args.total_server)
        

if __name__ == "__main__":
    main()
