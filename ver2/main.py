import pandas as pd
from vsp_grid import runVSPGridAnalysis
from mission_grid import runMissionGridSearch, ResultAnalysis
from mission_analysis import MissionAnalyzer
from vsp_analysis import removeAnalysisResults, loadAnalysisResults
from internal_dataclass import *
from setup_dataclass import *
import argparse

import cProfile
import pstats
from pstats import SortKey

import pandas as pd
from vsp_grid import runVSPGridAnalysis
from mission_grid import runMissionGridSearch, ResultAnalysis
from vsp_analysis import removeAnalysisResults
from internal_dataclass import *
from setup_dataclass import *
import argparse
import os
import glob, time

def get_config():
    presetValues = PresetValues(
        m_x1 = 200,                         # g
        x1_time_margin = 10,                # sec
        
        throttle_takeoff = 0.9,             # 0~1
        max_climb_angle = 40,                 #deg
        max_load = 30,                      # kg
        h_flap_transition = 5,              # m
        
        number_of_motor = 2,                 
        min_battery_voltage = 21.8,         # V 
        score_weight_ratio = 0.5            # mission2/3 score weight ratio (0~1)
        )
    
    propulsionSpecs = PropulsionSpecs(
        M2_propeller_data_path = "data/propDataCSV/PER3_8x6E.csv",
        M3_propeller_data_path = "data/propDataCSV/PER3_8x6E.csv",
        battery_data_path = "data/batteryDataCSV/Maxamps_2250mAh_6S.csv",
        Kv = 109.91,
        R = 0.062,
        number_of_battery = 2,
        n_cell = 6,
        battery_Wh = 49.95,
        max_current = 60,
        max_power = 1332    
    )
    
    aircraftParamConstraints = AircraftParamConstraints (
  
        span_min = 1800.0,                   # mm
        span_max = 1800.0,                   
        span_interval = 25.0,
    
        AR_min = 4.75,                  
        AR_max = 4.75,
        AR_interval = 0.25,
        
        taper_min = 0.9,
        taper_max = 0.9,                      
        taper_interval = 0.05,
        
        twist_min = 2.0,                     # degree
        twist_max = 2.0,                     
        twist_interval = 1.0,
        
        #airfoil_list = ['sg6043','s9027','hq3011','e216','s4022']
        # airfoil_list = ['sg6043','s9027','hq3011','s4022']
        airfoil_list = ['e216']
        )
    
    aerodynamicSetup = AerodynamicSetup(
        alpha_start = -3,
        alpha_end = 10,
        alpha_step = 1,
        fuselage_cross_section_area = 19427,    # mm2
        fuselage_Cd_datapath = "data/fuselageDragCSV/fuselageDragCoefficients.csv",
        AOA_stall = 13,
        AOA_takeoff_max = 10,
        AOA_climb_max = 8,
        AOA_turn_max = 8  
    )

    baseAircraft = Aircraft(
        m_fuselage = 2500,
        wing_area_blocked_by_fuselage = 72640,    #mm2
        wing_density = 0.0000588,

        mainwing_span = 1800,        
        mainwing_AR = 5.45,           
        mainwing_taper = 0.65,        
        mainwing_twist = 0.0,        
        mainwing_sweepback = 0,   
        mainwing_dihedral = 5.0,     
        mainwing_incidence = 0.0,    

        flap_start = [0.182, 0.402],            
        flap_end = [0.335, 0.628],              
        flap_angle = [20.0, 20.0],           
        flap_c_ratio = [0.35, 0.35],         

        horizontal_volume_ratio = 0.7,
        horizontal_area_ratio = 0.25, 
        horizontal_AR = 4.0,         
        horizontal_taper = 1,      
        horizontal_ThickChord = 8,

        vertical_volume_ratio = 0.053,
        vertical_taper = 0.6,        
        vertical_ThickChord = 9,  
        
        mainwing_airfoil_datapath = "data/airfoilDAT/sg6043.dat",
        horizontal_airfoil_datapath= "data/airfoilDAT/naca0008.dat",
        vertical_airfoil_datapath= "data/airfoilDAT/naca0009.dat"
        
        )

    missionParamConstraints = MissionParamConstraints (
                
                MTOW_min = 10.0,
                MTOW_max = 10.0,
                MTOW_analysis_interval = 0.2,
                
                M2_max_speed_min = 34,
                M2_max_speed_max = 34,
                M3_max_speed_min = 24,
                M3_max_speed_max = 24,
                max_speed_analysis_interval = 2,
                
                #Constraints for calculating mission2
                M2_climb_thrust_ratio_min = 0.9,
                M2_climb_thrust_ratio_max = 0.9,
                M2_turn_thrust_ratio_min = 0.7,
                M2_turn_thrust_ratio_max = 0.7,
                M2_level_thrust_ratio_min = 0.9,
                M2_level_thrust_ratio_max = 0.9,
                M2_thrust_analysis_interval = 0.05,
    
                #Constraints for calculating mission3  
                M3_climb_thrust_ratio_min = 0.9,
                M3_climb_thrust_ratio_max = 0.9,
                M3_turn_thrust_ratio_min = 0.4,
                M3_turn_thrust_ratio_max = 0.4,
                M3_level_thrust_ratio_min = 0.5,
                M3_level_thrust_ratio_max = 0.5,
                M3_thrust_analysis_interval = 0.05,
                
                wing_loading_min = 5,
                wing_loading_max = 15
                )
        

    return (presetValues, propulsionSpecs, aircraftParamConstraints, 
            aerodynamicSetup, baseAircraft, missionParamConstraints)

def run_vsp_analysis(server_id: int, total_servers: int):
    (presetValues, propulsionSpecs, aircraftParamConstraints, 
     aerodynamicSetup, baseAircraft, _) = get_config()
    
    # Use server-specific output path
    output_path = f"data/aircraft_{server_id}.csv"
    vsp_path = f"aircraft_{server_id}.vsp3"
    if os.path.exists(output_path):
        os.remove(output_path)
        
    runVSPGridAnalysis(aircraftParamConstraints, aerodynamicSetup, presetValues, 
                      baseAircraft, server_id, total_servers, csvPath=output_path,vspPath=vsp_path)

def run_mission_analysis(server_id: int, total_servers: int):
    (presetValues, propulsionSpecs, _, _, _, missionParamConstraints) = get_config()
    
    #final_hash_list = []
    #hash_list ={2:[],3:[]} 
    #for j in range(2,4):    
    #    csv_files = glob.glob(f"./data/mission{j}_results*.csv")
    #    for k, csv_file in enumerate(csv_files):
    #        if os.path.getsize(csv_file) == 0:  # 빈 파일이면 건너뛰기
    #            print(f"Skipping empty file: {csv_file}")
    #            continue
    #        
    #        df_temp = pd.read_csv(csv_file, sep='|', header=0, encoding='utf-8')
    #        try:
    #            hash_list[j]=[*hash_list[j], *df_temp['hash'].unique()[:-1]] # 마지막 hash 지우기
    #        except Exception as e:
    #            print("error")
    #final_hash_list = list(set(hash_list[2])&set(hash_list[3])) # mission2/3 같이 나타나는 hash만

    df_saved = pd.read_csv(f"./data/aircraft.csv", sep='|', header=0, encoding='utf-8')
    #df_saved = df_saved[~df_saved["hash"].isin(final_hash_list)]

    # Read from combined aircraft.csv (assumed to be already merged)
    results = df_saved

    # Divide hash values among servers
    all_hashes = results["hash"].tolist()
    worker_hashes = all_hashes[server_id-1::total_servers]
    
    # Use server-specific output path for mission results
    output2_path = f"data/mission2_results_{server_id}.csv"
    output3_path = f"data/mission3_results_{server_id}.csv"

    # Run mission analysis for this worker's hashes
    for hashVal in worker_hashes:
        print(f"\nWorker {server_id} analyzing hash {hashVal}")
        runMissionGridSearch(hashVal, presetValues, missionParamConstraints, 
                             propulsionSpecs, 
                             mission2Out=output2_path,
                             mission3Out=output3_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, required=True, help="current server ID")
    parser.add_argument("--total_server", type=int, required=True, help="total server number")
    parser.add_argument("--mode", choices=['vsp', 'mission'], required=True, 
                      help="Operation mode: 'vsp' for VSP analysis or 'mission' for mission analysis")
    args = parser.parse_args()
    
    print(f"Starting worker {args.server_id} of {args.total_server} in {args.mode} mode")

    if args.mode == 'vsp':
        run_vsp_analysis(args.server_id, args.total_server)
    else:
        run_mission_analysis(args.server_id, args.total_server)

if __name__ == "__main__":
    main()
