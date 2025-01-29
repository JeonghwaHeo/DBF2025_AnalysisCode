import numpy as np
from itertools import product
import time
from setup_dataclass import *
from dataclasses import replace
from vsp_analysis import  loadAnalysisResults
from mission_analysis import MissionAnalyzer, visualize_mission
from internal_dataclass import *
import os 
import os.path
import pandas as pd
import time
from dataclasses import asdict
import csv

def runMissionGridSearch(hashVal:str, 
                          presetValues:PresetValues,
                          missionParamConstraints:MissionParamConstraints, 
                          propulsionSpecs:PropulsionSpecs,
                          csvPath:str = "data/test.csv"
                          ) :


    analysisResults = loadAnalysisResults(hashVal, csvPath)
    ## Variable lists using for optimization
   
    M2_throttle_climb_list = np.arange(
            missionParamConstraints.M2_throttle_climb_min, 
            missionParamConstraints.M2_throttle_climb_max + missionParamConstraints.M2_throttle_analysis_interval/2, 
            missionParamConstraints.M2_throttle_analysis_interval
        )
    M2_throttle_turn_list = np.arange(
            missionParamConstraints.M2_throttle_turn_min, 
            missionParamConstraints.M2_throttle_turn_max + missionParamConstraints.M2_throttle_analysis_interval/2, 
            missionParamConstraints.M2_throttle_analysis_interval
        )
    M2_throttle_level_list = np.arange(
            missionParamConstraints.M2_throttle_level_min, 
            missionParamConstraints.M2_throttle_level_max + missionParamConstraints.M2_throttle_analysis_interval/2, 
            missionParamConstraints.M2_throttle_analysis_interval
        )
    
    print(f"\nMission 2 throttle climb list: {M2_throttle_climb_list}")
    print(f"Mission 2 throttle turn list: {M2_throttle_turn_list}")
    print(f"Mission 2 throttle level list: {M2_throttle_level_list}\n")

    M3_throttle_climb_list = np.arange(
            missionParamConstraints.M3_throttle_climb_min, 
            missionParamConstraints.M3_throttle_climb_max + missionParamConstraints.M3_throttle_analysis_interval/2, 
            missionParamConstraints.M3_throttle_analysis_interval
        )
    M3_throttle_turn_list = np.arange(
            missionParamConstraints.M3_throttle_turn_min, 
            missionParamConstraints.M3_throttle_turn_max + missionParamConstraints.M3_throttle_analysis_interval/2, 
            missionParamConstraints.M3_throttle_analysis_interval
        )
    M3_throttle_level_list = np.arange(
            missionParamConstraints.M3_throttle_level_min, 
            missionParamConstraints.M3_throttle_level_max + missionParamConstraints.M3_throttle_analysis_interval/2, 
            missionParamConstraints.M3_throttle_analysis_interval
        )

    print(f"\nMission 3 throttle climb list: {M3_throttle_climb_list}")
    print(f"Mission 3 throttle turn list: {M3_throttle_turn_list}")
    print(f"Mission 3 throttle level list: {M3_throttle_level_list}\n")

    # Create iterator for all combinations
    throttle_combinations = product(M2_throttle_climb_list, M2_throttle_turn_list, M2_throttle_level_list, M3_throttle_climb_list, M3_throttle_turn_list, M3_throttle_level_list)

    # Print total combinations
    total = len(M2_throttle_climb_list) * len(M2_throttle_turn_list) * len(M2_throttle_level_list) * len(M3_throttle_climb_list) * len(M3_throttle_turn_list) * len(M3_throttle_level_list)
    print(f"Testing {total} combinations...")

    # Test each combination
    for i, (M2_throttle_climb, M2_throttle_turn, M2_throttle_level, M3_throttle_climb, M3_throttle_turn, M3_throttle_level) in enumerate(throttle_combinations):
        print(f"[{time.strftime('%Y-%m-%d %X')}] Mission Grid Progress: {i+1}/{total} configurations")

        # Create mission 2 parameters for this combination
        mission2Params = MissionParameters(
            max_speed= 40,                       # Fixed
            max_load_factor = 4.0,               # Fixed
                  
            throttle_climb = M2_throttle_climb,
            throttle_level = M2_throttle_level,
            throttle_turn = M2_throttle_turn,    # Fixed

            propeller_data_path=propulsionSpecs.M2_propeller_data_path,
            max_battery_capacity = presetValues.max_battery_capacity 
        )

        # Create mission 3 parameters for this combination
        mission3Params = MissionParameters(
            max_speed= 40,                       # Fixed
            max_load_factor = 4.0,               # Fixed
                  
            throttle_climb = M3_throttle_climb,
            throttle_level = M3_throttle_level,
            throttle_turn = M3_throttle_turn,    # Fixed

            propeller_data_path=propulsionSpecs.M3_propeller_data_path,
            max_battery_capacity = presetValues.max_battery_capacity 
        )

        try:
            # Create mission analyzer and run mission 2
            mission2Analyzer = MissionAnalyzer(analysisResults, mission2Params, presetValues, propulsionSpecs)
            fuel_weight, flight_time = mission2Analyzer.run_mission2()
            obj2 = fuel_weight * 2.204 / flight_time # 2.204는 파운드 변환

            # Create mission analyzer and run mission 3           
            analysisResults_for_mission3 = replace(analysisResults,
                                                m_total=analysisResults.m_total - analysisResults.m_fuel,
                                                m_fuel=0.0)
            mission3Analyzer = MissionAnalyzer(analysisResults_for_mission3, mission3Params, presetValues, propulsionSpecs)
            N_laps = mission3Analyzer.run_mission3()
            obj3 = N_laps + 2.5 / (presetValues.m_x1 * 2.204)

            results = {
                'timestamp': time.strftime("%Y-%m-%d %X"),
                'hash': hashVal,
                'fuel_weight' : fuel_weight,
                'flight_time' : flight_time,
                'N_laps' : N_laps,
                'objective_2': obj2,
                'objective_3': obj3,
                'mission2_throttle_climb': M2_throttle_climb,
                'mission2_throttle_turn': M2_throttle_turn,
                'mission2_throttle_level': M2_throttle_level,
                'mission3_throttle_climb': M3_throttle_climb,
                'mission3_throttle_turn': M3_throttle_turn, 
                'mission3_throttle_level': M3_throttle_level
            }
    
            results = pd.DataFrame([results])
    
            writeMissionAnalysisResults(hashVal, results, presetValues, propulsionSpecs)

        except Exception as e:
            print(f"Failed with throttles M2 : Climb({M2_throttle_climb:.2f}) Trun({M2_throttle_turn:.2f}) Level ({M2_throttle_level:.2f})")
            print(f"Failed with throttles M3 : Climb({M3_throttle_climb:.2f}) Trun({M3_throttle_turn:.2f}) Level ({M3_throttle_level:.2f})")
            print(f"Error : {str(e)}")
            continue
   
    print("\nDone Mission Analysis ^_^")

def writeMissionAnalysisResults(hashVal:str, results, presetValues:PresetValues, propulsionSpecs:PropulsionSpecs, readcsvPath:str = "data/test.csv", writecsvPath:str = "data/total_results.csv"):
    existing_df = pd.read_csv(readcsvPath, sep='|', encoding='utf-8')
    base_row = existing_df[existing_df['hash'] == hashVal]
    base_row_dict = base_row.to_dict(orient="records")[0]
    preset_dict = vars(presetValues)
    propulsion_dict = vars(propulsionSpecs)
    
    combined_dict = {**base_row_dict, **preset_dict, **propulsion_dict}
    common_row = pd.DataFrame([combined_dict])
 
    new_row_df = pd.merge(common_row, results, on = 'hash')
    resultID = pd.util.hash_pandas_object(new_row_df, index=False)
    new_row_df['resultID'] = str(resultID.iloc[0])
    new_row_df['resultID'] = "'" + new_row_df['resultID'] + "'"
    

    if not os.path.isfile(writecsvPath):
        df_copy = new_row_df.copy()
        df_copy.to_csv(writecsvPath, sep='|', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
    else:
        df = pd.read_csv(writecsvPath, sep='|', encoding='utf-8')
        df= pd.concat([df,new_row_df])
        df_copy = df.copy()
        df_copy.to_csv(writecsvPath, sep='|', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)

def format_number(n: float) -> str:
    return f"{n:.6f}"  # 6 decimal places should be sufficient for most cases

def ResultAnalysis(presetValues:PresetValues,
                   readcsvPath:str = "data/total_results.csv",
                   writecsvPath:str = "data/organized_results.csv"):
    
    total_df = pd.read_csv(readcsvPath, sep='|', encoding='utf-8')
    max_obj2 = total_df['objective_2'].max()
    max_obj3 = total_df['objective_3'].max()

    total_df['score2'] = total_df['objective_2'] / max_obj2 + 1
    total_df['score3'] = total_df['objective_3'] / max_obj3 + 2
    total_df['SCORE'] = total_df['score2']*presetValues.score_weight_ratio + total_df['score3']*(1-presetValues.score_weight_ratio)

    organized_df = total_df[['resultID',
                            'hash',
                            'm_total',
                            'fuel_weight',
                            'span',
                            'AR',
                            'taper',
                            'twist',
                            'mission2_throttle_climb',
                            'mission2_throttle_turn',
                            'mission2_throttle_level',
                            'mission3_throttle_climb',
                            'mission3_throttle_turn',
                            'mission3_throttle_level',
                            'flight_time',
                            'N_laps',
                            'score2',
                            'score3',
                            'SCORE']]
    
    organized_df.to_csv(writecsvPath, sep='|', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)

    max_SCORE = organized_df['SCORE'].max()
    max_SCORE_row = organized_df[organized_df['SCORE'] == max_SCORE]
    print('max_SCORE info : \n')
    print(max_SCORE_row)

    


if __name__=="__main__":
    presetValues = PresetValues(
            m_x1 = 0.2,                       # kg
            x1_flight_time = 30,              # sec
            number_of_motor= 1,
            max_battery_capacity = 2250,      # mAh (per one battery)
            min_battery_voltage = 20,         # V (원래는 3 x 6 = 18 V 인데 안전하게 20 V)
            Thrust_max = 6.0,
            propulsion_efficiency = 0.8,      # Efficiency of the propulsion system
            score_weight_ratio = 0.5            # mission2/3 score weight ratio
            )
    a=loadAnalysisResults(6941088787683630519)
    score, param = runMissionGridSearch(6941088787683630519,
                          MissionParamConstraints (
                              throttle_climb_min = 0.9,
                              throttle_climb_max = 0.9,
                              throttle_turn_min = 0.5,
                              throttle_turn_max = 0.5,
                              throttle_level_min = 0.5,
                              throttle_level_max = 0.5,
                              throttle_analysis_interval = 0.05,
                              ),
                          presetValues
                          )

    missionAnalyzer = MissionAnalyzer(a,param[0],presetValues) 
    missionAnalyzer.run_mission2()
    visualize_mission(missionAnalyzer.stateLog)
    missionAnalyzer = MissionAnalyzer(a,param[1],presetValues) 
    missionAnalyzer.run_mission3()
    visualize_mission(missionAnalyzer.stateLog)





