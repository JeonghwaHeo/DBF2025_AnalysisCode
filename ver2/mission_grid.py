import numpy as np
from itertools import product
import time
from setup_dataclass import *
from vsp_analysis import  loadAnalysisResults
from mission_analysis import MissionAnalyzer, visualize_mission
from internal_dataclass import *
import os 
import os.path
import pandas as pd
import time
import csv

def runMissionGridSearch(hashVal:str, 
                          presetValues:PresetValues,
                          missionParamConstraints:MissionParamConstraints, 
                          propulsionSpecs:PropulsionSpecs,
                          csvPath:str = "data/aircraft.csv"
                          ) :


    analysisResults = loadAnalysisResults(hashVal, csvPath)
    ## Variable lists using for optimization
    
    MTOW_list = np.arange(
            missionParamConstraints.MTOW_min, 
            missionParamConstraints.MTOW_max + missionParamConstraints.MTOW_analysis_interval/2, 
            missionParamConstraints.MTOW_analysis_interval        
    )
    
    MTOW_min_condition = max(missionParamConstraints.wing_loading_min * analysisResults.Sref * 1e-6,
                             analysisResults.m_empty/1000)
    MTOW_max_condition = missionParamConstraints.wing_loading_max * analysisResults.Sref * 1e-6
    
    MTOW_list = MTOW_list[(MTOW_list >= MTOW_min_condition) & (MTOW_list <= MTOW_max_condition)]
    
    if len(MTOW_list) == 0: return
    
    M2_max_speed_list = np.arange(
            missionParamConstraints.M2_max_speed_min, 
            missionParamConstraints.M2_max_speed_max + missionParamConstraints.max_speed_analysis_interval/2, 
            missionParamConstraints.max_speed_analysis_interval
        )
    
    M2_climb_thrust_ratio_list = np.arange(
            missionParamConstraints.M2_climb_thrust_ratio_min, 
            missionParamConstraints.M2_climb_thrust_ratio_max + missionParamConstraints.M2_thrust_analysis_interval/2, 
            missionParamConstraints.M2_thrust_analysis_interval
        )
    M2_turn_thrust_ratio_list = np.arange(
            missionParamConstraints.M2_turn_thrust_ratio_min, 
            missionParamConstraints.M2_turn_thrust_ratio_max + missionParamConstraints.M2_thrust_analysis_interval/2, 
            missionParamConstraints.M2_thrust_analysis_interval
        )
    M2_level_thrust_ratio_list = np.arange(
            missionParamConstraints.M2_level_thrust_ratio_min, 
            missionParamConstraints.M2_level_thrust_ratio_max + missionParamConstraints.M2_thrust_analysis_interval/2, 
            missionParamConstraints.M2_thrust_analysis_interval
        )
    
    print(f"\nMTOW list: {MTOW_list}")
    print(f"\nMission 2 max speed list: {M2_max_speed_list}")
    print(f"Mission 2 throttle climb list: {M2_climb_thrust_ratio_list}")
    print(f"Mission 2 throttle turn list: {M2_turn_thrust_ratio_list}")
    print(f"Mission 2 throttle level list: {M2_level_thrust_ratio_list}\n")

    M3_max_speed_list = np.arange(
            missionParamConstraints.M3_max_speed_min, 
            missionParamConstraints.M3_max_speed_max + missionParamConstraints.max_speed_analysis_interval/2, 
            missionParamConstraints.max_speed_analysis_interval
        )
    
    M3_climb_thrust_ratio_list = np.arange(
            missionParamConstraints.M3_climb_thrust_ratio_min, 
            missionParamConstraints.M3_climb_thrust_ratio_max + missionParamConstraints.M3_thrust_analysis_interval/2, 
            missionParamConstraints.M3_thrust_analysis_interval
        )
    M3_turn_thrust_ratio_list = np.arange(
            missionParamConstraints.M3_turn_thrust_ratio_min, 
            missionParamConstraints.M3_turn_thrust_ratio_max + missionParamConstraints.M3_thrust_analysis_interval/2, 
            missionParamConstraints.M3_thrust_analysis_interval
        )
    M3_level_thrust_ratio_list = np.arange(
            missionParamConstraints.M3_level_thrust_ratio_min, 
            missionParamConstraints.M3_level_thrust_ratio_max + missionParamConstraints.M3_thrust_analysis_interval/2, 
            missionParamConstraints.M3_thrust_analysis_interval
        )

    print(f"\nMission 3 max speed list: {M3_max_speed_list}")
    print(f"Mission 3 throttle climb list: {M3_climb_thrust_ratio_list}")
    print(f"Mission 3 throttle turn list: {M3_turn_thrust_ratio_list}")
    print(f"Mission 3 throttle level list: {M3_level_thrust_ratio_list}\n")

    # Create iterator for all combinations
    combinations = product(MTOW_list, M2_max_speed_list, M2_climb_thrust_ratio_list, M2_turn_thrust_ratio_list, M2_level_thrust_ratio_list, M3_max_speed_list, M3_climb_thrust_ratio_list, M3_turn_thrust_ratio_list, M3_level_thrust_ratio_list)

    # Print total combinations
    total = len(MTOW_list) * len(M2_max_speed_list) * len(M2_climb_thrust_ratio_list) * len(M2_turn_thrust_ratio_list) * len(M2_level_thrust_ratio_list) * len(M3_max_speed_list) * len(M3_climb_thrust_ratio_list) * len(M3_turn_thrust_ratio_list) * len(M3_level_thrust_ratio_list)
    print(f"Testing {total} combinations...")

    # Test each combination
    for i, (MTOW, M2_max_speed, M2_climb_thrust_ratio, M2_turn_thrust_ratio, M2_level_thrust_ratio,M3_max_speed, M3_climb_thrust_ratio, M3_turn_thrust_ratio, M3_level_thrust_ratio) in enumerate(combinations):
        print(f"[{time.strftime('%Y-%m-%d %X')}] Mission Grid Progress: {i+1}/{total} configurations")

        # Create mission 2 parameters for this combination
        mission2Params = MissionParameters(
            m_takeoff = MTOW,
            max_speed= M2_max_speed,                      
            max_load_factor = presetValues.max_load / MTOW,          
                  
            climb_thrust_ratio = M2_climb_thrust_ratio,
            level_thrust_ratio = M2_level_thrust_ratio,
            turn_thrust_ratio = M2_turn_thrust_ratio,   

            propeller_data_path=propulsionSpecs.M2_propeller_data_path,
        )

        # Create mission 3 parameters for this combination
        mission3Params = MissionParameters(
            m_takeoff = analysisResults.m_empty/1000,
            max_speed= M3_max_speed,                      
            max_load_factor = presetValues.max_load * 1000 / analysisResults.m_empty,            
                  
            climb_thrust_ratio = M3_climb_thrust_ratio,
            level_thrust_ratio = M3_level_thrust_ratio,
            turn_thrust_ratio = M3_turn_thrust_ratio,   

            propeller_data_path=propulsionSpecs.M3_propeller_data_path
        )

        try:
            mission2Analyzer = MissionAnalyzer(analysisResults, mission2Params, presetValues, propulsionSpecs)
            fuel_weight, flight_time = mission2Analyzer.run_mission2()
            
            if(fuel_weight == -1 and flight_time == -1):
                print("mission2 fail")
                continue
            
            obj2 = fuel_weight * 2.204 / flight_time 

            mission3Analyzer = MissionAnalyzer(analysisResults, mission3Params, presetValues, propulsionSpecs)
            N_laps = mission3Analyzer.run_mission3()
            
            if(N_laps==-1):
                print("mission3 fail")
                continue
            obj3 = N_laps - 1 + 2.5 / (presetValues.m_x1 /1000 * 2.204 )

            results = {
                'timestamp': time.strftime("%Y-%m-%d %X"),
                'hash': hashVal,
                'fuel_weight' : fuel_weight,
                'flight_time' : flight_time,
                'N_laps' : N_laps,
                'objective_2': obj2,
                'objective_3': obj3,
                'MTOW' : MTOW,
                'M2_max_speed' : M2_max_speed,
                'M3_max_speed' : M3_max_speed,
                'mission2_climb_thrust_ratio': M2_climb_thrust_ratio,
                'mission2_turn_thrust_ratio': M2_turn_thrust_ratio,
                'mission2_level_thrust_ratio': M2_level_thrust_ratio,
                'mission3_climb_thrust_ratio': M3_climb_thrust_ratio,
                'mission3_turn_thrust_ratio': M3_turn_thrust_ratio, 
                'mission3_level_thrust_ratio': M3_level_thrust_ratio
            }
    
            results = pd.DataFrame([results])
    
            writeMissionAnalysisResults(hashVal, results, presetValues, propulsionSpecs)

        except Exception as e:
            print(f"Failed with throttles M2 : Climb({M2_climb_thrust_ratio:.2f}) Trun({M2_turn_thrust_ratio:.2f}) Level ({M2_level_thrust_ratio:.2f})")
            print(f"Failed with throttles M3 : Climb({M3_climb_thrust_ratio:.2f}) Trun({M3_turn_thrust_ratio:.2f}) Level ({M3_level_thrust_ratio:.2f})")
            print(f"Error : {str(e)}")
            continue
   
    print("\nDone Mission Analysis ^_^")

def writeMissionAnalysisResults(hashVal:str, results, presetValues:PresetValues, propulsionSpecs:PropulsionSpecs, readcsvPath:str = "data/aircraft.csv", writecsvPath:str = "data/total_results.csv"):
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
                            'MTOW',
                            'fuel_weight',
                            'span',
                            'AR',
                            'taper',
                            'twist',
                            'M2_max_speed',
                            'M3_max_speed',
                            'mission2_climb_thrust_ratio',
                            'mission2_turn_thrust_ratio',
                            'mission2_level_thrust_ratio',
                            'mission3_climb_thrust_ratio',
                            'mission3_turn_thrust_ratio',
                            'mission3_level_thrust_ratio',
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
        m_x1 = 200,                         # g
        x1_time_margin = 10,                # sec
        
        throttle_takeoff = 0.9,             # 0~1
        max_climb_angle=40,                 #deg
        max_load = 30,                      # kg
        h_flap_transition = 5,              # m
        
        number_of_motor = 2,                 
        min_battery_voltage = 21.8,         # V 
        score_weight_ratio = 0.5            # mission2/3 score weight ratio (0~1)
        )
    #a=loadAnalysisResults(6941088787683630519)
    score, param = runMissionGridSearch(6941088787683630519,
                          MissionParamConstraints (
                              climb_thrust_ratio_min = 0.9,
                              climb_thrust_ratio_max = 0.9,
                              turn_thrust_ratio_min = 0.5,
                              turn_thrust_ratio_max = 0.5,
                              level_thrust_ratio_min = 0.5,
                              level_thrust_ratio_max = 0.5,
                              throttle_analysis_interval = 0.05,
                              ),
                          presetValues
                          )

    #missionAnalyzer = MissionAnalyzer(a,param[0],presetValues) 
    #missionAnalyzer.run_mission2()
    #visualize_mission(missionAnalyzer.stateLog)
    #missionAnalyzer = MissionAnalyzer(a,param[1],presetValues) 
    #missionAnalyzer.run_mission3()
    #visualize_mission(missionAnalyzer.stateLog)





