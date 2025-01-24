import numpy as np
from itertools import product
import time
from config import *
from dataclasses import replace
from vsp_analysis import  loadAnalysisResults
from mission_analysis import MissionAnalyzer, visualize_mission
from models import *
import os 
import os.path
import pandas as pd
import time
from dataclasses import asdict
import csv

def runMissionGridSearch(hashVal:int, 
                          missionParamConstraints:MissionParamConstraints, 
                          presetValues:PresetValues,
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

    # best_score_2 = float('-inf')
    # best_params_2 = None
    
    # best_score_3 = float('-inf')
    # best_params_3 = None

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
            max_battery_capacity = presetValues.max_battery_capacity,
            throttle_takeoff = 0.9,              # Fixed
            throttle_climb = M2_throttle_climb,
            throttle_level = M2_throttle_level,
            throttle_turn = M2_throttle_turn,                
            max_climb_angle = 40,                # Fixed
            max_speed= 40,                       # Fixed
            max_load_factor = 4.0,               # Fixed
            h_flap_transition = 5                # Fixed
        )

        # Create mission 3 parameters for this combination
        mission3Params = MissionParameters(
            max_battery_capacity = presetValues.max_battery_capacity,
            throttle_takeoff = 0.9,              # Fixed
            throttle_climb = M3_throttle_climb,
            throttle_level = M3_throttle_level,
            throttle_turn = M3_throttle_turn,                
            max_climb_angle = 40,                # Fixed
            max_speed= 40,                       # Fixed
            max_load_factor = 4.0,               # Fixed
            h_flap_transition = 5                # Fixed
        )

        try:
            # Create mission analyzer and run mission 2
            mission2Analyzer = MissionAnalyzer(analysisResults, mission2Params, presetValues)
            fuel_weight, flight_time = mission2Analyzer.run_mission2()
            obj2 = fuel_weight * 2.204 / flight_time # 2.204는 파운드 변환

            # Create mission analyzer and run mission 3           
            analysisResults_for_mission3 = replace(analysisResults,
                                                m_total=analysisResults.m_total - analysisResults.m_fuel,
                                                m_fuel=0.0)
            mission3Analyzer = MissionAnalyzer(analysisResults_for_mission3, mission3Params, presetValues)
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
    
            writeMissionAnalysisResults(hashVal, results, presetValues)

            # # Update best score if current score is better
            # if score_2 > best_score_2:
            #     best_score_2 = score_2
            #     best_params_2 = missionParams
                
            #     print(f"\nNew best score for Mission 2: {best_score_2}")
            #     # print(f"> Total Mass: {total_mass:.2f}")
            #     print(f"> Throttle settings - Climb: {throttle_climb:.2f}, "
            #           f"Turn: {throttle_turn:.2f}, Level: {throttle_level:.2f}")
            # # Update best score if current score is better
            # if score_3 > best_score_3:
            #     best_score_3 = score_3
            #     best_params_3 = missionParams
                
            #     print(f"\nNew best score for Mission 3: {best_score_3}")
            #     # print(f"> Total Mass: {total_mass:.2f}")
            #     print(f"> Throttle settings - Climb: {throttle_climb:.2f}, "
            #           f"Turn: {throttle_turn:.2f}, Level: {throttle_level:.2f}")
        
        except Exception as e:
            print(f"Failed with throttles M2 : Climb({M2_throttle_climb:.2f}) Trun({M2_throttle_turn:.2f}) Level ({M2_throttle_level:.2f})")
            print(f"Failed with throttles M3 : Climb({M3_throttle_climb:.2f}) Trun({M3_throttle_turn:.2f}) Level ({M3_throttle_level:.2f}): {str(e)}")
            continue
   
    print("\nDone Mission Analysis ^_^")


    # print(f"\nBest score for Mission 2: {best_score_2}")
    # print(f"> Throttle settings - Climb: {best_params_2.throttle_climb:.2f}, "
    #       f"Turn: {best_params_2.throttle_turn:.2f}, Level: {best_params_2.throttle_level:.2f}")

    # print(f"\nBest score for Mission 3: {best_score_3}")
    # print(f"> Throttle settings - Climb: {best_params_3.throttle_climb:.2f}, "
    #       f"Turn: {best_params_3.throttle_turn:.2f}, Level: {best_params_3.throttle_level:.2f}")
    

if __name__=="__main__":
    presetValues = PresetValues(
            m_x1 = 0.2,                       # kg
            x1_flight_time = 30,              # sec
            max_battery_capacity = 2250,      # mAh (per one battery)
            Thrust_max = 6.6,                 # kg (two motors)
            min_battery_voltage = 20,         # V (원래는 3 x 6 = 18 V 인데 안전하게 20 V)
            propulsion_efficiency = 0.8,      # Efficiency of the propulsion system
            score_weight_ratio = 1            # mission2/3 score weight ratio
            )
    a=loadAnalysisResults(687192594661440415)
    score, param = runMissionGridSearch(687192594661440415,
                          MissionParamConstraints (
                              # total mass of the aircraft
                              m_total_max = 8000,
                              m_total_min = 6000,
                              m_total_interval = 5000,
                              #Constraints for calculating missions
                              throttle_climb_min = 1.0,
                              throttle_climb_max = 1.0,
                              throttle_turn_min = 0.7,
                              throttle_turn_max = 0.7,
                              throttle_level_min = 1.0,
                              throttle_level_max = 1.0,
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

def writeMissionAnalysisResults(hashVal, results, presetValues, readcsvPath:str = "data/test.csv", writecsvPath:str = "data/total_results.csv"):
    existing_df = pd.read_csv(readcsvPath, sep='|', encoding='utf-8')
    base_row = existing_df[existing_df['hash'] == int(hashVal)]
    preset_dict = vars(presetValues)
    preset_row = pd.DataFrame([preset_dict])
    common_row = pd.concat([base_row, preset_row],axis=1)
    
    new_row_df = pd.merge(common_row, results, on = 'hash')
    new_row_df['resultID'] = pd.util.hash_pandas_object(new_row_df, index=False)

    if not os.path.isfile(writecsvPath):
        df_copy = new_row_df.copy()
        df_copy.to_csv(writecsvPath, sep='|', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)
    else:
        df = pd.read_csv(writecsvPath, sep='|', encoding='utf-8')
        df= pd.concat([df,new_row_df])
        df_copy = df.copy()
        df_copy.to_csv(writecsvPath, sep='|', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)


# def save_mission_results(hashVal, scores, params, csv_path="./data/total_results.csv"):

#    # Create dictionary with results
#    results = {
#        'timestamp': time.strftime("%Y-%m-%d %X"),
#        'hash': hashVal,
#        'mission2_score': scores[0],
#        'mission3_score': scores[1],
#        'mission2_throttle_climb': params[0].throttle_climb,
#        'mission2_throttle_turn': params[0].throttle_turn,
#        'mission2_throttle_level': params[0].throttle_level,
#        'mission3_throttle_climb': params[1].throttle_climb,
#        'mission3_throttle_turn': params[1].throttle_turn, 
#        'mission3_throttle_level': params[1].throttle_level
#    }

#    # Convert to DataFrame
#    df = pd.DataFrame([results])

#    # Append to CSV or create new one
#    try:
#        df.to_csv(csv_path, mode='a',  index=False)
#    except Exception as e:
#        print(f"Failed to save results: {str(e)}")
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

    # total_df['resultID'] = 0


    # hash_dict = {
    #     "hash" : format_number(float(total_df['hash'])),
    #     "m_total" : format_number(float(total_df['m_total'])),
    #     "span" : format_number(float(total_df['span'])),
    #     "AR" : format_number(float(total_df['AR'])),
    #     "taper" : format_number(float(total_df['taper'])),
    #     "twist" : format_number(float(total_df['twist'])),
    #     "mission2_throttle_climb" : format_number(float(total_df['mission2_throttle_climb'])),
    #     "mission2_throttle_turn" : format_number(float(total_df['mission2_throttle_turn'])),
    #     "mission2_throttle_level" : format_number(float(total_df['mission2_throttle_level'])),
    #     "mission3_throttle_climb" : format_number(float(total_df['mission3_throttle_climb'])),
    #     "mission3_throttle_turn" : format_number(float(total_df['mission3_throttle_turn'])),
    #     "mission3_throttle_level" : format_number(float(total_df['mission3_throttle_level'])),

    # }

    # json_str = json.dumps(hash_dict, sort_keys=True)
    # hash_obj = hashlib.sha256(json_str.encode())
    # total_df['resultID'] = int.from_bytes(hash_obj.digest()[:8], byteorder='big')

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
    
    
    # organized_df['resultID'] = pd.util.hash_pandas_object(organized_df, index=False)
    # organized_df = organized_df[['resultID'] + [col for col in organized_df.columns if col != 'resultID']]
    
    organized_df.to_csv(writecsvPath, sep='|', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)

    max_SCORE = organized_df['SCORE'].max()
    max_SCORE_row = organized_df[organized_df['SCORE'] == max_SCORE]
    print('max_SCORE info : \n')
    print(max_SCORE_row)

    







