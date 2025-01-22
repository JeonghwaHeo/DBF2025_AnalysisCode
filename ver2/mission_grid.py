import numpy as np
from itertools import product
import time
from config import *
from vsp_analysis import  loadAnalysisResults
from mission_analysis import MissionAnalyzer, visualize_mission
from models import *

import pandas as pd
import time

def runMissionGridSearch(hashVal:int, 
                          missionParamConstraints:MissionParamConstraints, 
                          presetValues:PresetValues,
                          csvPath:str = "data/test.csv"
                          ) :


    analysisResults = loadAnalysisResults(hashVal, csvPath)
    ## Variable lists using for optimization
   
    throttle_climb_list = np.around(
        np.arange(
            missionParamConstraints.throttle_climb_min, 
            missionParamConstraints.throttle_climb_max + missionParamConstraints.throttle_analysis_interval, 
            missionParamConstraints.throttle_analysis_interval
        ), decimals=3
    )
    throttle_turn_list = np.around(
        np.arange(
            missionParamConstraints.throttle_turn_min, 
            missionParamConstraints.throttle_turn_max + missionParamConstraints.throttle_analysis_interval, 
            missionParamConstraints.throttle_analysis_interval
        ), decimals=3
    )
    throttle_level_list = np.around(
        np.arange(
            missionParamConstraints.throttle_level_min, 
            missionParamConstraints.throttle_level_max + missionParamConstraints.throttle_analysis_interval, 
            missionParamConstraints.throttle_analysis_interval
        ), decimals=3
    )
    
    print(f"\nthrottle climb list: {throttle_climb_list}")
    print(f"throttle turn list: {throttle_turn_list}")
    print(f"throttle level list: {throttle_level_list}\n")


    best_score_2 = float('-inf')
    best_params_2 = None
    
    best_score_3 = float('-inf')
    best_params_3 = None

    # Create iterator for all combinations
    throttle_combinations = product(throttle_climb_list, throttle_turn_list, throttle_level_list)

    # Print total combinations
    total = len(throttle_climb_list) * len(throttle_turn_list) * len(throttle_level_list)
    print(f"Testing {total} combinations...")

    # Test each combination
    for i, (total_mass, throttle_climb, throttle_turn, throttle_level) in enumerate(throttle_combinations):
        if(i%10 == 0): print(f"[{time.strftime("%Y-%m-%d %X")}] Progress: {i}/{total} configurations")
        # Create mission parameters for this combination
        missionParams = MissionParameters(
            m_total=total_mass,
            max_battery_capacity=presetValues.max_battery_capacity,
            throttle_takeoff=0.9,  # Fixed
            throttle_climb=throttle_climb,
            throttle_level=throttle_level,
            throttle_turn=throttle_turn,
            max_climb_angle=40,  # Fixed
            max_speed=40,  # Fixed
            max_load_factor=4.0,  # Fixed
            h_flap_transition=5  # Fixed
        )

        try:
            # Create mission analyzer and run mission 2
            missionAnalyzer = MissionAnalyzer(analysisResults, missionParams, presetValues)

            score_2 = missionAnalyzer.run_mission2()
            score_3 = missionAnalyzer.run_mission3()

            # Update best score if current score is better
            if score_2 > best_score_2:
                best_score_2 = score_2
                best_params_2 = missionParams
                
                print(f"\nNew best score for Mission 2: {best_score_2}")
                print(f"> Total Mass: {total_mass:.2f}")
                print(f"> Throttle settings - Climb: {throttle_climb:.2f}, "
                      f"Turn: {throttle_turn:.2f}, Level: {throttle_level:.2f}")
            # Update best score if current score is better
            if score_3 > best_score_3:
                best_score_3 = score_3
                best_params_3 = missionParams
                
                print(f"\nNew best score for Mission 3: {best_score_3}")
                print(f"> Total Mass: {total_mass:.2f}")
                print(f"> Throttle settings - Climb: {throttle_climb:.2f}, "
                      f"Turn: {throttle_turn:.2f}, Level: {throttle_level:.2f}")
        
        except Exception as e:
            print(f"Failed with throttles {throttle_climb:.2f}/{throttle_turn:.2f}/"
                  f"{throttle_level:.2f}: {str(e)}")
            continue
   
    print("\nDone!")
    print(f"\nBest score for Mission 2: {best_score_2}")
    print(f"> Total Mass: {best_params_2.m_total:.2f}")
    print(f"> Throttle settings - Climb: {best_params_2.throttle_climb:.2f}, "
          f"Turn: {best_params_2.throttle_turn:.2f}, Level: {best_params_2.throttle_level:.2f}")

    print(f"\nBest score for Mission 3: {best_score_3}")
    print(f"> Total Mass: {best_params_3.m_total:.2f}")
    print(f"> Throttle settings - Climb: {best_params_3.throttle_climb:.2f}, "
          f"Turn: {best_params_3.throttle_turn:.2f}, Level: {best_params_3.throttle_level:.2f}")
    return [best_score_2, best_score_3], [best_params_2, best_params_3]

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

def save_mission_results(hashVal, scores, params, csv_path="./data/mission_results.csv"):

   # Create dictionary with results
   results = {
       'timestamp': time.strftime("%Y-%m-%d %X"),
       'hash': hashVal,
       'mission2_score': scores[0],
       'mission3_score': scores[1],
       'mission2_total_mass': params[0].m_total,
       'mission2_throttle_climb': params[0].throttle_climb,
       'mission2_throttle_turn': params[0].throttle_turn,
       'mission2_throttle_level': params[0].throttle_level,
       'mission3_total_mass': params[1].m_total,
       'mission3_throttle_climb': params[1].throttle_climb,
       'mission3_throttle_turn': params[1].throttle_turn, 
       'mission3_throttle_level': params[1].throttle_level
   }

   # Convert to DataFrame
   df = pd.DataFrame([results])

   # Append to CSV or create new one
   try:
       df.to_csv(csv_path, mode='a',  index=False)
   except Exception as e:
       print(f"Failed to save results: {str(e)}")

