import cProfile
import pstats
import pandas as pd
from pstats import SortKey
from vsp_grid import runVSPGridAnalysis
from mission_grid import runMissionGridSearch, ResultAnalysis
from vsp_analysis import  loadAnalysisResults, visualize_results, resetAnalysisResults, removeAnalysisResults
from mission_analysis import MissionAnalyzer, visualize_mission
from models import *
from config import *

def main():

    removeAnalysisResults(csvPath = "data/test.csv")
    removeAnalysisResults(csvPath = "data/total_results.csv")
    removeAnalysisResults(csvPath = "data/organized_results.csv")

    presetValues = PresetValues(
        m_x1 = 0.25,                        # kg
        x1_flight_time = 30,                # sec
        max_battery_capacity = 2250,        # mAh (per one battery)
        Thrust_max = 6.6,                   # kg (two motors)
        min_battery_voltage = 21.8,         # V (3.6 V 을 넘으면 방전이 아주 급격해짐)
        propulsion_efficiency = 0.1326,     # Efficiency of the propulsion system
        score_weight_ratio = 0.5            # mission2/3 score weight ratio
        )
    
    aircraftParamConstraints = AircraftParamConstraints (
        #Constraints for constructing the aircraft

        m_total_min = 8500.0,                # g
        m_total_max = 8600.0,
        m_total_interval = 100.0,
        
        # wing parameter ranges
        span_min = 1600.0,                   # mm
        span_max = 1800.0,                   
        span_interval = 100.0,
    
        AR_min = 5.45,                  
        AR_max = 5.45,
        AR_interval = 0.5,
        
        taper_min = 0.55,
        taper_max = 0.55,                      
        taper_interval = 0.1,
        
        twist_min = 0.0,                     # degree
        twist_max = 0.0,                     
        twist_interval = 1.0,

        # wing loading limit
        wing_loading_min = 1,
        wing_loading_max = 17
        )
    
    baseAircraft = Aircraft(
        m_total = 8500, 
        m_fuselage = 3000,

        wing_density = 0.0000852, spar_density = 1.0,

        mainwing_span = 1800,        
        mainwing_AR = 5.45,           
        mainwing_taper = 0.65,        
        mainwing_twist = 0.0,        
        mainwing_sweepback = 0,   
        mainwing_dihedral = 5.0,     
        mainwing_incidence = 2.0,    

        flap_start = [0.05, 0.4],            
        flap_end = [0.25, 0.6],              
        flap_angle = [20.0, 15.0],           
        flap_c_ratio = [0.35, 0.35],         

        horizontal_volume_ratio = 0.7,
        horizontal_area_ratio = 0.25, 
        horizontal_AR = 4.0,         
        horizontal_taper = 1,      
        horizontal_ThickChord = 8,

        vertical_volume_ratio = 0.053,
        vertical_taper = 0.6,        
        vertical_ThickChord = 9  
        )

    runVSPGridAnalysis(aircraftParamConstraints, presetValues, baseAircraft)

    results = pd.read_csv("data/test.csv", sep='|', encoding='utf-8')
    print(results.head()) 

    for hashVal in results["hash"]:
        print(f"\nAnalyzing for hash{hashVal}")

        missionParamConstraints = MissionParamConstraints (
            #Constraints for calculating mission2
            M2_throttle_climb_min = 0.9,
            M2_throttle_climb_max = 0.9,
            M2_throttle_turn_min = 0.5,
            M2_throttle_turn_max = 0.5,
            M2_throttle_level_min = 0.5,
            M2_throttle_level_max = 0.6,
            M2_throttle_analysis_interval = 0.05,

            #Constraints for calculating mission3  
            M3_throttle_climb_min = 0.9,
            M3_throttle_climb_max = 0.9,
            M3_throttle_turn_min = 0.5,
            M3_throttle_turn_max = 0.5,
            M3_throttle_level_min = 0.6,
            M3_throttle_level_max = 0.6,
            M3_throttle_analysis_interval = 0.05
            )
        
        runMissionGridSearch(hashVal,missionParamConstraints,presetValues)
        
    
    ResultAnalysis(presetValues)

    return


if __name__== "__main__":
    main()
