import argparse
from vsp_analysis import loadAnalysisResults, visualize_results
import pandas as pd
from mission_analysis import MissionAnalyzer, visualize_mission, get_state_df
from internal_dataclass import MissionParameters
from setup_dataclass import PresetValues, PropulsionSpecs
import numpy as np

def get_result_by_id(resultID:str, mission_number : int, server_id : int) ->pd.DataFrame:
    
    if mission_number == 2:
        csvPath = f"data/mission2_results_{server_id}.csv"
    elif mission_number == 3: 
        csvPath = f"data/mission3_results_{server_id}.csv"

    resultID_df = pd.read_csv(csvPath, sep='|',encoding='utf-8')
    resultID_df = resultID_df[resultID_df['resultID'] == resultID ]
    return resultID_df
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="module that displays the result screen which user wants.")
    parser.add_argument("--server_id", type=int, required=False, help="Specify the server ID")

    subparsers = parser.add_subparsers(dest="main_command", required=True)
    show_parser = subparsers.add_parser("show", help="Show results for specific resultID.")
    
    show_subparsers = show_parser.add_subparsers(dest="type", required=True)
    
    show_aircraft_parser = show_subparsers.add_parser("aircraft", help="Show aircraft analysis results.")
    show_aircraft_parser.add_argument("hashVal", type=str, help="Enter the aircraft hash which you want to check.")
    
    show_mission_parser = show_subparsers.add_parser("mission2", help="Show aircraft analysis results.")
    show_mission_parser.add_argument("resultID", type=str, help="Enter the resultID which you want to check.")
    
    show_mission_parser = show_subparsers.add_parser("mission3", help="Show aircraft analysis results.")
    show_mission_parser.add_argument("resultID", type=str, help="Enter the resultID which you want to check.")
    
    save_parser = subparsers.add_parser("save", help="Save results for specific resultID.")
    save_subparsers = save_parser.add_subparsers(dest="type", required=True)

    save_mission_parser = save_subparsers.add_parser("mission2", help="Save mission2 analysis results.")
    save_mission_parser.add_argument("resultID", type=str, help="Enter the resultID which you want to save.")
    
    save_mission_parser = save_subparsers.add_parser("mission3", help="Save mission2 analysis results.")
    save_mission_parser.add_argument("resultID", type=str, help="Enter the resultID which you want to save.")

    
   
    
    args = parser.parse_args()

 
    if args.main_command == "show":
        if args.type == "aircraft":
            hashVal = "'" + args.hashVal +"'"
            aircraft_result = loadAnalysisResults(hashVal,f"data/aircraft.csv")
            visualize_results(aircraft_result)
            
        elif args.type == "mission2":
            resultID = "'" + args.resultID + "'"
            resultID_df = get_result_by_id(resultID,2,args.server_id)
            hashVal = resultID_df['hash']  
            aircraft = loadAnalysisResults(hashVal.iloc[0],"data/aircraft.csv")     
            param2 = MissionParameters(
                m_takeoff= resultID_df['MTOW'].iloc[0],
                max_speed= resultID_df['M2_max_speed'].iloc[0],                   
                max_load_factor = resultID_df['max_load'].iloc[0]/resultID_df['MTOW'].iloc[0],  
                                          
                climb_thrust_ratio = resultID_df['mission2_climb_thrust_ratio'].iloc[0],
                level_thrust_ratio = resultID_df['mission2_level_thrust_ratio'].iloc[0],
                turn_thrust_ratio = resultID_df['mission2_turn_thrust_ratio'].iloc[0],
                
                propeller_data_path = resultID_df['M2_propeller_data_path'].iloc[0]                          

            )
            
            presetValues = PresetValues(
                m_x1= resultID_df['m_x1'].iloc[0],
                x1_time_margin= resultID_df['x1_time_margin'].iloc[0],
                
                throttle_takeoff = resultID_df['throttle_takeoff'].iloc[0],
                max_climb_angle = resultID_df['max_climb_angle'].iloc[0],
                max_load = resultID_df['max_load'].iloc[0],
                h_flap_transition = resultID_df['h_flap_transition'].iloc[0],
                
                number_of_motor= resultID_df['number_of_motor'].iloc[0],
                min_battery_voltage= resultID_df['min_battery_voltage'].iloc[0],
                score_weight_ratio= resultID_df['score_weight_ratio'].iloc[0]               

            )
              
            propulsionSpecs = PropulsionSpecs(
                M2_propeller_data_path = resultID_df['M2_propeller_data_path'].iloc[0],
                M3_propeller_data_path = resultID_df['M3_propeller_data_path'].iloc[0],
                battery_data_path = resultID_df['battery_data_path'].iloc[0],
                Kv = resultID_df['Kv'].iloc[0],
                R = resultID_df['R'].iloc[0],
                number_of_battery= resultID_df['number_of_battery'].iloc[0],
                n_cell = resultID_df['n_cell'].iloc[0], 
                battery_Wh= resultID_df['battery_Wh'].iloc[0],
                max_current = resultID_df['max_current'].iloc[0],
                max_power = resultID_df['max_power'].iloc[0]    
            )  
              
                
            missionAnalyzer2 = MissionAnalyzer(aircraft,param2,presetValues, propulsionSpecs)
            missionAnalyzer2.run_mission2()
            visualize_mission(missionAnalyzer2.stateLog)  

        elif args.type == "mission3":
            resultID = "'" + args.resultID + "'"
            resultID_df = get_result_by_id(resultID,3,args.server_id)
            hashVal = resultID_df['hash']  
            aircraft = loadAnalysisResults(hashVal.iloc[0],"data/aircraft.csv")     

            param3 = MissionParameters(
                m_takeoff= resultID_df['m_empty'].iloc[0]/1000,
                max_speed= resultID_df['M3_max_speed'].iloc[0],                   
                max_load_factor = resultID_df['max_load'].iloc[0] * 1000 /resultID_df['m_empty'].iloc[0],  
                                          
                climb_thrust_ratio = resultID_df['mission3_climb_thrust_ratio'].iloc[0],
                level_thrust_ratio = resultID_df['mission3_level_thrust_ratio'].iloc[0],
                turn_thrust_ratio = resultID_df['mission3_turn_thrust_ratio'].iloc[0],
                
                propeller_data_path = resultID_df['M3_propeller_data_path'].iloc[0],                           

            )
            
            presetValues = PresetValues(
                m_x1= resultID_df['m_x1'].iloc[0],
                x1_time_margin= resultID_df['x1_time_margin'].iloc[0],
                
                throttle_takeoff = resultID_df['throttle_takeoff'].iloc[0],
                max_climb_angle = resultID_df['max_climb_angle'].iloc[0],
                max_load = resultID_df['max_load'].iloc[0],
                h_flap_transition = resultID_df['h_flap_transition'].iloc[0],
                
                number_of_motor= resultID_df['number_of_motor'].iloc[0],
                min_battery_voltage= resultID_df['min_battery_voltage'].iloc[0],
                score_weight_ratio= resultID_df['score_weight_ratio'].iloc[0]    

            )
             
            propulsionSpecs = PropulsionSpecs(
                M2_propeller_data_path = resultID_df['M2_propeller_data_path'].iloc[0],
                M3_propeller_data_path = resultID_df['M3_propeller_data_path'].iloc[0],
                battery_data_path = resultID_df['battery_data_path'].iloc[0],
                Kv = resultID_df['Kv'].iloc[0],
                R = resultID_df['R'].iloc[0],
                number_of_battery= resultID_df['number_of_battery'].iloc[0],
                n_cell = resultID_df['n_cell'].iloc[0], 
                battery_Wh= resultID_df['battery_Wh'].iloc[0],
                max_current = resultID_df['max_current'].iloc[0],
                max_power = resultID_df['max_power'].iloc[0]    
            )  
                     
            missionAnalyzer3 = MissionAnalyzer(aircraft,param3,presetValues,propulsionSpecs)
            missionAnalyzer3.run_mission3()
            visualize_mission(missionAnalyzer3.stateLog) 


    if args.main_command == "save":    
        if args.type == "mission2":
            resultID = "'" + args.resultID + "'"
            resultID_df = get_result_by_id(resultID,2,args.server_id)
            hashVal = resultID_df['hash']  
            aircraft = loadAnalysisResults(hashVal.iloc[0],"data/aircraft.csv")     
            param2 = MissionParameters(
                m_takeoff= resultID_df['MTOW'].iloc[0],
                max_speed= resultID_df['M2_max_speed'].iloc[0],                   
                max_load_factor = resultID_df['max_load'].iloc[0]/resultID_df['MTOW'].iloc[0],  
                                          
                climb_thrust_ratio = resultID_df['mission2_climb_thrust_ratio'].iloc[0],
                level_thrust_ratio = resultID_df['mission2_level_thrust_ratio'].iloc[0],
                turn_thrust_ratio = resultID_df['mission2_turn_thrust_ratio'].iloc[0],
                
                propeller_data_path = resultID_df['M2_propeller_data_path'].iloc[0]                          

            )
            
            presetValues = PresetValues(
                m_x1= resultID_df['m_x1'].iloc[0],
                x1_time_margin= resultID_df['x1_time_margin'].iloc[0],
                
                throttle_takeoff = resultID_df['throttle_takeoff'].iloc[0],
                max_climb_angle = resultID_df['max_climb_angle'].iloc[0],
                max_load = resultID_df['max_load'].iloc[0],
                h_flap_transition = resultID_df['h_flap_transition'].iloc[0],
                
                number_of_motor= resultID_df['number_of_motor'].iloc[0],
                min_battery_voltage= resultID_df['min_battery_voltage'].iloc[0],
                score_weight_ratio= resultID_df['score_weight_ratio'].iloc[0]               

            )
              
            propulsionSpecs = PropulsionSpecs(
                M2_propeller_data_path = resultID_df['M2_propeller_data_path'].iloc[0],
                M3_propeller_data_path = resultID_df['M3_propeller_data_path'].iloc[0],
                battery_data_path = resultID_df['battery_data_path'].iloc[0],
                Kv = resultID_df['Kv'].iloc[0],
                R = resultID_df['R'].iloc[0],
                number_of_battery= resultID_df['number_of_battery'].iloc[0],
                n_cell = resultID_df['n_cell'].iloc[0], 
                battery_Wh= resultID_df['battery_Wh'].iloc[0],
                max_current = resultID_df['max_current'].iloc[0],
                max_power = resultID_df['max_power'].iloc[0]    
            )  
              
                
            missionAnalyzer2 = MissionAnalyzer(aircraft,param2,presetValues, propulsionSpecs)
            missionAnalyzer2.run_mission2()
            stateLog = get_state_df(missionAnalyzer2.stateLog)
            stateLog.to_csv("data/mission_statelog.csv",index=False,encoding="utf-8")

        elif args.type == "mission3":
            resultID = "'" + args.resultID + "'"
            resultID_df = get_result_by_id(resultID,3,args.server_id)
            hashVal = resultID_df['hash']  
            aircraft = loadAnalysisResults(hashVal.iloc[0],"data/aircraft.csv")     

            param3 = MissionParameters(
                m_takeoff= resultID_df['m_empty'].iloc[0]/1000,
                max_speed= resultID_df['M3_max_speed'].iloc[0],                   
                max_load_factor = resultID_df['max_load'].iloc[0] * 1000 /resultID_df['m_empty'].iloc[0],  
                                          
                climb_thrust_ratio = resultID_df['mission3_climb_thrust_ratio'].iloc[0],
                level_thrust_ratio = resultID_df['mission3_level_thrust_ratio'].iloc[0],
                turn_thrust_ratio = resultID_df['mission3_turn_thrust_ratio'].iloc[0],
                
                propeller_data_path = resultID_df['M3_propeller_data_path'].iloc[0],                           

            )
            
            presetValues = PresetValues(
                m_x1= resultID_df['m_x1'].iloc[0],
                x1_time_margin= resultID_df['x1_time_margin'].iloc[0],
                
                throttle_takeoff = resultID_df['throttle_takeoff'].iloc[0],
                max_climb_angle = resultID_df['max_climb_angle'].iloc[0],
                max_load = resultID_df['max_load'].iloc[0],
                h_flap_transition = resultID_df['h_flap_transition'].iloc[0],
                
                number_of_motor= resultID_df['number_of_motor'].iloc[0],
                min_battery_voltage= resultID_df['min_battery_voltage'].iloc[0],
                score_weight_ratio= resultID_df['score_weight_ratio'].iloc[0]    

            )
             
            propulsionSpecs = PropulsionSpecs(
                M2_propeller_data_path = resultID_df['M2_propeller_data_path'].iloc[0],
                M3_propeller_data_path = resultID_df['M3_propeller_data_path'].iloc[0],
                battery_data_path = resultID_df['battery_data_path'].iloc[0],
                Kv = resultID_df['Kv'].iloc[0],
                R = resultID_df['R'].iloc[0],
                number_of_battery= resultID_df['number_of_battery'].iloc[0],
                n_cell = resultID_df['n_cell'].iloc[0], 
                battery_Wh= resultID_df['battery_Wh'].iloc[0],
                max_current = resultID_df['max_current'].iloc[0],
                max_power = resultID_df['max_power'].iloc[0]    
            )  
                     
            missionAnalyzer3 = MissionAnalyzer(aircraft,param3,presetValues, propulsionSpecs)
            missionAnalyzer3.run_mission3()
            stateLog = get_state_df(missionAnalyzer3.stateLog)
            stateLog.to_csv("data/mission_statelog.csv",index=False,encoding="utf-8")
