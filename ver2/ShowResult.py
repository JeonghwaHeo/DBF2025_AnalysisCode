from internal_dataclass import *
from setup_dataclass import *
from vsp_grid import *
from vsp_analysis import *
from mission_analysis import *


if __name__ == "__main__":


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

    #################### < 채워넣어야 할 부분 > ########################### 
    showAircraft = Aircraft(
        m_fuselage = 2500,
        wing_area_blocked_by_fuselage = 72640,    #mm2
        wing_density = 0.0000588,

        mainwing_span = 1800,        
        mainwing_AR = 5.45,           
        mainwing_taper = 0.9,        
        mainwing_twist = 2.0,        
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
        
        mainwing_airfoil_datapath = "data/airfoilDAT/e216.dat",
        horizontal_airfoil_datapath= "data/airfoilDAT/naca0008.dat",
        vertical_airfoil_datapath= "data/airfoilDAT/naca0009.dat"
        
        )
    ##################################################################

    vspAnalyzer = VSPAnalyzer(presetValues)
    vspAnalyzer.setup_vsp_model(aircraft = showAircraft,vspPath="ShowMothership.vsp3")

    alpha_start = aerodynamicSetup.alpha_start
    alpha_end = aerodynamicSetup.alpha_end
    alpha_step = aerodynamicSetup.alpha_step
    CD_fuse = get_fuselageCD_list(alpha_start,alpha_end,alpha_step,aerodynamicSetup.fuselage_Cd_datapath)
    fuselage_cross_section_area = aerodynamicSetup.fuselage_cross_section_area
    vspPath = f"ShowMothership.vsp3"

    analResults = vspAnalyzer.calculateCoefficients(
                        alpha_start = alpha_start, alpha_end = alpha_end, alpha_step = alpha_step,
                        CD_fuse = CD_fuse, fuselage_cross_section_area = fuselage_cross_section_area, 
                        wing_area_blocked_by_fuselage = showAircraft.wing_area_blocked_by_fuselage,

                        fileName=vspPath,
                        
                        AOA_stall = aerodynamicSetup.AOA_stall,
                        AOA_takeoff_max = aerodynamicSetup.AOA_takeoff_max,
                        AOA_climb_max = aerodynamicSetup.AOA_climb_max,
                        AOA_turn_max = aerodynamicSetup.AOA_turn_max,
                        
                        clearModel=False
                        )
    
    visualize_results(analResults)

    #################### < 채워넣어야 할 부분 > ########################### 
    mission2Params = MissionParameters(
        m_takeoff = 10,
        max_speed= 40,                      
        max_load_factor = presetValues.max_load / 10,          
                
        climb_thrust_ratio = 0.9,
        level_thrust_ratio = 0.6,
        turn_thrust_ratio = 0.6,   

        propeller_data_path=propulsionSpecs.M2_propeller_data_path,
    )    

    mission3Params = MissionParameters(
        m_takeoff = analResults.m_empty/1000,
        max_speed= 25,                      
        max_load_factor = presetValues.max_load * 1000 / analResults.m_empty,            
                
        climb_thrust_ratio = 0.9,
        level_thrust_ratio = 0.4,
        turn_thrust_ratio = 0.4,   

        propeller_data_path=propulsionSpecs.M3_propeller_data_path
    )
    ################################################################## 


    mission2Analyzer = MissionAnalyzer(analResults, mission2Params, presetValues, propulsionSpecs)
    mission2Analyzer.run_mission2()
    visualize_mission(mission2Analyzer.stateLog)

    mission3Analyzer = MissionAnalyzer(analResults, mission3Params, presetValues, propulsionSpecs)
    mission3Analyzer.run_mission3()
    visualize_mission(mission3Analyzer.stateLog)
