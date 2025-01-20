## Does all the main work

import cProfile
import pstats
from pstats import SortKey

from vsp_analysis import VSPAnalyzer, writeAnalysisResults, loadAnalysisResults, visualize_results, compare_aerodynamics
from mission_analysis import MissionAnalyzer, visualize_mission
from models import *

def main():
    presetValues = PresetValues(
        m_x1 = 0.2,                       # kg
        x1_flight_time = 30,              # sec
        max_battery_capacity = 2250,      # mAh (per one battery)
        Thrust_max = 6.6,                 # kg (two motors)
        min_battery_voltage = 25,         # V (원래는 3 x 6 = 18 V 인데 안전하게 20 V)
        propulsion_efficiency = 0.8,      # Efficiency of the propulsion system
        score_weight_ratio = 1            # mission2/3 score weight ratio
        )

    missionParam = MissionParameters(
        max_battery_capacity = 2250,      # mAh (per one battery)
        throttle_takeoff = 0.9,           # %
        throttle_climb = 0.9,             # %
        throttle_level = 0.6,             # %
        throttle_turn = 0.55,             # %
        max_climb_angle = 40,             # degree
        max_speed = 40,                   # m/s
        max_load_factor = 4.0,
        h_flap_transition = 5             # m
        )

    a=loadAnalysisResults(687192594661440415)
    b=loadAnalysisResults(1676891088784291821)
    c=loadAnalysisResults(1268481079834018136)
    d=loadAnalysisResults(1446022752061654911)
    #compare_aerodynamics([a,b,c,d])
    missionAnalyzer = MissionAnalyzer(a, missionParam, presetValues)
    
    profiler = cProfile.Profile()
    profiler.enable()

    missionAnalyzer.run_mission3()

    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(30)  # Print top 30 functions
    
    # Print stats sorted by total time
    print("\nStats sorted by total time:")
    stats.sort_stats(SortKey.TIME).print_stats(30)

    
    visualize_mission(missionAnalyzer.stateLog) 
    
    return
   
    aircraft = Aircraft(
            m_total=6000,m_fuselage=3000,
 
            wing_density=0.0000852, spar_density=1.0,
 
            mainwing_span=1800.0,        
            mainwing_AR=5.45,           
            mainwing_taper=0.65,        
            mainwing_twist=0,        
            mainwing_sweepback=0,   
            mainwing_dihedral=5.0,     
            mainwing_incidence=2.0,    
 
            flap_start=[0.05,0.4],            
            flap_end=[0.25,0.6],              
            flap_angle=[20.0,15.0],           
            flap_c_ratio=[0.35,0.35],         
 
            horizontal_volume_ratio=0.7,
            horizontal_area_ratio=0.25, 
            horizontal_AR=4.0,         
            horizontal_taper=1,      
            horizontal_ThickChord=1,
 
            vertical_volume_ratio=0.05,
            vertical_taper=0.7,        
            vertical_ThickChord=0.08   
            )
 
    vspAnalyzer = VSPAnalyzer(presetValues)
    vspAnalyzer.setup_vsp_model(aircraft)
    analResults = vspAnalyzer.calculateCoefficients(
            alpha_start=-5,alpha_end=10,alpha_step=1,
            CD_fuse=np.zeros(15),
 
            AOA_stall=10, 
            AOA_takeoff_max=10,
            AOA_climb_max=10,
            AOA_turn_max=20,
 
            clearModel=False)
    visualize_results(analResults)
    writeAnalysisResults(analResults)

if __name__== "__main__":
    main()
