import numpy as np
from config import *
from vsp_analysis import VSPAnalyzer, writeAnalysisResults, loadAnalysisResults, visualize_results
from models import *

aircraftParamConstraint = AircraftParamConstraints (
    #Constraints for constructing the aircraf
    # total mass of the aircraft
    m_total_max = float,
    m_total_min = float,
    m_total_interval = float,

    # wing parameter ranges
    span_max = 1800.0,                   # mm
    span_min = 1800.0,
    span_interval = 100.0,
    AR_max = 5.45,
    AR_min = 5.45,
    AR_interval = 0.5,
    taper_max = 0.65,                       # (root chord) / (tip chord)
    taper_min = 0.65,
    taper_interval = 0.1,
    twist_max = 0.0,                       # degree
    twist_min = 0.0,
    twist_interval = 1.0,
)

missionParamConstraints = MissionParamConstraints (
    #Constraints for calculating missions
    throttle_climb_min = 0.7,
    throttle_turn_min = 0.4,
    throttle_level_min = 0.7,
    throttle_climb_max = 1.0,
    throttle_turn_max = 0.7,
    throttle_level_max = 1.0,
    throttle_analysis_interval = 0.05,
)

presetValues = PresetValues(
    m_x1 = 0.2,                       # kg
    x1_flight_time = 30,              # sec
    max_battery_capacity = 2250,      # mAh (per one battery)
    Thrust_max = 6.6,                 # kg (two motors)
    min_battery_voltage = 20,         # V (원래는 3 x 6 = 18 V 인데 안전하게 20 V)
    propulsion_efficiency = 0.8,      # Efficiency of the propulsion system
    score_weight_ratio = 1            # mission2/3 score weight ratio
    )

## Variable lists using for optimization
span_list = np.arange(aircraftParamConstraint.span_min, aircraftParamConstraint.span_max + aircraftParamConstraint.span_interval, aircraftParamConstraint.span_interval) # (min, max + step_size, step_size)
AR_list = np.arange(aircraftParamConstraint.AR_min, aircraftParamConstraint.AR_max + aircraftParamConstraint.AR_interval, aircraftParamConstraint.AR_interval)
taper_list = np.arange(aircraftParamConstraint.taper_min, aircraftParamConstraint.taper_max + aircraftParamConstraint.taper_interval, aircraftParamConstraint.taper_interval)
twist_list = np.arange(aircraftParamConstraint.twist_min, aircraftParamConstraint.twist_max + aircraftParamConstraint.twist_interval, aircraftParamConstraint.twist_interval)

throttle_climb_list = np.arange(missionParamConstraints.throttle_climb_min, missionParamConstraints.throttle_climb_max + missionParamConstraints.throttle_analysis_interval, missionParamConstraints.throttle_analysis_interval)
throttle_turn_list = np.arange(missionParamConstraints.throttle_turn_min, missionParamConstraints.throttle_turn_max + missionParamConstraints.throttle_analysis_interval, missionParamConstraints.throttle_analysis_interval)
throttle_level_list = np.arange(missionParamConstraints.throttle_level_min, missionParamConstraints.throttle_climb_max + missionParamConstraints.throttle_analysis_interval, missionParamConstraints.throttle_analysis_interval)

print(f"\nSpan list: {span_list}")
print(f"Span list: {AR_list}")
print(f"Span list: {taper_list}")
print(f"Span list: {twist_list}")

print(f"\nthrottle climb list: {throttle_climb_list}")
print(f"throttle turn list: {throttle_turn_list}")
print(f"throttle level list: {throttle_level_list}\n")

for span in span_list:
    for AR in AR_list:
        for taper in taper_list:
            for twist in twist_list:
                aircraft = Aircraft(
                        m_total = 8500, m_fuselage = 5000,

                        wing_density = 0.0000852, spar_density = 1.0,

                        mainwing_span = span,        
                        mainwing_AR = AR,           
                        mainwing_taper = taper,        
                        mainwing_twist = twist,        
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

                vspAnalyzer = VSPAnalyzer(presetValues)
                vspAnalyzer.setup_vsp_model(aircraft)
                analResults = vspAnalyzer.calculateCoefficients(
                        alpha_start = -3.5, alpha_end = 13, alpha_step = 0.5,
                        CD_fuse = np.full(int(round((13 - (-3.5)) / 0.5)) + 1, 0.03),

                        AOA_stall = 13,
                        AOA_takeoff_max = 10,
                        AOA_climb_max = 8,
                        AOA_turn_max = 8,
                                        
                        # CL_max = 0.94,
                        # CL_flap_max = 1.1,
                        # CL_flap_zero=0.04,
                        # CD_flap_max=0.20,
                        # CD_flap_zero=0.10,

                        clearModel=False)
                writeAnalysisResults(analResults)