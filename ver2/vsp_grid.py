import numpy as np
from itertools import product
from dataclasses import replace
import time
import pandas as pd
from scipy.interpolate import interp1d
from setup_dataclass import *
from vsp_analysis import VSPAnalyzer, writeAnalysisResults
from internal_dataclass import *


def runVSPGridAnalysis(aircraftParamConstraint: AircraftParamConstraints,aerodynamicSetup: AerodynamicSetup, presetValues: PresetValues, baseAircraft: Aircraft, server_id : int=1, total_server : int=1,csvPath: str = "",vspPath: str=""):
        
        ## Variable lists using for optimization
        span_list = np.arange(
                aircraftParamConstraint.span_min, 
                aircraftParamConstraint.span_max + aircraftParamConstraint.span_interval/2, 
                aircraftParamConstraint.span_interval
                )
        AR_list = np.arange(
                aircraftParamConstraint.AR_min, 
                aircraftParamConstraint.AR_max + aircraftParamConstraint.AR_interval/2, 
                aircraftParamConstraint.AR_interval
                )
        taper_list = np.arange(
                aircraftParamConstraint.taper_min, 
                aircraftParamConstraint.taper_max + aircraftParamConstraint.taper_interval/2, 
                aircraftParamConstraint.taper_interval
                )
        twist_list = np.arange(
                aircraftParamConstraint.twist_min, 
                aircraftParamConstraint.twist_max + aircraftParamConstraint.twist_interval/2, 
                aircraftParamConstraint.twist_interval
                )
        airfoil_list = aircraftParamConstraint.airfoil_list
        
        total_grid_combinations = list(product(span_list,AR_list,taper_list,twist_list,airfoil_list))
        grid_chunks = list(split_into_chunks(total_grid_combinations,total_server))
        vsp_grid_combinations = grid_chunks[server_id-1]

        total = len(vsp_grid_combinations)
        print(f"\nTotal number of Aircraft combinations: {total}")
         
        alpha_start = aerodynamicSetup.alpha_start
        alpha_end = aerodynamicSetup.alpha_end
        alpha_step = aerodynamicSetup.alpha_step
        CD_fuse = get_fuselageCD_list(alpha_start,alpha_end,alpha_step,aerodynamicSetup.fuselage_Cd_datapath)
        fuselage_cross_section_area = aerodynamicSetup.fuselage_cross_section_area
        
        vspAnalyzer = VSPAnalyzer(presetValues)

        step = max(int(total/100) , 1)

        for i, (span, AR, taper, twist, airfoil_name) in enumerate(vsp_grid_combinations):
                
                if (i+1)%step==0:
                        print(f"\n[{time.strftime('%Y-%m-%d %X')}] VSP Grid Progress: {i+1}/{total} configurations: [{span:.2f}, {AR:.2f}, {taper:.2f}, {twist:.1f}, {airfoil_name}]")

                airfoil_datapath = "data/airfoilDAT/" + airfoil_name + ".dat"
                aircraft = replace(baseAircraft, mainwing_span = span, mainwing_AR = AR , mainwing_taper = taper, mainwing_twist = twist, mainwing_airfoil_datapath = airfoil_datapath)   

                vspAnalyzer.setup_vsp_model(aircraft,vspPath=vspPath)
                analResults = vspAnalyzer.calculateCoefficients(
                        alpha_start = alpha_start, alpha_end = alpha_end, alpha_step = alpha_step,
                        CD_fuse = CD_fuse, fuselage_cross_section_area = fuselage_cross_section_area, 
                        wing_area_blocked_by_fuselage = baseAircraft.wing_area_blocked_by_fuselage,

                        fileName=vspPath,
                        
                        AOA_stall = aerodynamicSetup.AOA_stall,
                        AOA_takeoff_max = aerodynamicSetup.AOA_takeoff_max,
                        AOA_climb_max = aerodynamicSetup.AOA_climb_max,
                        AOA_turn_max = aerodynamicSetup.AOA_turn_max,
                        
                        clearModel=False
                        )

                writeAnalysisResults(analResults,csvPath=csvPath)
                vspAnalyzer.clean()


def get_fuselageCD_list(alpha_start,alpha_end,alpha_step,csvPath):
        df = pd.read_csv(csvPath)
        alpha_list = df['AOA(degree)'].to_numpy()
        Cd_fuse_list = df['CD fuselage'].to_numpy()
        Cd_fuse_func = interp1d(alpha_list, Cd_fuse_list, kind="quadratic", fill_value="extrapolate") 
        alpha = np.arange(alpha_start,alpha_end + alpha_step/2, alpha_step)  
        CD_fuse = Cd_fuse_func(alpha)
        return CD_fuse
        
def split_into_chunks(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i*k + min(i, m) : (i+1)*k + min(i+1, m)] for i in range(n))

if __name__ == "__main__":
    runVSPGridAnalysis(
            AircraftParamConstraints (
                # wing parameter ranges
                span_max = 1800.0,                     # mm
                span_min = 1800.0,
                span_interval = 100.0,
                AR_max = 5.45,
                AR_min = 5.45,
                AR_interval = 0.05,
                taper_max = 0.45,                      # (root chord) / (tip chord)
                taper_min = 0.45,
                taper_interval = 0.05,
                twist_max = 0.0,                       # degree
                twist_min = 0.0,
                twist_interval = 1.0,
                airfoil_list = ['E852','MH122','S4062','S9027','HQ3010','HQ3011']
                ),
            PresetValues(
                    m_x1 = 0.2,                       # kg
                    x1_time_margin = 30,              # sec
                    Thrust_max = 6.6,                 # kg (two motors)
                    min_battery_voltage = 20,         # V (원래는 3 x 6 = 18 V 인데 안전하게 20 V)
                    score_weight_ratio = 1            # mission2/3 score weight ratio
                    ), 
            Aircraft(
               m_fuselage = 5000,

               wing_density = 0.0000852,

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
               ))

