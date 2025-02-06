from typing import List
import numpy as np
import pandas as pd
import time
import concurrent.futures
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from setup_dataclass import PresetValues, PropulsionSpecs
from internal_dataclass import PhysicalConstants, MissionParameters, AircraftAnalysisResults, PlaneState, PhaseType, MissionConfig, Aircraft
from propulsion import thrust_analysis, determine_max_thrust, thrust_reverse_solve, SoC2Vol
from vsp_analysis import  loadAnalysisResults


## Constant values
g = PhysicalConstants.g
rho = PhysicalConstants.rho

class MissionAnalyzer():
    def __init__(self, 
                 analResult:AircraftAnalysisResults, 
                 missionParam:MissionParameters, 
                 presetValues:PresetValues,
                 propulsionSpecs : PropulsionSpecs,
                 dt:float=0.1):

        self.analResult = self._convert_units(analResult, presetValues)
        self.aircraft = self.analResult.aircraft
        self.missionParam = missionParam
        self.presetValues = presetValues
        self.propulsionSpecs = propulsionSpecs
        self.dt = dt
        self.m_fuel = max(self.missionParam.m_takeoff - self.analResult.m_empty,0) 

        self.convert_propellerCSV_to_ndarray(self.missionParam.propeller_data_path)
        self.convert_batteryCSV_to_ndarray(self.propulsionSpecs.battery_data_path)
        self.clearState()
        self.setAuxVals()

    def _convert_units(self, results: AircraftAnalysisResults, presetValues:PresetValues) -> AircraftAnalysisResults:
        # Create new aircraft instance with converted units
        new_aircraft = Aircraft(
            # Mass conversions (g to kg)
            m_fuselage=results.aircraft.m_fuselage / 1000,
            wing_area_blocked_by_fuselage= results.aircraft.wing_area_blocked_by_fuselage / 1e6, 
            # Density conversions (g/mm³ to kg/m³)
            wing_density=results.aircraft.wing_density * 1e9,
            
            # Length conversions (mm to m)
            mainwing_span=results.aircraft.mainwing_span / 1000,
            
            # These are ratios, no conversion needed
            mainwing_AR=results.aircraft.mainwing_AR,
            mainwing_taper=results.aircraft.mainwing_taper,
            mainwing_twist=results.aircraft.mainwing_twist,
            mainwing_sweepback=results.aircraft.mainwing_sweepback,
            mainwing_dihedral=results.aircraft.mainwing_dihedral,
            mainwing_incidence=results.aircraft.mainwing_incidence,
            
            # Lists of ratios/angles, no conversion needed
            flap_start=results.aircraft.flap_start,
            flap_end=results.aircraft.flap_end,
            flap_angle=results.aircraft.flap_angle,
            flap_c_ratio=results.aircraft.flap_c_ratio,
            
            # Ratios and angles, no conversion needed
            horizontal_volume_ratio=results.aircraft.horizontal_volume_ratio,
            horizontal_area_ratio=results.aircraft.horizontal_area_ratio,
            horizontal_AR=results.aircraft.horizontal_AR,
            horizontal_taper=results.aircraft.horizontal_taper,
            horizontal_ThickChord=results.aircraft.horizontal_ThickChord,
            vertical_volume_ratio=results.aircraft.vertical_volume_ratio,
            vertical_taper=results.aircraft.vertical_taper,
            vertical_ThickChord=results.aircraft.vertical_ThickChord,
            mainwing_airfoil_datapath=results.aircraft.mainwing_airfoil_datapath,
            horizontal_airfoil_datapath=results.aircraft.horizontal_airfoil_datapath,
            vertical_airfoil_datapath=results.aircraft.vertical_airfoil_datapath
            
        )
        
        # Create new analysis results with converted units
        return AircraftAnalysisResults(
            aircraft=new_aircraft,
            alpha_list=results.alpha_list,
            
            # Mass conversions (g to kg)
            m_empty=results.m_empty / 1000,
            m_boom=results.m_boom / 1000,
            m_wing=results.m_wing / 1000,
            
            # Length conversions (mm to m)
            span=results.span / 1000,
            
            # These are ratios, no conversion needed
            AR=results.AR,
            taper=results.taper,
            twist=results.twist,
            
            # Area conversion (mm² to m²)
            Sref=results.Sref / 1e6,
            
            # Length conversions (mm to m)
            Lw=results.Lw / 1000,
            Lh=results.Lh / 1000,
            
            # These are dimensionless coefficients, no conversion needed
            CL=results.CL,
            # CL_max=results.CL_max,
            CD_wing=results.CD_wing,
            CD_fuse=results.CD_fuse,
            CD_total=results.CD_total,
            
            # Angles, no conversion needed
            AOA_stall=results.AOA_stall,
            AOA_takeoff_max=results.AOA_takeoff_max,
            AOA_climb_max=results.AOA_climb_max,
            AOA_turn_max=results.AOA_turn_max,
            
            # These are dimensionless coefficients, no conversion needed
            CL_flap_max=results.CL_flap_max,
            CL_flap_zero=results.CL_flap_zero,
            CD_flap_max=results.CD_flap_max,
            CD_flap_zero=results.CD_flap_zero,

            max_load=presetValues.max_load
        )

    def convert_propellerCSV_to_ndarray(self, csvPath):

        propeller_df = pd.read_csv(csvPath)
        propeller_df.dropna(how='any',inplace=True)
        propeller_df = propeller_df.sort_values(by=['RPM', 'V(speed) (m/s)']).reset_index(drop=True)

        rpm_array = propeller_df['RPM'].to_numpy()
        v_speed_array = propeller_df['V(speed) (m/s)'].to_numpy()
        torque_array = propeller_df['Torque (N-m)'].to_numpy()
        thrust_array = propeller_df['Thrust (kg)'].to_numpy()
        self.propeller_array = np.column_stack((rpm_array, v_speed_array, torque_array, thrust_array))
    
        return
    
    def convert_batteryCSV_to_ndarray(self, csvPath):

        df = pd.read_csv(csvPath,skiprows=[1]) 
        time_array = df['Time'].to_numpy()
        voltage_array = df['Voltage'].to_numpy()
        current_array = df['Current'].to_numpy()
        dt_array = np.diff(time_array, prepend=time_array[0])
        cumulative_Wh = np.cumsum(voltage_array*current_array*dt_array) * self.propulsionSpecs.n_cell / 3600
        SoC_array = 100 - (cumulative_Wh / self.propulsionSpecs.battery_Wh)*100
        mask = SoC_array >= 0
        time_array = time_array[mask]
        voltage_array = voltage_array[mask]
        current_array = current_array[mask]
        SoC_array = SoC_array[mask]
        battery_array = np.column_stack((time_array, voltage_array, current_array, SoC_array))
        self.battery_array = battery_array[battery_array[:, 3].argsort()]
        return
    
    def clearState(self):
        self.state = PlaneState()
        self.stateLog = []
    
    def setAuxVals(self) -> None:
        
        self.weight = self.missionParam.m_takeoff * g
        
        self.v_takeoff = (np.sqrt((2*self.weight) / (rho*self.analResult.Sref*self.analResult.CL_flap_max)))

        # Create focused alpha range from -10 to 10 degrees
        alpha_extended = np.linspace(-5, 15, 2000)  # 0.01 degree resolution
    
        CL_interp1d = interp1d(self.analResult.alpha_list, self.analResult.CL, kind="linear", fill_value="extrapolate")
        CD_interp1d = interp1d(self.analResult.alpha_list, self.analResult.CD_total, kind="quadratic", fill_value="extrapolate")
        # Create lookup tables
        CL_table = CL_interp1d(alpha_extended)
        CD_table = CD_interp1d(alpha_extended)
        
        self._cl_cache = {}
        self._cd_cache = {}

        # Create lambda functions for faster lookup
        self._cl_func_original = lambda alpha: np.interp(alpha, alpha_extended, CL_table)
        self._cd_func_original = lambda alpha: np.interp(alpha, alpha_extended, CD_table)
        self.alpha_func = lambda CL: np.interp(CL, CL_table, alpha_extended)
        return

    def CL_func(self,alpha):
        key = int(alpha*1000+0.5)  # Reduce precision for better cache hits
        if key not in self._cl_cache:
            self._cl_cache[key] = self._cl_func_original(alpha)

        #print(self._cl_cache[key] - self._cl_func_original(alpha))
        return self._cl_cache[key]
        #return np.interp(alpha, alpha_extended, CL_table)

    def CD_func(self,alpha):
        key = int(alpha*1000+0.5)  # Reduce precision for better cache hits
        if key not in self._cd_cache:
            self._cd_cache[key] = self._cd_func_original(alpha)

        #print(self._cl_cache[key] - self._cl_func_original(alpha))
        return self._cd_cache[key]
        #return np.interp(alpha, alpha_extended, CL_table)

    
    def run_mission(self, missionPlan: List[MissionConfig],clearState = True) -> int:

        flag = 0
        M3_time_limit = 300 - self.presetValues.x1_time_margin 
        if(clearState): self.clearState()

        for phase in missionPlan:
            try:
                match phase.phaseType:
                    case PhaseType.TAKEOFF:
                        flag = self.takeoff_simulation()
                        # print(f"takeoff = {flag}")
                    case PhaseType.CLIMB:
                        flag = self.climb_simulation(phase.numargs[0],phase.numargs[1],phase.direction) 
                        # print(f"climb = {flag}")  
                    case PhaseType.LEVEL_FLIGHT:
                        flag = self.level_flight_simulation(phase.numargs[0],phase.direction)
                        # print(f"level flight = {flag}")
                    case PhaseType.TURN:
                        flag = self.turn_simulation(phase.numargs[0],phase.direction)
                        # print(f"turn = {flag}")
                    case _: 
                        raise ValueError("Didn't provide a correct PhaseType!")
                if (self.state.time > M3_time_limit or self.state.battery_voltage < self.presetValues.min_battery_voltage):
                    return -2
                self.state.phase += 1
                
                if flag==-1: 
                    return -1
                
            except Exception as e:
                print(e)
                return -1        
    
        return 0

    def run_mission2(self) -> float:

        result = 0
        
        mission2 = [
                MissionConfig(PhaseType.TAKEOFF, []),
                MissionConfig(PhaseType.CLIMB, [30,-140], "left"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [-152], "left"),
                MissionConfig(PhaseType.TURN, [180], "CW"),
                MissionConfig(PhaseType.CLIMB, [30,-10], "right"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [0], "right"),
                MissionConfig(PhaseType.TURN, [360], "CCW"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [152], "right"),
                MissionConfig(PhaseType.TURN, [180], "CW"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [-152], "left"),
                MissionConfig(PhaseType.TURN, [180], "CW"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [0], "right"),
                MissionConfig(PhaseType.TURN, [360], "CCW"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [152], "right"),
                MissionConfig(PhaseType.TURN, [180], "CW"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [-152], "left"),
                MissionConfig(PhaseType.TURN, [180], "CW"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [0], "right"),
                MissionConfig(PhaseType.TURN, [360], "CCW"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [152], "right"),
                MissionConfig(PhaseType.TURN, [180], "CW"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [0], "left"),
                ]
        result = self.run_mission(mission2)  
        
        first_state = self.stateLog[0]
        first_state.mission = 2 
        last_state = self.stateLog[-1]
        last_state.N_laps = 3   
        last_z_pos = last_state.position[2] 
        last_battery_voltage = last_state.battery_voltage 
        if(result == -1 or last_z_pos < 20 or last_battery_voltage < self.presetValues.min_battery_voltage): return -1,-1
        
        return self.m_fuel, self.state.phase

    def run_mission3(self) -> float:
        result = 0
        mission3 = [
                MissionConfig(PhaseType.TAKEOFF, []),
                MissionConfig(PhaseType.CLIMB, [60,-140], "left"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [-152], "left"),
                MissionConfig(PhaseType.TURN, [180], "CW"),
                MissionConfig(PhaseType.CLIMB, [60,-10], "right"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [0], "right"),
                MissionConfig(PhaseType.TURN, [360], "CCW"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [152], "right"),
                MissionConfig(PhaseType.TURN, [180], "CW"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [0], "left"),
                ]

        # Run initial mission sequence
        result = self.run_mission(mission3)
        first_state = self.stateLog[0]
        first_state.mission = 3
        if(result == -1): 
            return -1

        # Store starting index for each lap to handle truncation if needed
        self.state.N_laps = 1
        time_limit = 300 - self.presetValues.x1_time_margin  

        # Define lap2 phases
        lap2 = [
            MissionConfig(PhaseType.LEVEL_FLIGHT, [-152], "left"),
            MissionConfig(PhaseType.TURN, [180], "CW"),
            MissionConfig(PhaseType.LEVEL_FLIGHT, [0], "right"),
            MissionConfig(PhaseType.TURN, [360], "CCW"),
            MissionConfig(PhaseType.LEVEL_FLIGHT, [152], "right"),
            MissionConfig(PhaseType.TURN, [180], "CW"),
            MissionConfig(PhaseType.LEVEL_FLIGHT, [0], "left"),
        ]

        while True:
            lap_start_index = len(self.stateLog)
            self.state.N_laps += 1
            
            result = self.run_mission(lap2,clearState=False)
            if(result == -1): return -1
            if(result == -2):
                self.state.N_laps -= 1
                return self.state.N_laps, self.state.phase, self.state.time
            
            # Check if we've exceeded time limit or voltage limit
            if (self.state.time > time_limit or 
                self.state.battery_voltage < self.presetValues.min_battery_voltage):
                
                # Truncate the results and finish
                self.stateLog = self.stateLog[:lap_start_index]
                self.state.N_laps -= 1
                break
        
        return self.state.N_laps, self.state.phase, self.state.time
        
    def calculate_level_alpha(self, v):
        #  Function that calculates the AOA required for level flight using the velocity vector and thrust
        return self.calculate_level_alpha_fast(v)
        speed = fast_norm(v)
        def equation(alpha:float):

            CL = float(self.CL_func(alpha)[0])
            L,_ = self.calculate_Lift_and_Loadfactor(CL,float(speed))
            return float(L-self.weight)

        alpha_solution = fsolve(equation, 5, xtol=1e-4, maxfev=1000)

        fast_sol = self.calculate_level_alpha_fast(v)
        print(fast_sol - alpha_solution[0])
        return alpha_solution[0]
    
    # Use the fact that CL is quadratic(increasing) and do binary search instead
    def calculate_level_alpha_fast(self,v):
        speed = fast_norm(v)
        # Pre-calculate shared values
        dynamic_pressure = 0.5 * PhysicalConstants.rho * speed**2 * self.analResult.Sref
        weight = self.missionParam.m_takeoff * PhysicalConstants.g
        
        # Binary search instead of fsolve
        alpha_min, alpha_max = -3, 13
        tolerance = 1e-4
        
        while (alpha_max - alpha_min) > tolerance:
            alpha = (alpha_min + alpha_max) / 2
            CL = float(self.CL_func(alpha))
            L = dynamic_pressure * CL
            
            if L > weight:
                alpha_max = alpha
            else:
                alpha_min = alpha
                
        return (alpha_min + alpha_max) / 2

    def calculate_Lift_and_Loadfactor(self, CL, speed:float=-1):
        if(speed == -1): speed = fast_norm(self.state.velocity)
        L = 0.5 * rho * speed**2 * self.analResult.Sref * CL
        return L, L/self.weight 
    
    def isBelowFlapTransition(self):
        return self.state.position[2] < self.presetValues.h_flap_transition  
    
    def updateBatteryState(self,SoC):
        capacity = self.propulsionSpecs.battery_Wh * SoC / 100
        capacity -= self.state.motor_input_power * self.dt / 3600 
        self.state.battery_SoC = capacity/self.propulsionSpecs.battery_Wh * 100  
        voltage_per_cell = SoC2Vol(self.state.battery_SoC,self.battery_array)
        self.state.battery_voltage = self.propulsionSpecs.n_cell * voltage_per_cell
        return
        
   
    def takeoff_simulation(self):
    
        self.dt= 0.1
        step=0
        max_steps = int(15 / self.dt) # 15 sec simulation
        self.state.velocity = np.array([0.0, 0.0, 0.0])
        self.state.position = np.array([0.0, 0.0, 0.0])
        self.state.time = 0.0
        self.state.battery_voltage = 4.2 * self.propulsionSpecs.n_cell 
        self.state.battery_SoC = 100.0
   
        for step in range(max_steps): 
            # Ground roll until 0.9 times takeoff speed
            if fast_norm(self.state.velocity) < 0.9 * self.v_takeoff :
                
                self.state.time += self.dt
                
                self.state.throttle = self.presetValues.throttle_takeoff
                _, _, self.state.Amps, self.state.motor_input_power, thrust_per_motor = thrust_analysis(
                                self.state.throttle,
                                fast_norm(self.state.velocity),
                                self.state.battery_voltage,
                                self.propulsionSpecs,
                                self.propeller_array,
                                0 #graphFlag
                )
                self.state.thrust = self.presetValues.number_of_motor * thrust_per_motor 
                T_takeoff = self.state.thrust * g
                
                
                self.state.acceleration = calculate_acceleration_groundroll(
                        self.state.velocity,
                        self.missionParam.m_takeoff,
                        self.weight,
                        self.analResult.Sref,
                        self.analResult.CD_flap_zero, self.analResult.CL_flap_zero,
                        T_takeoff
                        )

                self.state.velocity -= self.state.acceleration * self.dt
                self.state.position += self.state.velocity * self.dt
                
                _, loadfactor = self.calculate_Lift_and_Loadfactor(self.analResult.CL_flap_zero)
                self.state.loadfactor = loadfactor

                self.state.AOA = 0
                self.state.climb_pitch_angle =np.nan
                self.state.bank_angle = np.nan

                self.updateBatteryState(self.state.battery_SoC)
                self.logState()
            
            # Ground rotation until takeoff speed    
            elif 0.9 * self.v_takeoff <= fast_norm(self.state.velocity) <= self.v_takeoff:
                self.state.time += self.dt

                self.state.throttle = self.presetValues.throttle_takeoff
                _, _, self.state.Amps, self.state.motor_input_power, thrust_per_motor = thrust_analysis(
                                self.presetValues.throttle_takeoff,
                                fast_norm(self.state.velocity),
                                self.state.battery_voltage,
                                self.propulsionSpecs,
                                self.propeller_array,
                                0 #graphFlag
                )
                self.state.thrust = self.presetValues.number_of_motor * thrust_per_motor 
                T_takeoff = self.state.thrust * g
                
                self.state.acceleration = calculate_acceleration_groundrotation(
                        self.state.velocity,
                        self.missionParam.m_takeoff,
                        self.weight,
                        self.analResult.Sref,
                        self.analResult.CD_flap_max, self.analResult.CL_flap_max,
                        T_takeoff
                        )
                self.state.velocity -= self.state.acceleration * self.dt
                self.state.position += self.state.velocity * self.dt
                
            
                
                _, loadfactor = self.calculate_Lift_and_Loadfactor(self.analResult.CL_flap_max)
                self.state.loadfactor = loadfactor

                self.state.AOA=10
                self.state.climb_pitch_angle=np.nan
                self.state.bank_angle = np.nan

                self.updateBatteryState(self.state.battery_SoC)
                self.logState()
            else:
                break
            
            if(step == max_steps-1) : return -1  
            

    def climb_simulation(self, h_target, x_max_distance, direction):
        """
        Args:
            h_target (float): Desired altitude to climb at the maximum climb AOA (m)
            x_max_distance (float): Restricted x-coordinate for climb (m)
            direction (string): The direction of movement. Must be either 'left' or 'right'.
        """
     
        
        if self.state.position[2] > h_target: return
        self.dt = 0.1
        step=0
        max_steps = int(60 / self.dt)  # Max 60 seconds simulation
        break_flag = False
        alpha_w_deg = 0 
        
        for step in range(max_steps):
            self.state.time += self.dt

            # Calculate climb angle
            gamma_rad = np.atan2(self.state.velocity[2], abs(self.state.velocity[0]))

            if direction == 'right':
                # set AOA at climb (if altitude is below target altitude, set AOA to AOA_climb. if altitude exceed target altitude, decrease AOA gradually to -5 degree)
                if(self.state.position[2] < self.presetValues.h_flap_transition and 
                   self.state.position[0] < x_max_distance):
                    alpha_w_deg = self.analResult.AOA_takeoff_max
                elif(self.presetValues.h_flap_transition <= self.state.position[2] < h_target and 
                     self.state.position[0] < x_max_distance):
                    load_factor = self.calculate_Lift_and_Loadfactor(float(self.CL_func(self.analResult.AOA_climb_max)))[1]
            
                    if (load_factor < self.missionParam.max_load_factor and 
                        gamma_rad < np.radians(self.presetValues.max_climb_angle)):
                        alpha_w_deg = self.analResult.AOA_climb_max
                    elif (load_factor >= self.missionParam.max_load_factor and 
                          gamma_rad < np.radians(self.presetValues.max_climb_angle)):
                        alpha_w_deg = float(self.alpha_func((2 * self.weight * self.missionParam.max_load_factor)/
                                                          (rho * fast_norm(self.state.velocity)**2 * self.analResult.Sref)))
                    else:
                        alpha_w_deg -= 1
                        alpha_w_deg = max(alpha_w_deg, -5)
                else:
                    break_flag = True
                    if gamma_rad > np.radians(self.presetValues.max_climb_angle):
                        alpha_w_deg -= 1
                        alpha_w_deg = max(alpha_w_deg, -5)
                    else:
                        alpha_w_deg -= 1
                        alpha_w_deg = max(alpha_w_deg, -5)
            
            elif direction == 'left':
                # set AOA at climb (if altitude is below target altitude, set AOA to AOA_climb. if altitude exceed target altitude, decrease AOA gradually to -5 degree)
                if(self.state.position[2] < self.presetValues.h_flap_transition and 
                   self.state.position[0] > x_max_distance):
                    alpha_w_deg = self.analResult.AOA_takeoff_max
                elif(self.presetValues.h_flap_transition <= self.state.position[2] < h_target and 
                     self.state.position[0] > x_max_distance):
                    load_factor = self.calculate_Lift_and_Loadfactor(float(self.CL_func(self.analResult.AOA_climb_max)))[1]
            
                    if (load_factor < self.missionParam.max_load_factor and 
                        gamma_rad < np.radians(self.presetValues.max_climb_angle)):
                        alpha_w_deg = self.analResult.AOA_climb_max
                    elif (load_factor >= self.missionParam.max_load_factor and 
                          gamma_rad < np.radians(self.presetValues.max_climb_angle)):
                        alpha_w_deg = float(self.alpha_func((2 * self.weight * self.missionParam.max_load_factor)/
                                                          (rho * fast_norm(self.state.velocity)**2 * self.analResult.Sref)))
                    else:
                        alpha_w_deg -= 3
                        alpha_w_deg = max(alpha_w_deg, -5)
                else:
                    break_flag = True
                    if gamma_rad > np.radians(self.presetValues.max_climb_angle):
                        alpha_w_deg -= 2
                        alpha_w_deg = max(alpha_w_deg, -5)
                    else:
                        alpha_w_deg -= 2
                        alpha_w_deg = max(alpha_w_deg, -5)            
                    
                    
            if (self.isBelowFlapTransition()):
                CL = self.analResult.CL_flap_max
            else:
                CL = float(self.CL_func(alpha_w_deg))
                
            speed = fast_norm(self.state.velocity)  

            T_climb_max_per_motor = determine_max_thrust(speed,
                                            self.state.battery_voltage,
                                            self.propulsionSpecs,
                                            self.propeller_array,
                                            0#graphFlag
            ) #kg
            thrust_per_motor = T_climb_max_per_motor * self.missionParam.climb_thrust_ratio #kg    

            if speed >= self.missionParam.max_speed * 0.95:

                D = 0.5 * rho * speed**2 * self.analResult.Sref * self.CD_func(alpha_w_deg)
                T_desired = (D + self.weight * np.sin(gamma_rad)) / np.cos(np.deg2rad(alpha_w_deg))
                thrust_per_motor_desired = T_desired / (2*g)
                thrust_per_motor = min(thrust_per_motor, thrust_per_motor_desired)


            _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(thrust_per_motor, speed,self.state.battery_voltage, self.propulsionSpecs.Kv, self.propulsionSpecs.R, self.propeller_array)
            
            T_climb = thrust_per_motor * self.presetValues.number_of_motor * g # total N
            
            self.state.acceleration = RK4_step(self.state.velocity,self.dt,
                         lambda v: calculate_acceleration_climb(v, 
                                                                self.missionParam.m_takeoff,
                                                                self.weight,
                                                                self.analResult.Sref,
                                                                self.CL_func,
                                                                self.CD_func,
                                                                self.analResult.CL_flap_max,
                                                                self.analResult.CD_flap_max,
                                                                alpha_w_deg,
                                                                gamma_rad,
                                                                T_climb,
                                                                not self.isBelowFlapTransition()
                                                                ))
            self.state.velocity[2] += self.state.acceleration[2]*self.dt
            if direction == 'right':
                self.state.velocity[0] += self.state.acceleration[0]*self.dt
            else:
                self.state.velocity[0] -= self.state.acceleration[0]*self.dt
            
            self.state.position[0] += self.state.velocity[0]* self.dt
            self.state.position[2] += self.state.velocity[2]* self.dt

            _, loadfactor = self.calculate_Lift_and_Loadfactor(CL)
            
            self.state.loadfactor = loadfactor

            self.state.throttle = self.missionParam.climb_thrust_ratio
             
            self.state.AOA = alpha_w_deg
            self.state.climb_pitch_angle =alpha_w_deg + np.degrees(gamma_rad)
            self.state.bank_angle = np.nan


            self.updateBatteryState(self.state.battery_SoC)
            self.logState()

            # break when climb angle goes to zero
            if break_flag == 1 and gamma_rad < 0:
                # print(f"cruise altitude is {z_pos:.2f} m.")
                break
            
            if step==max_steps-1 : return -1 

    def level_flight_simulation(self, x_final, direction):
     
        #print("\nRunning Level Flight Simulation...")
        # print(max_steps)
        step = 0
        self.dt = 0.1
        max_steps = int(180/self.dt) # max 3 minuites
        # Initialize vectors
        self.state.velocity[2] = 0  # Zero vertical velocity
        speed = fast_norm(self.state.velocity)

        if direction == 'right':
            self.state.velocity = np.array([speed, 0, 0])  # Align with x-axis
        elif direction=='left':
            self.state.velocity = np.array([-speed, 0, 0])
        
        cruise_flag = 0
        
        for step in range(max_steps):
            
            self.state.time += self.dt
            speed = fast_norm(self.state.velocity)
            
            # Calculate alpha_w first
            alpha_w_deg = self.calculate_level_alpha(self.state.velocity)
                
            # Speed limiting while maintaining direction
            if speed >= self.missionParam.max_speed - 0.005:  # Original speed limit
                cruise_flag = 1

            if cruise_flag == 1:
                self.state.velocity = self.state.velocity * (self.missionParam.max_speed / speed)
                T_cruise = 0.5 * rho * self.missionParam.max_speed**2 \
                                * self.analResult.Sref * float(self.CD_func(alpha_w_deg))
                T_cruise_max = determine_max_thrust(speed,
                                               self.state.battery_voltage,
                                               self.propulsionSpecs,
                                               self.propeller_array,
                                               0#graphFlag
                ) #kg
                T_cruise_max = T_cruise_max * self.presetValues.number_of_motor * g
                T_cruise = min(T_cruise, T_cruise_max )
                self.state.thrust = T_cruise / g #kg
            
                alpha_w_deg = self.calculate_level_alpha(self.state.velocity)
                _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(
                    self.state.thrust/self.presetValues.number_of_motor,
                    speed,self.state.battery_voltage,
                    self.propulsionSpecs.Kv,
                    self.propulsionSpecs.R,
                    self.propeller_array)

                self.updateBatteryState(self.state.battery_SoC)
    
                self.state.acceleration = RK4_step(self.state.velocity,self.dt,
                             lambda v: calculate_acceleration_level(v,self.missionParam.m_takeoff, 
                                                                    self.analResult.Sref,
                                                                    self.CD_func, alpha_w_deg,
                                                                    T_cruise))
                if abs(self.state.acceleration[0]) > 0.1 : cruise_flag = 0
            else:
                
                T_level_max_per_motor = determine_max_thrust(
                                                speed,
                                                self.state.battery_voltage,
                                                self.propulsionSpecs,
                                                self.propeller_array,
                                                0#graphFlag
                ) #kg
                thrust_per_motor = T_level_max_per_motor * self.missionParam.level_thrust_ratio #kg
                self.state.thrust = self.presetValues.number_of_motor * thrust_per_motor #kg
                _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(thrust_per_motor, speed,self.state.battery_voltage, self.propulsionSpecs.Kv, self.propulsionSpecs.R, self.propeller_array)
                
                T_climb = self.state.thrust * g # total N
                self.updateBatteryState(self.state.battery_SoC)

                self.state.acceleration =  RK4_step(self.state.velocity,self.dt,
                             lambda v: calculate_acceleration_level(v,self.missionParam.m_takeoff, 
                                                                    self.analResult.Sref,
                                                                    self.CD_func, alpha_w_deg,
                                                                    T_climb))

                
            # Update Acc, Vel, position
            if direction == 'right': 
                self.state.velocity += self.state.acceleration * self.dt
            elif direction == 'left': 
                self.state.velocity -= self.state.acceleration * self.dt
            
            self.state.position[0] += self.state.velocity[0] * self.dt
            self.state.position[1] += self.state.velocity[1] * self.dt
            
            # Calculate and store results

            _,load_factor = self.calculate_Lift_and_Loadfactor(float(self.CL_func(alpha_w_deg)))
            
            self.state.loadfactor = load_factor 
            self.state.AOA = alpha_w_deg
            self.state.bank_angle = np.nan
            self.state.climb_pitch_angle = np.nan
            self.logState()
            
            # Check if we've reached target x position
            if direction == 'right':
                if self.state.position[0] >= x_final:
                    break
            elif direction == 'left':
                if self.state.position[0] <= x_final:
                    break
            
            if step==max_steps-1 : return -1        


    def turn_simulation(self, target_angle_deg, direction):
        """
        Args:
            target_angle_degree (float): Required angle of coordinate level turn (degree)
            direction (string): The direction of movement. Must be either 'CW' or 'CCW'.
        """     
        
        speed = fast_norm(self.state.velocity) 
        self.dt = 0.1  
        step = 0
        max_steps = int(180/self.dt) 
        # Initialize turn tracking
        target_angle_rad = np.radians(target_angle_deg)
        turned_angle_rad = 0

        # Get initial heading and setup turn center
        initial_angle_rad = np.atan2(self.state.velocity[1], self.state.velocity[0])
        current_angle_rad = initial_angle_rad

        # Pre-calculate constants
        dynamic_pressure_base = 0.5 * rho * self.analResult.Sref
        max_speed = self.missionParam.max_speed
        max_load = self.missionParam.max_load_factor
        weight = self.weight

        for step in range(max_steps):
            # print(step)
            if abs(turned_angle_rad) < abs(target_angle_rad):
                self.state.time += self.dt

                if speed < max_speed - 0.005: # numerical error
                        # Pre-calculate shared terms
                        dynamic_pressure = dynamic_pressure_base * speed * speed
                        
                        CL = min(float(self.CL_func(self.analResult.AOA_turn_max)), 
                                float((max_load * weight)/(dynamic_pressure)))

                        alpha_turn = float(self.alpha_func(CL))
                        L = dynamic_pressure * CL
                        if weight / L >=1: 
                            # print("too heavy")
                            return -1
                        phi_rad = np.acos(min(weight/L,0.99))
                        
                        a_centripetal = (L * np.sin(phi_rad)) / self.missionParam.m_takeoff
                        R = (self.missionParam.m_takeoff * speed**2)/(L * np.sin(phi_rad))
                        omega = speed / R

                        self.state.loadfactor = 1 / np.cos(phi_rad)

                        CD = float(self.CD_func(alpha_turn))
                        D = CD * dynamic_pressure
                    
                        T_turn_max_per_motor = determine_max_thrust(
                                                        speed,
                                                        self.state.battery_voltage,
                                                        self.propulsionSpecs,
                                                        self.propeller_array,
                                                        0#graphFlag
                        ) #kg
                        thrust_per_motor = T_turn_max_per_motor * self.missionParam.turn_thrust_ratio #kg
                        self.state.thrust = self.presetValues.number_of_motor * thrust_per_motor #kg
                        _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(thrust_per_motor, speed,self.state.battery_voltage, self.propulsionSpecs.Kv, self.propulsionSpecs.R, self.propeller_array)
                        
                        T_turn = self.state.thrust * g # total N              
                        a_tangential = (T_turn - D) / self.missionParam.m_takeoff
                        
                        speed += a_tangential * self.dt
                        self.updateBatteryState(self.state.battery_SoC)

                else:
                        speed = max_speed
                        dynamic_pressure = dynamic_pressure_base * speed * speed
                    
                        CL = min(float(self.CL_func(self.analResult.AOA_turn_max)), 
                            float((max_load * weight)/(dynamic_pressure)))
                            
                        alpha_turn = float(self.alpha_func(CL))
                        L = dynamic_pressure * CL
                        if weight / L >=1: 
                            #print("too heavy")
                            return -1
                        phi_rad = np.acos(min(weight/L,0.99))

                        a_centripetal = (L * np.sin(phi_rad)) / self.missionParam.m_takeoff
                        R = (self.missionParam.m_takeoff * speed**2)/(L * np.sin(phi_rad))
                        omega = speed / R

                        self.state.loadfactor = 1 / np.cos(phi_rad)

                        CD = float(self.CD_func(alpha_turn))
                        D = CD * dynamic_pressure
                    
                        T_turn_max_per_motor = determine_max_thrust(
                                                        speed,
                                                        self.state.battery_voltage,
                                                        self.propulsionSpecs,
                                                        self.propeller_array,
                                                        0#graphFlag
                        ) #kg
                        
                        T = min(D, T_turn_max_per_motor*self.presetValues.number_of_motor*self.missionParam.turn_thrust_ratio*g)
                        self.state.thrust = T/g
                        thrust_per_motor = self.state.thrust / self.presetValues.number_of_motor
                        _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(thrust_per_motor, speed,self.state.battery_voltage, self.propulsionSpecs.Kv, self.propulsionSpecs.R, self.propeller_array)
                        
                        a_tangential = (T - D) / self.missionParam.m_takeoff
                        speed += a_tangential * self.dt

                        self.updateBatteryState(self.state.battery_SoC)

                # Calculate turn center
                sin_current = np.sin(current_angle_rad)
                cos_current = np.cos(current_angle_rad)
                
                if direction == "CCW":
                    center_x = self.state.position[0] - R * sin_current
                    center_y = self.state.position[1] + R * cos_current
                    current_angle_rad += omega * self.dt
                    turned_angle_rad += omega * self.dt
                else:  # CW
                    center_x = self.state.position[0] + R * sin_current
                    center_y = self.state.position[1] - R * cos_current
                    current_angle_rad -= omega * self.dt
                    turned_angle_rad -= omega * self.dt

                # Update position
                sin_new = np.sin(current_angle_rad)
                cos_new = np.cos(current_angle_rad)
                
                if direction == "CCW":
                    self.state.position[0] = center_x + R * sin_new
                    self.state.position[1] = center_y - R * cos_new
                else:  # CW
                    self.state.position[0] = center_x - R * sin_new
                    self.state.position[1] = center_y + R * cos_new

                # Update velocity direction
                self.state.velocity = np.array([
                    speed * cos_new,
                    speed * sin_new,
                    0
                ])

                self.state.acceleration = np.array([
                    a_tangential * cos_new - a_centripetal * sin_new,
                    a_tangential * sin_new + a_centripetal * cos_new,
                    0
                ])

                self.state.AOA = alpha_turn
                self.state.bank_angle = np.degrees(phi_rad)
                self.state.climb_pitch_angle = np.nan

                self.logState() 
            else:
                break
            if step==max_steps-1 :
                # print("declined")
                return -1
        
    
    def logState(self) -> None:
        # Append current state as a copy
        self.stateLog.append(PlaneState(
            mission=self.state.mission,
            N_laps=self.state.N_laps,
            position=self.state.position.copy(),
            velocity=self.state.velocity.copy(), 
            acceleration=self.state.acceleration.copy(),
            time=self.state.time,
            throttle=self.state.throttle,
            thrust=self.state.thrust,
            loadfactor=self.state.loadfactor,
            AOA=self.state.AOA,
            climb_pitch_angle=self.state.climb_pitch_angle,
            bank_angle=self.state.bank_angle,
            phase=self.state.phase,
            battery_SoC=self.state.battery_SoC,
            battery_voltage=self.state.battery_voltage,
            Amps=self.state.Amps
        ))
    
## end of class    
#########################################################


def RK4_step(v, dt, func):
    """ Given v and a = f(v), solve for (v(t+dt)-v(dt))/dt or approximately a(t+dt/2)"""

    dt2 = dt/2
    a1 = func(v)
    a2 = func(v + a1 * dt2)
    a3 = func(v + a2 * dt2)
    a4 = func(v + a3 * dt)
    return (a1 + 2*(a2 + a3) + a4) * (1/6)

def fast_norm(v):
    """Faster alternative to np.linalg.norm for 3D vectors"""
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def calculate_acceleration_groundroll(v, m, Weight,
                                      Sref,
                                      CD_zero_flap,CL_zero_flap,
                                      T_takeoff)->np.ndarray:
    # Function that calculates the acceleration of an aircraft during ground roll
    speed = fast_norm(v)
    D = 0.5 * rho * speed**2 * Sref * CD_zero_flap
    L = 0.5 * rho * speed**2 * Sref * CL_zero_flap
    a_x = (T_takeoff - D - 0.03*(Weight-L)) / m              # calculate x direction acceleration 
    return np.array([a_x, 0, 0])

def calculate_acceleration_groundrotation(v, m, Weight,
                                          Sref,
                                          CD_max_flap,CL_max_flap,
                                          T_takeoff)->np.ndarray:
    # Function that calculate the acceleration of the aircraft during rotation for takeoff
    speed = fast_norm(v)
    D = 0.5 * rho * speed**2 * Sref * CD_max_flap
    L = 0.5 * rho * speed**2 * Sref * CL_max_flap
    a_x = (T_takeoff - D - 0.03*(Weight-L)) / m            # calculate x direction acceleration 
    return np.array([a_x, 0, 0])

def calculate_acceleration_level(v, m, Sref, CD_func, alpha_deg, T):
    # Function that calculates the acceleration during level flight
    speed = fast_norm(v)
    CD = float(CD_func(alpha_deg))
    D = 0.5 * rho * speed**2 * Sref * CD
    a_x = (T * np.cos(np.radians(alpha_deg)) - D) / m
    return np.array([a_x, 0, 0])

def calculate_acceleration_climb(v, m, Weight, 
                                 Sref, 
                                 CL_func, CD_func, 
                                 CL_max_flap, CD_max_flap, 
                                 alpha_deg, gamma_rad, 
                                 T_climb, 
                                 over_flap_transition)->np.ndarray:
    # gamma rad : climb angle
    # over_flap_transition: checks if plane is over the flap transition (boolean)
    # Function that calculates the acceleration during climb
    CL=0
    CD=0
    speed = fast_norm(v)
    if (over_flap_transition):
        CL = float(CL_func(alpha_deg))
        CD = float(CD_func(alpha_deg))
    else:
        CL = CL_max_flap
        CD = CD_max_flap
    theta_deg = np.degrees(gamma_rad) + alpha_deg
    theta_rad = np.radians(theta_deg)
    
    D = 0.5 * rho * speed**2 * Sref * CD
    L = 0.5 * rho * speed**2 * Sref * CL

    a_x = (T_climb * np.cos(theta_rad) - L * np.sin(gamma_rad) - D * np.cos(gamma_rad) )/ m
    a_z = (T_climb * np.sin(theta_rad) + L * np.cos(gamma_rad) - D * np.sin(gamma_rad) - Weight )/ m
    
    return np.array([a_x,0,a_z])

def get_state_df(stateLog):
    # Convert numpy arrays to lists for proper DataFrame conversion
    states_dict = []
    for state in stateLog:
        state_dict = {
            'mission' : state.mission,
            "N_laps" : state.N_laps,
            'position': np.array([state.position[0],state.position[1],state.position[2]]),
            'velocity': np.array([state.velocity[0],state.velocity[1],state.velocity[2]]),
            'acceleration': np.array([state.acceleration[0],state.acceleration[1],state.acceleration[2]]),
            'time': state.time,
            'throttle': state.throttle,
            'thrust' : state.thrust,
            'loadfactor': state.loadfactor,
            'AOA': state.AOA,
            'climb_pitch_angle': state.climb_pitch_angle,
            'bank_angle': state.bank_angle,
            'phase': state.phase,
            'battery_SoC': state.battery_SoC,
            'battery_voltage': state.battery_voltage,
            'Amps' : state.Amps,
            'motor_input_power': state.motor_input_power
        }
        states_dict.append(state_dict)
    return pd.DataFrame(states_dict)

def visualize_mission(stateLog):
    """Generate all visualization plots for the mission in a single window"""
    stateLog = get_state_df(stateLog)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(4, 4)
    
    # Get phases and colors
    phases = stateLog['phase'].unique()
    # colors = plt.cm.rainbow(np.random.rand(len(phases)))
    color_list = ['red', 'green', 'blue', 'orange', 'black']
    colors = [color_list[i % len(color_list)] for i in range(len(phases))]
    
    # Graph1 : 3D trajectory colored by phase
    ax_3d = fig.add_subplot(gs[0:2,0], projection='3d')
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_3d.plot(stateLog[mask]['position'].apply(lambda x: x[0]), 
                  stateLog[mask]['position'].apply(lambda x: x[1]), 
                  stateLog[mask]['position'].apply(lambda x: x[2]),
                  color=color, label=f'Phase {phase}')
        
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D Trajectory')

    x_lims = ax_3d.get_xlim3d()
    y_lims = ax_3d.get_ylim3d()
    z_lims = ax_3d.get_zlim3d()
    max_range = max(x_lims[1] - x_lims[0], y_lims[1] - y_lims[0])
    x_center = (x_lims[1] + x_lims[0]) / 2
    y_center = (y_lims[1] + y_lims[0]) / 2
    z_center = (z_lims[1] + z_lims[0]) / 2
    ax_3d.set_xlim3d([x_center - max_range/2, x_center + max_range/2])
    ax_3d.set_ylim3d([y_center - max_range/2, y_center + max_range/2])
    ax_3d.set_zlim3d([0, z_lims[1]*1.2])

    # Graph2 : Side view colored by phase
    ax_side = fig.add_subplot(gs[2, 0])
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_side.plot(stateLog[mask]['position'].apply(lambda x: x[0]), 
                    stateLog[mask]['position'].apply(lambda x: x[2]),
                    color=color, label=f'Phase {phase}')
    ax_side.set_xlabel('X Position (m)')
    ax_side.set_ylabel('Altitude (m)')
    ax_side.set_title('Side View')
    ax_side.grid(True)
    ax_side.set_aspect(1)


    # Graph3 : Top-down view colored by phase
    ax_top = fig.add_subplot(gs[0,1])
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_top.plot(stateLog[mask]['position'].apply(lambda x: x[0]), 
                   stateLog[mask]['position'].apply(lambda x: x[1]),
                   color=color, label=f'Phase {phase}')
    ax_top.set_xlabel('X Position (m)')
    ax_top.set_ylabel('Y Position (m)')
    ax_top.set_title('Top-Down View')
    ax_top.grid(True)
    ax_top.set_aspect('equal')
    # ax_top.legend()

    # Graph4 : AOA
    ax_aoa = fig.add_subplot(gs[1, 1])
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_aoa.plot(stateLog[mask]['time'], 
                    stateLog[mask]['AOA'],
                    color=color)
    ax_aoa.set_xlabel('Time (s)')
    ax_aoa.set_ylabel('AOA (degrees)')
    ax_aoa.set_xlim(0,None)
    ax_aoa.set_ylim(-3,14.3)
    ax_aoa.set_yticks(np.arange(0, 14.1, 2))
    ax_aoa.set_yticks(np.arange(0, 14.1, 1),minor=True)
    ax_aoa.set_title('Angle of Attack')
    ax_aoa.grid(True, which='major', linestyle='-', linewidth=1) 
    ax_aoa.grid(True, which='minor', linestyle=':', linewidth=0.5)
    

    # Graph5 : Bank, Pitch angle
    ax_angles = fig.add_subplot(gs[2, 1])

    ax_angles.plot(stateLog['time'], stateLog['bank_angle'], label='Bank Angle', color='blue')
    ax_angles.set_xlabel('Time (s)')
    ax_angles.set_ylabel('Angle (degrees)') 
    ax_angles.plot(stateLog['time'], stateLog['climb_pitch_angle'], label='Climb Pitch Angle', color='red')
    ax_angles.set_yticks(np.arange(0, stateLog['bank_angle'].max()+6, 5),minor=True)
    ax_angles.grid(True, which='major', linestyle='-', linewidth=1) 
    ax_angles.grid(True, which='minor', linestyle=':', linewidth=0.5)
    ax_angles.set_title('Bank, Pitch Angles')
  


    # Graph6 : speed
    ax_speed = fig.add_subplot(gs[0, 2])
    speeds = np.sqrt(stateLog['velocity'].apply(lambda x: x[0]**2 + x[1]**2 + x[2]**2))
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_speed.plot(stateLog[mask]['time'], speeds[mask], 
                     color=color, label=f'Phase {phase}')
    ax_speed.set_xlabel('Time (s)')
    ax_speed.set_ylabel('Speed (m/s)')
    ax_speed.set_title('Speed by Phase')
    ax_speed.set_yticks(np.arange(0, max(speeds)+6, 5),minor=True)
    ax_speed.grid(True, which='major', linestyle='-', linewidth=1) 
    ax_speed.grid(True, which='minor', linestyle=':', linewidth=0.5)
    
    # Graph7 : throttle
    ax_throttle = fig.add_subplot(gs[1, 2])
    ax_throttle.plot(stateLog['time'], stateLog['throttle']*100,'r-')
    ax_throttle.set_ylim(40,100)
    ax_throttle.set_title('Throttle level')
    ax_throttle.set_xlabel('Time (s)')
    ax_throttle.set_ylabel('Throttle (%)')
    ax_throttle.tick_params(axis='y')
    ax_throttle.set_yticks(np.arange(40, 101, 5),minor=True)
    ax_throttle.grid(True, which='major', linestyle='-', linewidth=1) 
    ax_throttle.grid(True, which='minor', linestyle=':', linewidth=0.5)
    

    # Graph8 : thrust
    ax_thrust = fig.add_subplot(gs[2, 2])
    ax_thrust.plot(stateLog['time'], stateLog['thrust'],'r-', label='Thrust')
    ax_thrust.set_ylim(0,max(stateLog['thrust'])+1)
    ax_thrust.tick_params(axis='y')
    ax_thrust.set_yticks(np.arange(0, max(stateLog['thrust'])+0.5, 1),minor=True)
    ax_thrust.set_title('Thrust')
    ax_thrust.set_xlabel('Time (s)')
    ax_thrust.set_ylabel('Thrust (kg)')
    ax_thrust.grid(True, which='major', linestyle='-', linewidth=1) 
    ax_thrust.grid(True, which='minor', linestyle=':', linewidth=0.5)

    # Graph9 :Load factor
    ax_load = fig.add_subplot(gs[0, 3])
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_load.plot(stateLog[mask]['time'], 
                    stateLog[mask]['loadfactor'], 
                    color=color, label=f'Phase {phase}')
    ax_load.tick_params(axis='y')
    ax_load.set_yticks(np.arange(0, max(stateLog['loadfactor'])+0.5, 1),minor=True)
    ax_load.set_xlabel('Time (s)')
    ax_load.set_ylabel('Load Factor')
    ax_load.set_title('Load Factor by Phase')
    ax_load.grid(True, which='major', linestyle='-', linewidth=1) 
    ax_load.grid(True, which='minor', linestyle=':', linewidth=0.5)

    # Graph10 : SoC, Voltage
    ax_SoC = fig.add_subplot(gs[1, 3])

    ax_SoC.plot(stateLog['time'], stateLog['battery_SoC'], label='SoC', color='blue')
    ax_SoC.set_xlabel('Time (s)')
    ax_SoC.set_ylabel('SoC (%)', color='blue')
    ax_SoC.set_ylim(0,100)
    ax_SoC.set_yticks(np.arange(0, 100+0.5, 20))
    ax_SoC.grid(True)

    ax_voltage = ax_SoC.twinx()  
    ax_voltage.plot(stateLog['time'], stateLog['battery_voltage'], label='voltage', color='red')
    ax_voltage.set_ylabel('Voltage(V)', color='red')
    ax_voltage.set_ylim(22,25.5)
    ax_voltage.set_yticks(np.arange(21, 25.6, 1.0))
    ax_voltage.set_yticks(np.arange(21, 25.6, 0.5),minor=True)
    ax_voltage.tick_params(axis='y', labelcolor='red') 
    ax_voltage.grid(True, linestyle=':', linewidth=0.5)  
    ax_voltage.grid(True, which='minor',linestyle=':', linewidth=0.5)  
    ax_SoC.set_title('SoC, Voltage')

    
    # Graph11 : Amps
    ax_amps = fig.add_subplot(gs[2, 3])
    ax_amps.plot(stateLog['time'], stateLog['Amps'], color='red')
    ax_amps.set_xlabel('Time (s)')
    ax_amps.set_ylabel('Current (A)')
    ax_amps.set_title('Current')
    ax_amps.grid(True)

    # Graph12 : Phase
    ax_phase = fig.add_subplot(gs[3, :])
    ax_phase.step(stateLog['time'], stateLog['phase'], where='post', color='purple')
    ax_phase.set_xlabel('Time (s)')
    ax_phase.set_ylabel('Phase')
    ax_phase.set_title('Mission Phases')
    ax_phase.set_yticks(phases)
    ax_phase.grid(True, which='major', linestyle='-', linewidth=1)
    ax_phase.grid(True, which='minor', linestyle=':', linewidth=0.5)

    plt.suptitle(f"Mission : {stateLog['mission'].iloc[0]}        Total flight time : {stateLog['time'].iloc[-1]:.2f}s        N_laps : {stateLog['N_laps'].iloc[-1]}", fontsize = 16, y = 0.95)

    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


if __name__=="__main__":

    a=loadAnalysisResults("'700773271413233544'")
    
    param = MissionParameters(
        m_takeoff= 10,
        max_speed= 40,                       # Fixed
        max_load_factor = 4.0,               # Fixed
        climb_thrust_ratio = 0.9,
        level_thrust_ratio = 0.5,
        turn_thrust_ratio = 0.5,               
        propeller_data_path = "data/propDataCSV/PER3_8x6E.csv", 

    )
    
    presetValues = PresetValues(
        m_x1 = 200,                         # g
        x1_time_margin = 150,                # sec
        
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
        
    missionAnalyzer = MissionAnalyzer(a, param, presetValues, propulsionSpecs)
    print(missionAnalyzer.run_mission3())
    visualize_mission(missionAnalyzer.stateLog)
 
