from dataclasses import replace, dataclass
from typing import List
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from config import PhysicalConstants, PresetValues
from models import MissionParameters, AircraftAnalysisResults, PlaneState, PhaseType, MissionConfig, Aircraft

## Constant values
g = PhysicalConstants.g
rho = PhysicalConstants.rho

class MissionAnalyzer():
    def __init__(self, 
                 analResult:AircraftAnalysisResults, 
                 missionParam:MissionParameters, 
                 presetValues:PresetValues,
                 dt:float=0.1):

        self.analResult = self._convert_units(analResult)
        self.aircraft = self.analResult.aircraft
        self.missionParam = missionParam
        self.presetValues = presetValues
        self.dt = dt


        self.analResult.m_fuel += missionParam.m_total - self.aircraft.m_total

        self.clearState()

        self.setAuxVals()

    def _convert_units(self, results: AircraftAnalysisResults) -> AircraftAnalysisResults:
        # Create new aircraft instance with converted units
        new_aircraft = Aircraft(
            # Mass conversions (g to kg)
            m_total=results.aircraft.m_total / 1000,
            m_fuselage=results.aircraft.m_fuselage / 1000,
            
            # Density conversions (g/mm³ to kg/m³)
            wing_density=results.aircraft.wing_density * 1e9,
            spar_density=results.aircraft.spar_density * 1e9,
            
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
            vertical_ThickChord=results.aircraft.vertical_ThickChord
        )
        
        # Create new analysis results with converted units
        return AircraftAnalysisResults(
            aircraft=new_aircraft,
            alpha_list=results.alpha_list,
            
            # Mass conversions (g to kg)
            m_fuel=results.m_fuel / 1000,
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
            CL_max=results.CL_max,
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
            CD_flap_zero=results.CD_flap_zero
        )

    def run_mission(self, missionPlan: List[MissionConfig],clearState = True) -> int:

        if(clearState): self.clearState()

        for phase in missionPlan:
            match phase.phaseType:
                case PhaseType.TAKEOFF:
                    self.takeoff_simulation()
                case PhaseType.CLIMB:
                    self.climb_simulation(phase.numargs[0],phase.numargs[1],phase.direction)
                case PhaseType.LEVEL_FLIGHT:
                    self.level_flight_simulation(phase.numargs[0],phase.direction)
                case PhaseType.TURN:
                    self.turn_simulation(phase.numargs[0],phase.direction)
                case _: 
                    raise ValueError("Didn't provide a correct PhaseType!")
            self.state.phase += 1
            #print("Changed Phase")
            if(not self._mission_viable()):
                return -1
        return 0

    ## TODO Maybe implement this?
    def _mission_viable(self):
        if(self.aircraft.m_total < 0):
            return False

        return True


    def run_mission2(self) -> float:

        mission2 = [
                MissionConfig(PhaseType.TAKEOFF, []),
                MissionConfig(PhaseType.CLIMB, [25,-140], "left"),
                MissionConfig(PhaseType.LEVEL_FLIGHT, [-152], "left"),
                MissionConfig(PhaseType.TURN, [180], "CW"),
                MissionConfig(PhaseType.CLIMB, [25,-10], "right"),
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

        if(result == -1): return 0
        
        return self.analResult.m_fuel / self.state.time

    def run_mission3(self) -> float:
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

        if(result == -1): return 0

        # Store starting index for each lap to handle truncation if needed
        N_laps = 1
        time_limit = 300 - self.presetValues.x1_flight_time  # 270 seconds

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
            N_laps += 1

            self.run_mission(lap2,clearState=False)

            # Check if we've exceeded time limit or voltage limit
            if (self.state.time > time_limit or 
                self.state.battery_voltage < self.presetValues.min_battery_voltage):
                #print("Time ran out")
                # Truncate the results and finish
                self.stateLog = self.stateLog[:lap_start_index]
                N_laps -= 1
                break

        # Calculate objective score
        obj3 = N_laps + 2.5 / self.presetValues.m_x1
        
        return obj3
        
    def clearState(self):
        self.state = PlaneState()
        self.stateLog = []

    # update the current state of the simulation
    def logState(self) -> None:
        # Append current state as a copy
        self.stateLog.append(PlaneState(
            position=self.state.position.copy(),
            velocity=self.state.velocity.copy(), 
            acceleration=self.state.acceleration.copy(),
            time=self.state.time,
            throttle=self.state.throttle,
            loadfactor=self.state.loadfactor,
            AOA=self.state.AOA,
            climb_pitch_angle=self.state.climb_pitch_angle,
            bank_angle=self.state.bank_angle,
            battery_capacity=self.state.battery_capacity,
            battery_voltage=self.state.battery_voltage,
            current_draw=self.state.current_draw,
            phase=self.state.phase
        ))


    def setAuxVals(self) -> None:
        self.weight = self.aircraft.m_total * g

        # self.v_stall = math.sqrt((2*self.weight) / (rho*self.analResult.Sref*self.analResult.CL_max))
        self.v_takeoff = (math.sqrt((2*self.weight) / (rho*self.analResult.Sref*self.analResult.CL_flap_max)))

        # Convert kgf to N
        self.T_max = self.presetValues.Thrust_max * g 

        # Calculate maximum thrust at each phase (N)
        self.T_takeoff = self.missionParam.throttle_takeoff * self.T_max
        self.T_climb = self.missionParam.throttle_climb * self.T_max
        self.T_level = self.missionParam.throttle_level * self.T_max
        self.T_turn = self.missionParam.throttle_turn * self.T_max

        ## calulate lift, drag coefficient at a specific AOA using interpolation function (with no flap)
        # how to use : if you want to know CL at AOA 3.12, use float(CL_func(3.12)) 
        # multiply (lh-lw) / lh at CL to consider the effect from horizontal tail wing
        # interpolate CD using quadratic function 
        # alpha_func : function to calculate AOA from given CL value

        tail_effect = float((self.analResult.Lh-self.analResult.Lw) / self.analResult.Lh)

         # Create focused alpha range from -10 to 10 degrees
        alpha_extended = np.linspace(-10, 10, 2000)  # 0.01 degree resolution
    
        # Create lookup tables
        CL_table = np.interp(alpha_extended, 
                            self.analResult.alpha_list,
                            tail_effect * np.array(self.analResult.CL))
        CD_table = np.interp(alpha_extended, 
                            self.analResult.alpha_list,
                            self.analResult.CD_total)
        
        # Create lambda functions for faster lookup
        self.CL_func = lambda alpha: np.interp(alpha, alpha_extended, CL_table)
        self.CD_func = lambda alpha: np.interp(alpha, alpha_extended, CD_table)
        self.alpha_func = lambda CL: np.interp(CL, CL_table, alpha_extended)

       # self.CL_func = interp1d(self.analResult.alpha_list,
       #                         float((self.analResult.Lh-self.analResult.Lw) / self.analResult.Lh) * np.array(self.analResult.CL), 
       #                         kind = 'linear', 
       #                         bounds_error = False, fill_value = "extrapolate")

       # self.CD_func = interp1d(self.analResult.alpha_list, 
       #                         self.analResult.CD_total, 
       #                         kind = 'quadratic',
       #                         bounds_error = False, fill_value = 'extrapolate')

       # self.alpha_func = interp1d(
       #                         (self.analResult.Lh-self.analResult.Lw) 
       #                         / self.analResult.Lh * np.array(self.analResult.CL), 
       #                            self.analResult.alpha_list, 
       #                            kind='linear',
       #                            bounds_error=False, fill_value='extrapolate') 
        return

    ## Previously battery
    def updateBatteryState(self, T) -> None :
        """
        T: 두 모터의 총 추력(N), 배터리 하나의 전기용량으로 계산하기 때문에 power 계산식에서 /2
    
        SoC vs Voltage 정보는 노션 Sizing/추친 참고
        """
        # TODO power 계산식 정확한 식으로 수정 필요하다.

        # SoC: in units of %
        SoC = self.state.battery_capacity / self.presetValues.max_battery_capacity* 100 

        battery_voltage_one_cell = 1.551936106250200e-09 * SoC**5 + -4.555798937007528e-07 * SoC**4 + 4.990928058346135e-05 * SoC**3 - 0.002445976965781 * SoC**2 + 0.054846035479305 * SoC + 3.316267645398081

        self.state.battery_voltage = battery_voltage_one_cell * 6
    
        # Calculate power required (simplified model: P = T^(3/2) / eta) (Watt)
        power = (T / 2) ** 1.5 / self.presetValues.propulsion_efficiency 

        # Calculate current draw (I = P / V) in Amps, convert to mA
        self.state.current_draw = (power / self.state.battery_voltage) * 1000.0 

        # Convert mA to mAh/s, calculate battery_capacity
        self.state.battery_capacity -= (self.state.current_draw / 3600.0) * self.dt     

        return

    def calculate_level_alpha(self, T, v):
        #  Function that calculates the AOA required for level flight using the velocity vector and thrust
        speed = fast_norm(v)
        def equation(alpha:float):
            CL = float(self.CL_func(alpha))
            L,_ = self.calculateLift(CL,float(speed))
            return float(L + T * math.sin(math.radians(alpha)) - self.weight)

        alpha_solution = fsolve(equation, 5, xtol=1e-8, maxfev=1000)
        return alpha_solution[0]
    
    def calculateLift(self, CL, speed:float=-1):
        if(speed == -1): speed = fast_norm(self.state.velocity)
        L = 0.5 * rho * fast_norm(self.state.velocity)**2 * self.analResult.Sref * CL
        return L, L/self.weight 

    def isBelowFlapTransition(self):
        return self.state.position[2] < self.missionParam.h_flap_transition
    
    # dt 0.01
    def takeoff_simulation(self):
        self.dt= 0.01
        self.state.velocity = np.array([0.0, 0.0, 0.0])
        self.state.position = np.array([0.0, 0.0, 0.0])
        self.state.time = 0.0

        self.state.battery_capacity = self.presetValues.max_battery_capacity
        
        # Ground roll until 0.9 times takeoff speed
        while fast_norm(self.state.velocity) < 0.9 * self.v_takeoff:
            
            self.state.time += self.dt

            self.state.acceleration = calculate_acceleration_groundroll(
                    self.state.velocity,
                    self.aircraft.m_total,
                    self.weight,
                    self.analResult.Sref,
                    self.analResult.CD_flap_zero, self.analResult.CL_flap_zero,
                    self.T_takeoff
                    )

            self.state.velocity -= self.state.acceleration * self.dt
            self.state.position += self.state.velocity * self.dt
            
            L, loadfactor = self.calculateLift(self.analResult.CL_flap_zero)
            
            self.state.loadfactor = loadfactor

            self.state.throttle = self.missionParam.throttle_takeoff
            
            self.state.AOA = 0
            self.state.climb_pitch_angle =np.nan
            self.state.bank_angle = np.degrees(0)


            self.updateBatteryState(self.T_takeoff)
            self.logState()



            
        # Ground rotation until takeoff speed    
        while 0.9 * self.v_takeoff <= fast_norm(self.state.velocity) <= self.v_takeoff:

            self.state.time += self.dt

            self.state.acceleration = calculate_acceleration_groundrotation(
                    self.state.velocity,
                    self.aircraft.m_total,
                    self.weight,
                    self.analResult.Sref,
                    self.analResult.CD_flap_max, self.analResult.CL_flap_max,
                    self.T_takeoff
                    )
            self.state.velocity -= self.state.acceleration * self.dt
            self.state.position += self.state.velocity * self.dt
            
            L, loadfactor = self.calculateLift(self.analResult.CL_flap_max)
            self.state.loadfactor = loadfactor

            self.state.throttle = self.missionParam.throttle_takeoff
            
            self.state.AOA=0
            self.state.climb_pitch_angle=np.nan
            self.state.bank_angle = np.degrees(0)


            self.updateBatteryState(self.T_takeoff)
            self.logState()

    # dt 0.01
    def climb_simulation(self, h_target, x_max_distance, direction):
        """
        Args:
            h_target (float): Desired altitude to climb at the maximum climb AOA (m)
            x_max_distance (float): Restricted x-coordinate for climb (m)
            direction (string): The direction of movement. Must be either 'left' or 'right'.
        """
        if self.state.position[2] > h_target: return
        self.dt = 0.01
        n_steps = int(60 / self.dt)  # Max 60 seconds simulation
        break_flag = False
        alpha_w_deg = 0 
        for step in range(n_steps):

            self.state.time += self.dt

            # Calculate climb angle
            gamma_rad = math.atan2(self.state.velocity[2], abs(self.state.velocity[0]))

            if direction == 'right':
                # set AOA at climb (if altitude is below target altitude, set AOA to AOA_climb. if altitude exceed target altitude, decrease AOA gradually to -2 degree)
                if(self.state.position[2] < self.missionParam.h_flap_transition and 
                   self.state.position[0] < x_max_distance):
                    alpha_w_deg = self.analResult.AOA_takeoff_max
                elif(self.missionParam.h_flap_transition <= self.state.position[2] < h_target and 
                     self.state.position[0] < x_max_distance):
                    load_factor = self.calculateLift(float(self.CL_func(self.analResult.AOA_climb_max)))[1]
            
                    if (load_factor < self.missionParam.max_load_factor and 
                        gamma_rad < math.radians(self.missionParam.max_climb_angle)):
                        alpha_w_deg = self.analResult.AOA_climb_max
                    elif (load_factor >= self.missionParam.max_load_factor and 
                          gamma_rad < math.radians(self.missionParam.max_climb_angle)):
                        alpha_w_deg = float(self.alpha_func((2 * self.weight * self.missionParam.max_load_factor)/
                                                          (rho * fast_norm(self.state.velocity)**2 * self.analResult.Sref)))
                    else:
                        alpha_w_deg -= 1
                        alpha_w_deg = max(alpha_w_deg, -5)
                else:
                    break_flag = True
                    if gamma_rad > math.radians(self.missionParam.max_climb_angle):
                        alpha_w_deg -= 1
                        alpha_w_deg = max(alpha_w_deg, -5)
                    else:
                        alpha_w_deg -= 0.1
                        alpha_w_deg = max(alpha_w_deg, -5)
            
            elif direction == 'left':
                # set AOA at climb (if altitude is below target altitude, set AOA to AOA_climb. if altitude exceed target altitude, decrease AOA gradually to -2 degree)
                if(self.state.position[2] < self.missionParam.h_flap_transition and 
                   self.state.position[0] > x_max_distance):
                    alpha_w_deg = self.analResult.AOA_takeoff_max
                elif(self.missionParam.h_flap_transition <= self.state.position[2] < h_target and 
                     self.state.position[0] > x_max_distance):
                    load_factor = self.calculateLift(float(self.CL_func(self.analResult.AOA_climb_max)))[1]
            
                    if (load_factor < self.missionParam.max_load_factor and 
                        gamma_rad < math.radians(self.missionParam.max_climb_angle)):
                        alpha_w_deg = self.analResult.AOA_climb_max
                    elif (load_factor >= self.missionParam.max_load_factor and 
                          gamma_rad < math.radians(self.missionParam.max_climb_angle)):
                        alpha_w_deg = float(self.alpha_func((2 * self.weight * self.missionParam.max_load_factor)/
                                                          (rho * fast_norm(self.state.velocity)**2 * self.analResult.Sref)))
                    else:
                        alpha_w_deg -= 1
                        alpha_w_deg = max(alpha_w_deg, -5)
                else:
                    break_flag = True
                    if gamma_rad > math.radians(self.missionParam.max_climb_angle):
                        alpha_w_deg -= 1
                        alpha_w_deg = max(alpha_w_deg, -5)
                    else:
                        alpha_w_deg -= 0.1
                        alpha_w_deg = max(alpha_w_deg, -5)            
                    
            # Calculate load factor
            if (self.isBelowFlapTransition()):
                CL = self.analResult.CL_flap_max
            else:
                CL = float(self.CL_func(alpha_w_deg))

            # print(f"Gamma: f{gamma_rad}") 
            self.state.acceleration = RK4_step(self.state.velocity,self.dt,
                         lambda v: calculate_acceleration_climb(v, self.aircraft.m_total,self.weight,
                                                                self.analResult.Sref,
                                                                self.CL_func,self.CD_func,
                                                                self.analResult.CL_flap_max,self.analResult.CD_flap_max,
                                                                alpha_w_deg,gamma_rad,
                                                                self.T_climb,
                                                                not self.isBelowFlapTransition()
                                                                ))
            self.state.velocity[2] += self.state.acceleration[2]*self.dt
            if direction == 'right':
                self.state.velocity[0] += self.state.acceleration[0]*self.dt
            else:
                self.state.velocity[0] -= self.state.acceleration[0]*self.dt
            
            self.state.position[0] += self.state.velocity[0]* self.dt
            self.state.position[2] += self.state.velocity[2]* self.dt

            L, loadfactor = self.calculateLift(CL)
            
            self.state.loadfactor = loadfactor

            self.state.throttle = self.missionParam.throttle_climb
             
            self.state.AOA = alpha_w_deg
            self.state.climb_pitch_angle =alpha_w_deg + math.degrees(gamma_rad)
            self.state.bank_angle = np.degrees(0)


            self.updateBatteryState(self.T_climb)
            self.logState()

            # break when climb angle goes to zero
            if break_flag == 1 and gamma_rad < 0:
                # print(f"cruise altitude is {z_pos:.2f} m.")
                break

    # dt = 0.1!
    def level_flight_simulation(self, x_final, direction):
        #print("\nRunning Level Flight Simulation...")
        max_steps = int(180/self.dt) # max 3 minuites
        # print(max_steps)
        step = 0
        self.dt = 0.1
        # Initialize vectors
        self.state.velocity[2] = 0  # Zero vertical velocity
        speed = fast_norm(self.state.velocity)

        if direction == 'right':
            self.state.velocity = np.array([speed, 0, 0])  # Align with x-axis
        elif direction=='left':
            self.state.velocity = np.array([-speed, 0, 0])
        
        cruise_flag = 0
        
        while step < max_steps:
            step += 1
            self.state.time += self.dt
            speed = fast_norm(self.state.velocity)
            
            # Calculate alpha_w first
            alpha_w_deg=self.calculate_level_alpha(self.T_level,self.state.velocity)
                
            

            # Speed limiting while maintaining direction

            if speed > self.missionParam.max_speed:  # Original speed limit
                cruise_flag = 1

            if cruise_flag == 1:
                self.state.velocity = self.state.velocity * (self.missionParam.max_speed / speed)
                T_cruise = 0.5 * rho * self.missionParam.max_speed**2 \
                                * self.analResult.Sref * float(self.CD_func(alpha_w_deg))

                alpha_w_deg = self.calculate_level_alpha(T_cruise,self.state.velocity)
                self.state.throttle = T_cruise / self.T_max

                self.updateBatteryState(T_cruise)
    
                self.state.acceleration = RK4_step(self.state.velocity,self.dt,
                             lambda v: calculate_acceleration_level(v,self.aircraft.m_total, 
                                                                    self.analResult.Sref,
                                                                    self.CD_func, alpha_w_deg,
                                                                    T_cruise))
            else:
                self.state.throttle= self.missionParam.throttle_level

                self.updateBatteryState(self.T_level)

                self.state.acceleration =  RK4_step(self.state.velocity,self.dt,
                             lambda v: calculate_acceleration_level(v,self.aircraft.m_total, 
                                                                    self.analResult.Sref,
                                                                    self.CD_func, alpha_w_deg,
                                                                    self.T_level))

                
            # Update Acc, Vel, position
            if direction == 'right': 
                self.state.velocity += self.state.acceleration * self.dt
            elif direction == 'left': 
                self.state.velocity -= self.state.acceleration * self.dt
            
            self.state.position[0] += self.state.velocity[0] * self.dt
            self.state.position[1] += self.state.velocity[1] * self.dt
            
            # Calculate and store results

            # TODO: 원본파일 (mission2 line 444)에선 speed를 다시 계산하지 않음!

            L,load_factor = self.calculateLift(float(self.CL_func(alpha_w_deg)))
            
            self.state.loadfactor = load_factor 
            self.state.AOA = alpha_w_deg
            self.state.bank_angle = math.degrees(0)
            self.state.climb_pitch_angle = np.nan

            
            self.logState()
            # Check if we've reached target x position
            if direction == 'right':
                if self.state.position[0] >= x_final:
                    break
            elif direction == 'left':
                if self.state.position[0] <= x_final:
                    break


        return
    def turn_simulation(self, target_angle_deg, direction):
      """
      Args:
          target_angle_degree (float): Required angle of coordinate level turn (degree)
          direction (string): The direction of movement. Must be either 'CW' or 'CCW'.
      """     
      speed = fast_norm(self.state.velocity) 

      # Initialize turn tracking
      target_angle_rad = math.radians(target_angle_deg)
      turned_angle_rad = 0

      # Get initial heading and setup turn center
      initial_angle_rad = math.atan2(self.state.velocity[1], self.state.velocity[0])
      current_angle_rad = initial_angle_rad

      # Pre-calculate constants
      dynamic_pressure_base = 0.5 * rho * self.analResult.Sref
      max_speed = self.missionParam.max_speed
      max_load = self.missionParam.max_load_factor
      weight = self.weight

      while abs(turned_angle_rad) < abs(target_angle_rad):
          self.state.time += self.dt

          if speed < max_speed:
              # Pre-calculate shared terms
              dynamic_pressure = dynamic_pressure_base * speed * speed
              
              CL = min(float(self.CL_func(self.analResult.AOA_turn_max)), 
                      float((max_load * weight)/(dynamic_pressure)))

              alpha_turn = float(self.alpha_func(CL))
              L = dynamic_pressure * CL
              phi_rad = math.acos(weight/L)
              
              a_centripetal = (L * math.sin(phi_rad)) / self.aircraft.m_total
              R = (self.aircraft.m_total * speed**2)/(L * math.sin(phi_rad))
              omega = speed / R

              self.state.loadfactor = 1 / math.cos(phi_rad)

              CD = float(self.CD_func(alpha_turn))
              D = CD * dynamic_pressure

              a_tangential = (self.T_turn - D) / self.aircraft.m_total
              self.state.throttle = self.missionParam.throttle_turn

              speed += a_tangential * self.dt
              self.updateBatteryState(self.T_turn)

          else:
              speed = max_speed
              dynamic_pressure = dynamic_pressure_base * speed * speed
              
              CL = min(float(self.CL_func(self.analResult.AOA_turn_max)), 
                      float((max_load * weight)/(dynamic_pressure)))
                      
              alpha_turn = float(self.alpha_func(CL))
              L = dynamic_pressure * CL
              phi_rad = math.acos(weight/L)

              a_centripetal = (L * math.sin(phi_rad)) / self.aircraft.m_total
              R = (self.aircraft.m_total * speed**2)/(L * math.sin(phi_rad))
              omega = speed / R

              self.state.loadfactor = 1 / math.cos(phi_rad)

              CD = float(self.CD_func(alpha_turn))
              D = CD * dynamic_pressure
              T = min(D, self.T_turn)
              self.state.throttle = T/self.T_max
              a_tangential = (T - D) / self.aircraft.m_total
              speed += a_tangential * self.dt

              self.updateBatteryState(T)

          # Calculate turn center
          sin_current = math.sin(current_angle_rad)
          cos_current = math.cos(current_angle_rad)
          
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
          sin_new = math.sin(current_angle_rad)
          cos_new = math.cos(current_angle_rad)
          
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
          self.state.bank_angle = math.degrees(phi_rad)
          self.state.climb_pitch_angle = np.nan

          self.logState() 


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
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def calculate_acceleration_groundroll(v, m_total, Weight,
                                      Sref,
                                      CD_zero_flap,CL_zero_flap,
                                      T_takeoff)->np.ndarray:
    # Function that calculates the acceleration of an aircraft during ground roll
    speed = fast_norm(v)
    D = 0.5 * rho * speed**2 * Sref * CD_zero_flap
    L = 0.5 * rho * speed**2 * Sref * CL_zero_flap
    a_x = (T_takeoff - D - 0.03*(Weight-L)) / m_total              # calculate x direction acceleration 
    return np.array([a_x, 0, 0])

def calculate_acceleration_groundrotation(v, m_total, Weight,
                                          Sref,
                                          CD_max_flap,CL_max_flap,
                                          T_takeoff)->np.ndarray:
    # Function that calculate the acceleration of the aircraft during rotation for takeoff
    speed = fast_norm(v)
    D = 0.5 * rho * speed**2 * Sref * CD_max_flap
    L = 0.5 * rho * speed**2 * Sref * CL_max_flap
    a_x = (T_takeoff - D - 0.03*(Weight-L)) / m_total            # calculate x direction acceleration 
    return np.array([a_x, 0, 0])

def calculate_acceleration_level(v, m_total, Sref, CD_func, alpha_deg, T):
    # Function that calculates the acceleration during level flight
    speed = fast_norm(v)
    CD = float(CD_func(alpha_deg))
    D = 0.5 * rho * speed**2 * Sref * CD
    a_x = (T * math.cos(math.radians(alpha_deg)) - D) / m_total
    return np.array([a_x, 0, 0])

def calculate_acceleration_climb(v, m_total, Weight, 
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
    theta_deg = math.degrees(gamma_rad) + alpha_deg
    theta_rad = math.radians(theta_deg)
    
    D = 0.5 * rho * speed**2 * Sref * CD
    L = 0.5 * rho * speed**2 * Sref * CL


    a_x = (T_climb * math.cos(theta_rad) - L * math.sin(gamma_rad) - D * math.cos(gamma_rad) )/ m_total
    a_z = (T_climb * math.sin(theta_rad) + L * math.cos(gamma_rad) - D * math.sin(gamma_rad) - Weight )/ m_total

    return np.array([a_x, 0, a_z])

def get_state_df(stateLog):
    # Convert numpy arrays to lists for proper DataFrame conversion
    states_dict = []
    for state in stateLog:
        state_dict = {
            'position': np.array([state.position[0],state.position[1],state.position[2]]),
            'velocity': np.array([state.velocity[0],state.velocity[1],state.velocity[2]]),
            'acceleration': np.array([state.acceleration[0],state.acceleration[1],state.acceleration[2]]),
            'time': state.time,
            'throttle': state.throttle,
            'loadfactor': state.loadfactor,
            'AOA': state.AOA,
            'climb_pitch_angle': state.climb_pitch_angle,
            'bank_angle': state.bank_angle,
            'phase': state.phase,
            'battery_capacity': state.battery_capacity,
            'battery_voltage': state.battery_voltage,
            'current_draw': state.current_draw
        }
        states_dict.append(state_dict)
    return pd.DataFrame(states_dict)

def visualize_mission(stateLog):
    """Generate all visualization plots for the mission in a single window"""
    stateLog = get_state_df(stateLog)

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)
    
    # Get phases and colors
    phases = stateLog['phase'].unique()
    colors = plt.cm.rainbow(np.random.rand(len(phases)))
    
    # 3D trajectory colored by phase
    ax_3d = fig.add_subplot(gs[0:2, 0], projection='3d')
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_3d.plot(stateLog[mask]['position'].apply(lambda x: x[0]), 
                  stateLog[mask]['position'].apply(lambda x: x[1]), 
                  stateLog[mask]['position'].apply(lambda x: x[2]),
                  color=color, label=f'Phase {phase}')
    ax_3d.set_xlabel('X Position (m)')
    ax_3d.set_ylabel('Y Position (m)')
    ax_3d.set_zlabel('Altitude (m)')
    ax_3d.set_title('3D Trajectory')
    # ax_3d.legend()
    
    # Set equal scale for 3D plot
    x_lims = ax_3d.get_xlim3d()
    y_lims = ax_3d.get_ylim3d()
    z_lims = ax_3d.get_zlim3d()
    max_range = max(x_lims[1] - x_lims[0], y_lims[1] - y_lims[0])
    x_center = (x_lims[1] + x_lims[0]) / 2
    y_center = (y_lims[1] + y_lims[0]) / 2
    z_center = (z_lims[1] + z_lims[0]) / 2
    ax_3d.set_xlim3d([x_center - max_range/2, x_center + max_range/2])
    ax_3d.set_ylim3d([y_center - max_range/2, y_center + max_range/2])
    ax_3d.set_zlim3d([0, z_lims[1]*1.5])

    # Top-down view colored by phase
    ax_top = fig.add_subplot(gs[2, 0])
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

    # Speed
    ax_speed = fig.add_subplot(gs[0, 1])
    speeds = np.sqrt(stateLog['velocity'].apply(lambda x: x[0]**2 + x[1]**2 + x[2]**2))
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_speed.plot(stateLog[mask]['time'], speeds[mask], 
                     color=color, label=f'Phase {phase}')
    ax_speed.set_xlabel('Time (s)')
    ax_speed.set_ylabel('Speed (m/s)')
    ax_speed.set_title('Speed by Phase')
    ax_speed.grid(True)
    # ax_speed.legend()

    # Load factor
    ax_load = fig.add_subplot(gs[0, 2])
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_load.plot(stateLog[mask]['time'], 
                    stateLog[mask]['loadfactor'], 
                    color=color, label=f'Phase {phase}')
    ax_load.set_xlabel('Time (s)')
    ax_load.set_ylabel('Load Factor')
    ax_load.set_title('Load Factor by Phase')
    ax_load.grid(True)
    # ax_load.legend()

    # Angles
    ax_angles = fig.add_subplot(gs[1, 1])
    ax_angles.plot(stateLog['time'], stateLog['AOA'], label='Angle of Attack')
    ax_angles.plot(stateLog['time'], stateLog['climb_pitch_angle'], label='Climb Pitch Angle')
    ax_angles.plot(stateLog['time'], stateLog['bank_angle'], label='Bank Angle')
    ax_angles.set_xlabel('Time (s)')
    ax_angles.set_ylabel('Angle (degrees)')
    ax_angles.set_title('Aircraft Angles')
    ax_angles.grid(True)
    ax_angles.legend()

    # Battery and throttle
    ax_battery = fig.add_subplot(gs[1, 2])
    ax_battery.plot(stateLog['time'], stateLog['battery_capacity'], label='Battery Capacity')
    ax_battery.set_xlabel('Time (s)')
    ax_battery.set_ylabel('Battery Capacity (mAh)')
    ax_battery.set_title('Battery Capacity')
    ax_battery.grid(True)

    ax_throttle = ax_battery.twinx()
    ax_throttle.plot(stateLog['time'], stateLog['throttle'], 
                    'r-', label='Throttle')
    ax_throttle.set_ylabel('Throttle', color='r')
    ax_throttle.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax_battery.get_legend_handles_labels()
    lines2, labels2 = ax_throttle.get_legend_handles_labels()
    ax_battery.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Side view colored by phase
    ax_side = fig.add_subplot(gs[2, 1:])
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_side.plot(stateLog[mask]['position'].apply(lambda x: x[0]), 
                    stateLog[mask]['position'].apply(lambda x: x[2]),
                    color=color, label=f'Phase {phase}')
    ax_side.set_xlabel('X Position (m)')
    ax_side.set_ylabel('Altitude (m)')
    ax_side.set_title('Side View')
    ax_side.grid(True)
    ax_side.set_aspect('equal')
    # ax_side.legend()

    plt.tight_layout()
    plt.show()
