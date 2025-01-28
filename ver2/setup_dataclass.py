from dataclasses import dataclass

@dataclass
class PresetValues:
    m_x1: float
    x1_flight_time: float
    number_of_motor : int

    max_battery_capacity: float
    min_battery_voltage: float
    propulsion_efficiency: float
    score_weight_ratio: float=0.5 

@dataclass
class PropulsionSpecs:
    propeller_data_path: str
    battery_data_path : str
    Kv : float
    R : float
    number_of_battery : int
    n_cell : int
    battery_Wh : float
    max_current : float
    max_power : float
    
@dataclass
class AircraftParamConstraints:
    # total mass of the aircraft
    m_total_max: float
    m_total_min: float
    m_total_interval: float
    
    # wing parameter ranges
    span_max: float
    span_min: float
    span_interval: float
    AR_max: float
    AR_min: float
    AR_interval: float
    taper_max: float
    taper_min: float
    taper_interval: float
    twist_max: float
    twist_min: float
    twist_interval: float

    # wing loading limit
    wing_loading_max: float
    wing_loading_min: float

@dataclass
class MissionParamConstraints:
    #Constraints for calculating mission2
    M2_throttle_climb_min : float
    M2_throttle_climb_max : float
    M2_throttle_turn_min : float
    M2_throttle_turn_max : float
    M2_throttle_level_min : float
    M2_throttle_level_max : float
    M2_throttle_analysis_interval : float
    
    #Constraints for calculating mission3  
    M3_throttle_climb_min : float
    M3_throttle_climb_max : float
    M3_throttle_turn_min : float
    M3_throttle_turn_max : float
    M3_throttle_level_min : float
    M3_throttle_level_max : float
    M3_throttle_analysis_interval : float