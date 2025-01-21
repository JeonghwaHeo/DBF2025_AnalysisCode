""" All the configs for the constraints of simulations etc. """

from dataclasses import dataclass

@dataclass
class __PhysicalConstants__:
    g: float = 9.81
    rho: float = 1.20

PhysicalConstants = __PhysicalConstants__()

@dataclass
class PresetValues:
    m_x1: float
    x1_flight_time: float

    max_battery_capacity: float
    min_battery_voltage: float

    Thrust_max: float
    propulsion_efficiency: float
    score_weight_ratio: float=1 # What is this? 
    # 경락: 이거 mission2/3 점수의 비율을 정해서 특정 형상이 mission2/3 에서 얼마나 좋은지 보는 거라는데 일단 무시 ㅋㅋ


@dataclass
class AircraftParamConstraints:
    """Constraints for constructing the aircraft"""

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

@dataclass
class MissionParamConstraints:
    """Constraints for calculating missions"""
    # total mass of the aircraft
    m_total_max: float
    m_total_min: float
    m_total_interval: float

    # Throttle
    throttle_climb_min: float
    throttle_turn_min: float
    throttle_level_min: float
    throttle_climb_max: float
    throttle_turn_max: float
    throttle_level_max: float
    throttle_analysis_interval: float
