import time
import pandas as pd
import traceback
import numpy as np

import math
import matplotlib.pyplot as plt
from setup_dataclass import PropulsionSpecs
from scipy.interpolate import interp1d

import cProfile
import pstats
from pstats import SortKey


def determine_max_thrust_fast(speed:float, voltage:float, 
                              propulsionSpecs:PropulsionSpecs, propeller_array:np.ndarray):
    Kv = propulsionSpecs.Kv
    R = propulsionSpecs.R
    max_current = propulsionSpecs.max_current
    max_power = propulsionSpecs.max_power

    limited_current = min(max_current, max_power/voltage)
    
    propeller_array_fixspeed = propeller_fixspeed_data(speed,propeller_array)

    if all(propeller_array_fixspeed[0] == -1):
        return 0
    
    #print(propeller_array_fixspeed)
    #time.sleep(10000)
    #propeller_sortedby_torque = propeller_array_fixspeed[propeller_array_fixspeed[:, 1].argsort()]  
    
    propeller_sortedby_torque = propeller_array_fixspeed

    max_torque = limited_current/Kv

    propeller_max_rpm = np.interp(max_torque,propeller_sortedby_torque[:, 1],propeller_sortedby_torque[:, 0])
    motor_max_rpm = Kv*(voltage-limited_current*R)*30/math.pi
    
    # Skip finding stuff
    if motor_max_rpm >= propeller_max_rpm:
        max_thrust = np.interp(max_torque,propeller_sortedby_torque[:,1],propeller_sortedby_torque[:,2])
        return max(max_thrust,0)


    def rpm(I):
        return Kv * (voltage - I * R) * 30 / math.pi
    def torque(I):
        return I / Kv
    ## find intersection
    #I_list = np.arange(0,max(limited_current,0)+0.5,1)
    #RPM_list_fullthrottle = Kv * (voltage - I_list * R) * 30 / math.pi
    #Torque_list_fullthrottle = I_list / Kv

    #motor_results_array = np.column_stack((I_list,RPM_list_fullthrottle,Torque_list_fullthrottle))       
    #motor_sorted = motor_results_array[motor_results_array[:, 1].argsort()] 

    min_rpm = max(rpm(max(limited_current,0)), propeller_array_fixspeed[:, 0][0])
    max_rpm = min(rpm(0), propeller_array_fixspeed[:, 0][-1])


    if max_rpm < min_rpm: # Propeller Windmilling
        return 0
    
    # Find intersection

    min_rpm = np.ceil(min_rpm/100)*100
    max_rpm = np.floor(max_rpm/100)*100

    min_index = int((min_rpm - propeller_array_fixspeed[:,0][0])/100)
    max_index = int((max_rpm - propeller_array_fixspeed[:,0][0])/100)+1

    
    #torque1 = 
    #torque2 = propeller_array_fixspeed[:,1]

    rpm_interp = np.linspace(min_rpm, max_rpm, 500)

    #torque1 = np.interp(rpm_interp, rpm(I_list[::-1]), torque(I_list[::-1])) 
    #torque1= limited_current/Kv - (rpm_interp * np.pi/30)/(Kv**2 * voltage) * (1 + limited_current*R/voltage)
    torque1 = (1/(R*Kv)) * (voltage - (np.pi)/(30 * Kv) *propeller_array_fixspeed[min_index:max_index,0] )

    torque2 = propeller_array_fixspeed[min_index:max_index,1] 

    diff = torque1 - torque2

    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0: # Overcurrent
        max_thrust = np.interp(max_torque,propeller_sortedby_torque[:,1],propeller_sortedby_torque[:,2]) 
        return max(max_thrust,0)

    idx = sign_changes[0]

    #RPM = rpm_interp[idx]
    #max_thrust = np.interp(RPM,propeller_array_fixspeed[:, 0], propeller_array_fixspeed[:, 2])

    max_thrust = propeller_array_fixspeed[min_index+idx,2] 
    return max(max_thrust,0)

def determine_max_thrust(speed:float, voltage:float, 
                         propulsionSpecs:PropulsionSpecs, propeller_array:np.ndarray, 
                         graphFlag:bool):
    #old = determine_max_thrust_old(speed,voltage,propulsionSpecs,propeller_array,graphFlag)
    new = 0.5

    #profiler = cProfile.Profile()

    #profiler.enable()
    new= determine_max_thrust_fast(speed,voltage,propulsionSpecs,propeller_array)

    #profiler.disable()
    #   
    #   # Print stats sorted by cumulative time
    #stats = pstats.Stats(profiler)
    #stats.sort_stats(SortKey.TIME)
    #stats.print_stats(20)  # Show top 20 time-consuming lines

    #print(old-new)

    return new



def propeller_fixspeed_data_fast(speed, propeller_array):
    interval = 0.05
    if not hasattr(propeller_fixspeed_data_fast, "processed"):
        rpm_unique = np.sort(np.unique(propeller_array[:, 0]))
        v_max = propeller_array[:, 1].max()
        v_speeds = np.arange(0, v_max + interval, interval)  # 0.01 m/s intervals
        

        # Helper function to resize results
        maxAirspeed = []

        for val in rpm_unique:
            # Get all rows where the first column equals the current unique value
            rows = propeller_array[propeller_array[:, 0] == val]
            
            # Find the maximum value in the second column for the current group
            max_value = np.max(rows[:, 1])
            
            # Append the pair (val, max_value) to the result list
            maxAirspeed.append( max_value)
        
        max_speed_rpms = np.zeros(int(v_max/interval) + 1, dtype=np.int32)
        for i, max_speed in enumerate(maxAirspeed):
            idx_range = slice(int(max_speed/interval), len(max_speed_rpms))
            max_speed_rpms[idx_range] = i
            
        propeller_fixspeed_data_fast.max_speed_rpms = max_speed_rpms
        propeller_fixspeed_data_fast.rpm_starts = np.array([i*10 for i in range(len(rpm_unique))])

        # First interpolation: for each RPM interpolate over speeds
        rpm_data = {}
        for rpm in rpm_unique:
            mask = propeller_array[:, 0] == rpm
            speeds = propeller_array[mask, 1]
            torques = propeller_array[mask, 2]
            thrusts = propeller_array[mask, 3]

            # Create arrays with NaN where speed is out of range
            interp_torques = np.full_like(v_speeds, np.nan)
            interp_thrusts = np.full_like(v_speeds, np.nan)

            valid_mask = (v_speeds >= speeds.min()) & (v_speeds <= speeds.max())
            interp_torques[valid_mask] = np.interp(
                v_speeds[valid_mask], speeds, torques
            )
            interp_thrusts[valid_mask] = np.interp(
                v_speeds[valid_mask], speeds, thrusts
            )

            rpm_data[rpm] = (interp_torques, interp_thrusts)

        # Second interpolation: for each speed, interpolate over RPMs
        expanded_rpms = np.arange(int(rpm_unique.min()), int(rpm_unique.max()) + 1, 100)

        n_speeds = len(v_speeds)
        n_rpms = len(expanded_rpms)
        
        # Pre-allocate arrays for all speeds and RPMs
        torque_lookup = np.full((n_speeds, n_rpms), np.nan)
        thrust_lookup = np.full((n_speeds, n_rpms), np.nan)
        

        for i, v in enumerate(v_speeds):
            torques = np.array([rpm_data[rpm][0][i] for rpm in rpm_unique])
            thrusts = np.array([rpm_data[rpm][1][i] for rpm in rpm_unique])

            # Only interpolate where we have valid data (not NaN)
            valid = ~np.isnan(torques)
            if np.any(valid):
                interp_torques = np.interp(
                    expanded_rpms, rpm_unique[valid], torques[valid]
                )
                interp_thrusts = np.interp(
                    expanded_rpms, rpm_unique[valid], thrusts[valid]
                )

                torque_lookup[i]=interp_torques
                thrust_lookup[i]=interp_thrusts


        propeller_fixspeed_data_fast.v_speeds = v_speeds
        propeller_fixspeed_data_fast.torques = torque_lookup
        propeller_fixspeed_data_fast.thrusts = thrust_lookup
        propeller_fixspeed_data_fast.expanded_rpms = expanded_rpms
        propeller_fixspeed_data_fast.unique_rpms = rpm_unique
        propeller_fixspeed_data_fast.processed = True
        propeller_fixspeed_data_fast.maxAirspeed = maxAirspeed

    if not hasattr(propeller_fixspeed_data_fast, "cache"):
        propeller_fixspeed_data_fast.cache = {}

    idx = int(speed/interval+0.5)
    cached_speed = idx*interval 

    if cached_speed in propeller_fixspeed_data_fast.cache:
        return propeller_fixspeed_data_fast.cache[cached_speed]

    v_speeds = propeller_fixspeed_data_fast.v_speeds

    if (
        speed < v_speeds[0]
        or speed > v_speeds[-1]
    ):
        return np.array([-1])

    torque_lookup = propeller_fixspeed_data_fast.torques
    thrust_lookup = propeller_fixspeed_data_fast.thrusts
    maxAirspeed = propeller_fixspeed_data_fast.maxAirspeed

    # Final interpolation for requested speed
    
    
     
    # Find nearest pre-computed speeds
    #idx = np.searchsorted(propeller_fixspeed_data_fast.v_speeds, speed)


    v1 = v_speeds[idx]
    torque1=torque_lookup[idx]
    thrust1=thrust_lookup[idx]
    
    #v2 = v_speeds[idx+1]
    #torque2=torque_lookup[idx+1]
    #thrust2=thrust_lookup[idx+1]


    # Linear interpolation between pre-computed values (SLOW!)
    #t = 0

    #if (v2==v1): t = 0
    #else: t = (speed - v1) / (v2 - v1)
    #torques = (1 - t) * torque1 + t * torque2 
    #thrusts = (1 - t) * thrust1 + t * thrust2 
    
    torques =  torque1 
    thrusts =  thrust1

    # Get the rpm where the value is under max_speed

    idx2 = propeller_fixspeed_data_fast.max_speed_rpms[idx]
    start = propeller_fixspeed_data_fast.rpm_starts[idx2+1]
    #idx2 = np.searchsorted(maxAirspeed, speed)
    #start = idx2*10
    #minRPM = propeller_fixspeed_data_fast.unique_rpms[idx2] 

    # print(propeller_fixspeed_data_fast.expanded_rpms)
    # time.sleep(10000)
    # Get expanded RPMs reference

    rpms = propeller_fixspeed_data_fast.expanded_rpms[start:]
    
    # Create output array with minimal copying
    result = np.empty((len(rpms), 3))
    result[:, 0] = rpms
    result[:, 1] = torques[start:]
    result[:, 2] = thrusts[start:]
    

    propeller_fixspeed_data_fast.cache[cached_speed] = result

    return result


def propeller_fixspeed_data(speed,propeller_array):
    
    #profiler = cProfile.Profile()

    #profiler.enable()
    new = propeller_fixspeed_data_fast(speed,propeller_array)
        #return new

    #profiler.disable()
        
        # Print stats sorted by cumulative time
    #stats = pstats.Stats(profiler)
    #stats.sort_stats(SortKey.TIME)
    #stats.print_stats(20)  # Show top 20 time-consuming lines
    
    return new

    results=[]

    rpm_array = propeller_array[:, 0]      
    v_speed_array = propeller_array[:, 1]    
    torque_array = propeller_array[:, 2]    
    thrust_array = propeller_array[:, 3]     
    
    unique_rpms = sorted(set(rpm_array))
    
    for rpm in unique_rpms:
    
        mask = rpm_array == rpm

        v_subset = v_speed_array[mask]
        torque_subset = torque_array[mask]
        thrust_subset = thrust_array[mask]
    
        min_v = v_subset.min()
        max_v = v_subset.max()

        if min_v <= speed <= max_v:
            torque_at_v = np.interp(speed, v_subset, torque_subset)
            thrust_at_v = np.interp(speed, v_subset, thrust_subset)
            results.append({
            'RPM': rpm,
            'Torque': torque_at_v,
            'Thrust': thrust_at_v
            })
            
    results_array = np.array([(d['RPM'], d['Torque'], d['Thrust']) for d in results])
    if results_array.shape[0] == 0: return np.array([-1]) 
    
    rpm_values = results_array[:, 0] 
    torque_values = results_array[:, 1]  
    thrust_values = results_array[:, 2] 
    
    min_rpm = int(rpm_values.min())
    max_rpm = int(rpm_values.max())

    expanded_rpm_values = np.arange(min_rpm, max_rpm + 1, 100)

    torque_interpolated = np.interp(expanded_rpm_values, rpm_values, torque_values)
    thrust_interpolated = np.interp(expanded_rpm_values, rpm_values, thrust_values)
    
    propeller_array_fixspeed = np.column_stack((expanded_rpm_values, torque_interpolated, thrust_interpolated)) 
    
    

    np.testing.assert_allclose(new,propeller_array_fixspeed,  rtol=1e-20)

    return new
    return propeller_array_fixspeed

def thrust_reverse_solve(T_desired,speed,voltage, Kv, R, propeller_array):
    if T_desired == 0 : return 0,0,0,0,0
    # Add function attribute cache
    if not hasattr(thrust_reverse_solve, '_cache'):
        thrust_reverse_solve._cache = {}
    
    # Round inputs for better cache hits    
    key = (int(T_desired*1000+0.5), int(speed*100+0.5), int(voltage*100+0.5))

    if key in thrust_reverse_solve._cache:
        return thrust_reverse_solve._cache[key]

    propeller_array_fixspeed = propeller_fixspeed_data(speed,propeller_array)
    if (propeller_array_fixspeed==-1).all()==True : return 0,0,0,0,0
    
    propeller_sortedby_thrust = propeller_array_fixspeed[propeller_array_fixspeed[:, 2].argsort()]

    complex_values = propeller_sortedby_thrust[:,0] + 1j * propeller_sortedby_thrust[:,1]

    result = np.interp(
        T_desired,
        propeller_sortedby_thrust[:,2],
        complex_values)

    RPM_desired = result.real
    torque_desired = result.imag
    #RPM_desired = np.interp(T_desired,propeller_sortedby_thrust[:,2],propeller_sortedby_thrust[:,0])
    #torque_desired = np.interp(T_desired,propeller_sortedby_thrust[:,2],propeller_sortedby_thrust[:,1])
    
    I = Kv * torque_desired
    I = max(I,0)
    throttle = ((math.pi/30) * RPM_desired / Kv + I*R)/voltage
    throttle = max(throttle,0)
    Power = voltage * I
    result = (RPM_desired, torque_desired, I, Power, throttle)
    thrust_reverse_solve._cache[key] = result

    return result


def SoC2Vol(SoC,battery_array):
    #battery_array must be sorted by SoC
    
    voltage_array = battery_array[:,1]
    SoC_array = battery_array[:,3]
    voltage = np.interp(SoC,SoC_array, voltage_array)
    return voltage 
    

def thrust_analysis(throttle:float, speed:float, voltage:float, propulsionSpecs:PropulsionSpecs, propeller_array:np.ndarray, graphFlag:bool):

    Kv = propulsionSpecs.Kv
    R = propulsionSpecs.R
    max_current = propulsionSpecs.max_current
    max_power = propulsionSpecs.max_power
    
    expanded_results_array = propeller_fixspeed_data(speed,propeller_array)
    if (expanded_results_array==-1).all()==True : return 0,0,0,0,0
    
    I_list = np.arange(0,max(min(max_current,max_power/voltage),0)+0.5,1)
    RPM_list = Kv * (voltage * throttle - I_list * R) * 30 / math.pi
    Torque_list = I_list / Kv

    motor_results_array = np.column_stack((I_list,RPM_list,Torque_list))

    if graphFlag == 1:
    
        plt.figure(figsize=(6, 3))
        plt.plot( expanded_results_array[:,0], expanded_results_array[:,1], label='Propeller')
        plt.plot(motor_results_array[:,1],motor_results_array[:,2], label='Motor')

        plt.xlabel('RPM')
        plt.ylabel('Torque')
        plt.title('Torque vs RPM')
        plt.grid(True)
        plt.legend()

        plt.show()
        

    motor_sorted = motor_results_array[motor_results_array[:, 1].argsort()] 

    min_rpm = max(motor_sorted[:, 1].min(), expanded_results_array[:, 0].min())
    max_rpm = min(motor_sorted[:, 1].max(), expanded_results_array[:, 0].max())

    if max_rpm < min_rpm: # Propeller Windmilling
        # print("Can't make thrust")
        return expanded_results_array[0,0],0,0,0,0

    rpm_interp = np.linspace(min_rpm, max_rpm, 500)

    torque1 = np.interp(rpm_interp, motor_sorted[:, 1], motor_sorted[:, 2]) 
    torque2 = np.interp(rpm_interp, expanded_results_array[:, 0], expanded_results_array[:, 1]) 
    diff = torque1 - torque2

    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0: # Overcurrent
        Torque = motor_sorted[0,2]
        propeller_sorted = expanded_results_array[expanded_results_array[:, 1].argsort()] 
        RPM = np.interp(Torque, propeller_sorted[:, 1], propeller_sorted[:, 0])
        I = min(max_current,max_power/voltage)
        Power = I * voltage
        Thrust = np.interp(Torque,propeller_sorted[:, 1], propeller_sorted[:, 2])
        # print("Over-current")
        return RPM, Torque, I, Power, Thrust

    idx = sign_changes[0]

    RPM = rpm_interp[idx]
    Torque = torque1[idx]
    motor_torque_sorted = motor_sorted[motor_sorted[:, 2].argsort()]  
    I = np.interp(Torque,motor_torque_sorted[:, 2], motor_torque_sorted[:, 0])
    Power = I * voltage
    Thrust = np.interp(RPM,expanded_results_array[:, 0], expanded_results_array[:, 2])
    
    return RPM, Torque, I, Power, Thrust


if __name__=="__main__":
    csvPath = "data/propDataCSV/PER3_10x6E.csv"
    propeller_df = pd.read_csv(csvPath,skiprows=[1])
    propeller_df.dropna(how='any',inplace=True)
    propeller_df = propeller_df.sort_values(by=['RPM', 'V(speed)']).reset_index(drop=True)

    rpm_array = propeller_df['RPM'].to_numpy()
    v_speed_array = propeller_df['V(speed)'].to_numpy()
    torque_array = propeller_df['Torque'].to_numpy()
    thrust_array = propeller_df['Thrust'].to_numpy()
    propeller_array = np.column_stack((rpm_array, v_speed_array, torque_array, thrust_array))
    
    speed = 20          # m/s
    Kv = 109.91         # rad/s/V
    R = 0.062           # ohm
    throttle = 0.5      # 0~1
    max_current = 60    # A
    max_power = 1332    # W
    voltage = 23.0      # V
    graphFlag = 0       # 0 : off 1 : on
    
    RPM, Torque, I, Power, Thrust = thrust_analysis(throttle, speed, voltage, Kv, R, max_current, max_power, propeller_array, graphFlag)
    print(f"RPM = {RPM:.0f}\nThrust(kg) = {Thrust:.2f}\nI(A) = {I:.2f}\nPower(W) = {Power:.2f}\nTorque(Nm) = {Torque:.2f}\n")

    max_thrust = determine_max_thrust(speed, voltage,Kv,R,max_current,max_power,propeller_array,graphFlag)
    print(f"Maximum Thrust : {max_thrust}kg\n")
    
    T_desired = 1.0
    speed = 20
    Kv = 110
    R = 0.062
    voltage = 23.0
    RPM, Torque, I, Power, throttle = thrust_reverse_solve(T_desired,speed,voltage, Kv, R, propeller_array)
    print(f"RPM = {RPM:.0f}\nThrottle(kg) = {throttle:.2f}\nI(A) = {I:.2f}\nPower(W) = {Power:.2f}\nTorque(Nm) = {Torque:.2f}\n")
    
    
    max_battery_capacity = 2250
    
    df = pd.read_csv("data/Maxamps_2250mAh_6S.csv",skiprows=[1]) 
    time_array = df['Time'].to_numpy()
    voltage_array = df['Voltage'].to_numpy()
    current_array = df['Current'].to_numpy()
    dt_array = np.diff(time_array, prepend=time_array[0])
    cumulative_current = np.cumsum(current_array*dt_array)
    SoC_array = 100 - (1000 * cumulative_current / (36 * max_battery_capacity))
    mask = SoC_array >= 0
    time_array = time_array[mask]
    voltage_array = voltage_array[mask]
    current_array = current_array[mask]
    SoC_array = SoC_array[mask]
    battery_array = np.column_stack((time_array, voltage_array, current_array, SoC_array))
    battery_array = battery_array[battery_array[:, 3].argsort()]
    
    voltage_at_50= SoC2Vol(50,battery_array)
    print(f"battery voltage at 50% SoC : {voltage_at_50:.2f}\n")
    
    
    
    
