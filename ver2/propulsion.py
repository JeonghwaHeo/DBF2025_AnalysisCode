import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from setup_dataclass import PropulsionSpecs


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


def determine_max_thrust(speed:float, voltage:float, propulsionSpecs:PropulsionSpecs, propeller_array:np.ndarray, graphFlag:bool):

    Kv = propulsionSpecs.Kv
    R = propulsionSpecs.R
    max_current = propulsionSpecs.max_current
    max_power = propulsionSpecs.max_power
    
    propeller_array_fixspeed = propeller_fixspeed_data(speed,propeller_array)
    if (propeller_array_fixspeed==-1).all()==True : return 0
    
    propeller_sortedby_torque = propeller_array_fixspeed[propeller_array_fixspeed[:, 1].argsort()]  
    
    max_torque = min(max_current,max_power/voltage)/Kv
    propeller_max_rpm = np.interp(max_torque,propeller_sortedby_torque[:, 1],propeller_sortedby_torque[:, 0])
    motor_max_rpm = Kv*(voltage-min(max_current,max_power/voltage)*R)*30/math.pi


    if graphFlag == 1:
        
        I_list = np.arange(0,max(min(max_current,max_power/voltage),0)+0.5,1)
        RPM_list = Kv * (voltage - I_list * R) * 30 / math.pi
        Torque_list = I_list / Kv

        motor_results_array = np.column_stack((I_list,RPM_list,Torque_list))
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot( propeller_array_fixspeed[:,0], propeller_array_fixspeed[:,1], label='Propeller')
        axs[0].plot(motor_results_array[:,1],motor_results_array[:,2], label='Motor')

        axs[0].set_xlabel('RPM')
        axs[0].set_ylabel('Torque')
        axs[0].set_title('Torque vs RPM')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(propeller_array_fixspeed[:,0],propeller_array_fixspeed[:,2], label='Propeller')
        axs[1].set_xlabel('RPM')
        axs[1].set_ylabel('Thrust')
        axs[1].set_title('Thrust vs RPM')
        axs[1].grid(True)
        axs[1].legend()
        plt.show()
        
        
    if motor_max_rpm >= propeller_max_rpm:
        max_thrust = np.interp(max_torque,propeller_sortedby_torque[:,1],propeller_sortedby_torque[:,2])
        return max(max_thrust,0)
    else:
        
        ## find intersection
        I_list = np.arange(0,max(min(max_current,max_power/voltage),0)+0.5,1)
        RPM_list_fullthrottle = Kv * (voltage - I_list * R) * 30 / math.pi
        Torque_list_fullthrottle = I_list / Kv

        motor_results_array = np.column_stack((I_list,RPM_list_fullthrottle,Torque_list_fullthrottle))       
        motor_sorted = motor_results_array[motor_results_array[:, 1].argsort()] 

        min_rpm = max(motor_sorted[:, 1].min(), propeller_array_fixspeed[:, 0].min())
        max_rpm = min(motor_sorted[:, 1].max(), propeller_array_fixspeed[:, 0].max())

        if max_rpm < min_rpm: # Propeller Windmilling
            return 0

        rpm_interp = np.linspace(min_rpm, max_rpm, 500)

        torque1 = np.interp(rpm_interp, motor_sorted[:, 1], motor_sorted[:, 2]) 
        torque2 = np.interp(rpm_interp, propeller_array_fixspeed[:, 0], propeller_array_fixspeed[:, 1]) 
        diff = torque1 - torque2

        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign_changes) == 0: # Overcurrent
            max_thrust = np.interp(max_torque,propeller_sortedby_torque[:,1],propeller_sortedby_torque[:,2]) 
            return max(max_thrust,0)

        idx = sign_changes[0]

        RPM = rpm_interp[idx]
        max_thrust = np.interp(RPM,propeller_array_fixspeed[:, 0], propeller_array_fixspeed[:, 2])
        
        return max(max_thrust,0)
    
    
def propeller_fixspeed_data(speed,propeller_array):
    
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
    return propeller_array_fixspeed

def thrust_reverse_solve(T_desired,speed,voltage, Kv, R, propeller_array):
    
    if T_desired == 0 : return 0,0,0,0,0
    
    propeller_array_fixspeed = propeller_fixspeed_data(speed,propeller_array)
    if (propeller_array_fixspeed==-1).all()==True : return 0,0,0,0,0
    
    propeller_sortedby_thrust = propeller_array_fixspeed[propeller_array_fixspeed[:, 2].argsort()]
    
    RPM_desired = np.interp(T_desired,propeller_sortedby_thrust[:,2],propeller_sortedby_thrust[:,0])
    torque_desired = np.interp(T_desired,propeller_sortedby_thrust[:,2],propeller_sortedby_thrust[:,1])
    
    I = Kv * torque_desired
    I = max(I,0)
    throttle = ((math.pi/30) * RPM_desired / Kv + I*R)/voltage
    throttle = max(throttle,0)
    Power = voltage * I
    
    return RPM_desired, torque_desired, I, Power, throttle


def SoC2Vol(SoC,battery_array):
    #battery_array must be sorted by SoC
    
    voltage_array = battery_array[:,1]
    SoC_array = battery_array[:,3]
    voltage = np.interp(SoC,SoC_array, voltage_array)
    return voltage 
    



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
    
    
    
    