import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def find_intersection(df1, df2):

    df1_sorted = df1.sort_values('RPM').reset_index(drop=True)
    df2_sorted = df2.sort_values('RPM').reset_index(drop=True)

    min_rpm = max(df1_sorted['RPM'].min(), df2_sorted['RPM'].min())
    max_rpm = min(df1_sorted['RPM'].max(), df2_sorted['RPM'].max())

    if max_rpm < min_rpm:
        return None, None

    rpm_interp = np.linspace(min_rpm, max_rpm, 1000)

    torque1 = np.interp(rpm_interp, df1_sorted['RPM'], df1_sorted['Torque'])  # Propeller
    torque2 = np.interp(rpm_interp, df2_sorted['RPM'], df2_sorted['Torque'])  # Motor
    diff = torque1 - torque2
    
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        return None, None

    idx = sign_changes[0]
    
    return rpm_interp[idx], torque1[idx]

def thrust_analysis(throttle:float, speed:float, voltage:float, Kv:float, R:float, max_current:float, max_power:float, csvPath:str, graphFlag:bool):

    
    propeller_df = pd.read_csv(csvPath,skiprows=[1])
    propeller_df.dropna(how='any',inplace=True)
    results=[]
    unique_rpms = sorted(propeller_df['RPM'].unique())

    
    for rpm in unique_rpms:
        
        subset = propeller_df[propeller_df['RPM'] == rpm].copy()
        subset = subset.sort_values(by='V(speed)')

        min_v = subset['V(speed)'].min()
        max_v = subset['V(speed)'].max()

        if min_v <= speed <= max_v:
            torque_at_v = np.interp(speed, subset['V(speed)'], subset['Torque'])
            thrust_at_v = np.interp(speed, subset['V(speed)'], subset['Thrust'])
        else:
            torque_at_v = None
            thrust_at_v = None

        results.append({
            'RPM': rpm,
            'Torque': torque_at_v,
            'Thrust': thrust_at_v
        })
    results_df = pd.DataFrame(results)
    results_df.dropna(how='any',inplace=True)

    df_sorted = results_df.sort_values('RPM').reset_index(drop=True)

    min_rpm = df_sorted['RPM'].min()
    max_rpm = df_sorted['RPM'].max()

    new_rpm_values = np.arange(min_rpm, max_rpm + 1, 100)

    torque_interpolated = np.interp(new_rpm_values, df_sorted['RPM'], df_sorted['Torque'])
    thrust_interpolated = np.interp(new_rpm_values, df_sorted['RPM'], df_sorted['Thrust'])

    df_expanded_results = pd.DataFrame({
        'RPM': new_rpm_values,
        'Torque': torque_interpolated,
        'Thrust': thrust_interpolated
    })

    I_list = np.arange(0,min(max_current,max_power/voltage)+0.5,1)
    RPM_list = Kv * (voltage * throttle - I_list * R) * 30 / math.pi
    Torque_list = I_list / Kv

    motor_df = pd.DataFrame({
        'I': I_list,
        'RPM': RPM_list,
        'Torque': Torque_list
    })

    if graphFlag == 1:
    
        plt.figure(figsize=(6, 3))
        plt.plot(df_expanded_results['RPM'], df_expanded_results['Torque'], label='Propeller')
        plt.plot(motor_df['RPM'],motor_df['Torque'], label='Motor')

        plt.xlabel('RPM')
        plt.ylabel('Torque')
        plt.title('Torque vs RPM')
        plt.grid(True)
        plt.legend()

        plt.show()
        

    RPM, Torque = find_intersection(motor_df, df_expanded_results)
    if RPM == None or Torque == None:
        print("적절하지 않음.\n")
        return 0,0,0,0,0
    I = np.interp(Torque,motor_df['Torque'],motor_df['I'])
    Power = voltage * I
    Thrust = np.interp(RPM,df_expanded_results['RPM'],df_expanded_results['Thrust'])

    return RPM, Torque, I, Power, Thrust

csvPath = r"C:\Users\user\Desktop\Propeller10x6E.csv"
speed = 0
Kv = 109.91
R = 0.062
throttle = 0.9
max_current = 60
max_power = 1332
voltage = 23.0
graphFlag = 1

# analysis sweep
# for throttle in np.arange(0.2,1.05,0.1):
#     print(f"throttle = {throttle:.2f}\n")
#     RPM, Torque, I, Power, Thrust = thrust_analysis(throttle, speed, voltage, Kv, R, max_current, max_power, csvPath, graphFlag)
#     print(f"RPM = {RPM:.0f}\nThrust(kg) = {Thrust:.2f}\nI(A) = {I:.2f}\nPower(W) = {Power:.2f}\nTorque(Nm) = {Torque:.2f}\n")
    

RPM, Torque, I, Power, Thrust = thrust_analysis(throttle, speed, voltage, Kv, R, max_current, max_power, csvPath, graphFlag)
print(f"RPM = {RPM:.0f}\nThrust(kg) = {Thrust:.2f}\nI(A) = {I:.2f}\nPower(W) = {Power:.2f}\nTorque(Nm) = {Torque:.2f}\n")