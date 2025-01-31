vsp_combination = 6*7*9*3
mission_combination = 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2
server_number = 30

total_time = (vsp_combination/server_number*45 + vsp_combination/server_number * mission_combination*3)/3600
print(f"\ntotal running time = {total_time} hour")