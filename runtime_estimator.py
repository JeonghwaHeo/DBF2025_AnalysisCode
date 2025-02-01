vsp_combination = 5*9*11*3
mission2_combination = 13*3*7*11*11
mission3_combination = 9*3*11*11
vsp_server_number = 24
mission_server_number = 150
#total_time = (vsp_combination * 15 / vsp_server_number) + (vsp_combination/mission_server_number)*(mission2_combination*0.18 + mission3_combination*1)
total_time = (vsp_combination * 15 / vsp_server_number) 
total_time = total_time/3600 

print(f"\ntotal running time = {total_time} hour")