vsp_combination = 6*7*9*3
mission_combination = 10 * 10 * 10 * 4 + 10  * 10 * 10 * 4
server_number = 90

total_time = (vsp_combination*15 + vsp_combination* mission_combination*2)/(3600*server_number)
print(f"\ntotal running time = {total_time} hour")