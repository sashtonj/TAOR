import numpy as np

# Inputs from toar_mini_case.mos staff & D

terminal = np.zeros((68,1),dtype=int)
no_terminal = np.zeros((68,1),dtype=int)
demand = np.zeros((68,1),dtype=int)

for i in list(staff.keys()):
    if i[0] == 1:
        terminal[i[1]-1,0] = staff[i]
        demand[i[1]-1,0] = D[i]
    if i[0] == 2:
        no_terminal[i[1]-1,0] = staff[i]
        demand[i[1]-1,0] = demand[i[1]-1,0] + D[i]

# Prints Results
for i in range(1,69):
    print("Time Block:", i, "- Fare Collectors with Terminal: ", terminal[i-1,0], "- Fare Collectors without Terminal: ", no_terminal[i-1,0],"- Total Demand: ",demand[i-1,0])

V[1] = 5

# Terminates the loop
cond = cond + 1
