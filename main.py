from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import PHM08


filename = './data/train_FD002.txt'

PHMs = []
units = set()

with open(filename, 'r') as f:
    temp = PHM08.PHM08()
    temp.unit = 1
    units.add(temp.unit)
    for line in f:
        cols = line.strip().split()
        unit = int(cols[0])
        if unit not in units:
            PHMs.append(temp)
            temp = PHM08.PHM08()
            temp.unit = unit
            units.add(temp.unit)
        temp.time.append(int(cols[1]))
        for i in range(3):
            temp.settings[i].append(float(cols[i+2]))
        for i in range(21):
            temp.sensors[i].append(float(cols[i+5]))
    PHMs.append(temp)

phm = PHMs[14]
x = np.array(phm.settings[0])
y = np.array(phm.settings[1])
z = np.array(phm.settings[2])

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x, y, z)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

