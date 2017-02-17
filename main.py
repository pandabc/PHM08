from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


filename = './data/train_FD002.txt'

unit = []
x = []
y = []
z = []

with open(filename, 'r') as f:
    for line in f:
        cols = line.strip().split()
        unit.append(int(cols[0]))
        x.append(float(cols[2]))
        y.append(float(cols[3]))
        z.append(float(cols[4]))

unit = np.array(unit)
x = np.array(x)
y = np.array(y)
z = np.array(z)

idx = np.where(unit > 0)

x = x[idx]
y = y[idx]
z = z[idx]

print len(z)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x, y, z)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

