import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from timeit import default_timer as timer
from json import JSONEncoder
import json

start = timer()

filename = './data/Distribution'
#np.savez_compressed(filename, a=x2, b=y2, c=density)
loaded = np.load(filename +'.npz')

x2 = loaded['a']
y2 = loaded['b']
density = loaded['c']
Sine = loaded['d']

X, Y = np.meshgrid(x2, y2)

fig = plt.figure()

ax = fig.add_subplot(projection='3d')

cp = ax.plot_surface(X, Y, density, cmap=cm.viridis)

mz = np.max(density)

if(Sine):
 ax.contourf(X, Y, density, 10, zdir = 'z', offset = -1.5, cmap = cm.viridis, antialiased=True)
 ax.set_zlim(-1.5, 1.0)
else: 
 ax.contourf(X, Y, density, 60, zdir = 'x', offset = np.min(X), cmap = cm.viridis, antialiased=True)
 ax.contourf(X, Y, density, 60, zdir = 'y', offset = np.max(Y), cmap = cm.viridis, antialiased=True)
 ax.contourf(X, Y, density, 10, zdir='z', offset= -mz/2, cmap=cm.viridis, antialiased=True)
 ax.set_zlim(-mz/2, mz)

plt.colorbar(cp, shrink = 0.25)
ax.view_init(27, -21)

plt.title("Sine distribution, joined density", fontsize=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_zlabel("p(X)")

ax.text2D(0.03, 0.5, "p(Y)", transform=ax.transAxes)

end = timer()
print(" Duration ", end - start)

plt.show()

start = timer()

#######################################

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Serialization
numpyData = {"array1": x2, "array2": y2, "array3": density}
with open("./data/Distribution.json", "w") as write_file:
    json.dump(numpyData, write_file, cls=NumpyArrayEncoder)

# Deserialization
with open("./data/Distribution.json", "r") as read_file:
    decodedArray = json.load(read_file)
    x2 = np.asarray(decodedArray["array1"])
    y2 = np.asarray(decodedArray["array2"])
    density = np.asarray(decodedArray["array3"])

#######################################

Sine = loaded['d']

X, Y = np.meshgrid(x2, y2)

fig = plt.figure()

ax = fig.add_subplot(projection='3d')

cp = ax.plot_surface(X, Y, density, cmap=cm.viridis)

mz = np.max(density)

if(Sine):
 ax.contourf(X, Y, density, 10, zdir = 'z', offset = -1.5, cmap = cm.viridis, antialiased=True)
 ax.set_zlim(-1.5, 1.0)
else: 
 ax.contourf(X, Y, density, 60, zdir = 'x', offset = np.min(X), cmap = cm.viridis, antialiased=True)
 ax.contourf(X, Y, density, 60, zdir = 'y', offset = np.max(Y), cmap = cm.viridis, antialiased=True)
 ax.contourf(X, Y, density, 10, zdir='z', offset= -mz/2, cmap=cm.viridis, antialiased=True)
 ax.set_zlim(-mz/2, mz)

plt.colorbar(cp, shrink = 0.25)
ax.view_init(27, -21)

plt.title("Sine distribution, joined density", fontsize=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_zlabel("p(X)")

ax.text2D(0.03, 0.5, "p(Y)", transform=ax.transAxes)

end = timer()
print(" Duration ", end - start)

plt.show()







