import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from json import JSONEncoder
import urllib.request, json

url = "https://www.phimagic.com/data/SineDistribution.json"

response = urllib.request.urlopen(url)

data = json.loads(response.read())
x2 = np.asarray(data["array1"])
y2 = np.asarray(data["array2"])
density = np.asarray(data["array3"])

#######################################

Sine = 1

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


plt.show()

