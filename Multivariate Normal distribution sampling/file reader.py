import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from timeit import default_timer as timer


filename = './data/Distribution'
#np.savez_compressed(filename, a=x2, b=y2, c=density)
loaded = np.load(filename +'.npz')

X = loaded['a']
Y = loaded['b']
density = loaded['c']
Sine = loaded['d']

X, Y = np.meshgrid(X, Y)

fig = plt.figure()

ax = fig.add_subplot(projection='3d')

start = timer()

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
print(" Duration Plot ", end - start)

plt.show()










