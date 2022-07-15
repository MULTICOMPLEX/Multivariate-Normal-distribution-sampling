import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

mu = 0
sigma = 1

x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)

X, Y = np.meshgrid(x, y)

Z = 1 / np.sqrt(2 * np.pi) * sigma * np.exp(-0.5 * ( ((X-mu)/sigma)**2 + ((Y-mu)/sigma)**2))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(X, Y, Z, cmap=cm.viridis)

#cset = ax.contourf(X, Y, Z, zdir='z', offset=-1.0, cmap=cm.viridis)

#ax.set_zlim(-1.0, 0.4)


plt.show()