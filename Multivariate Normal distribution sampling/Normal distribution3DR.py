import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

mu = 0
sigma = 1

x = np.linspace(-4.0, 4.0, 100)
y = np.linspace(-4.0, 4.0, 100)

noisex = np.random.normal(mu, sigma, 20000000)
noisey = np.random.normal(mu, sigma, 20000000)

data = [noisex, noisey]

H, [xedges, yedges] = np.histogramdd(data, bins=(x,y), density=True)

fig = plt.figure()

ax = fig.add_subplot(projection='3d')

bin_cx = 0.5*(xedges[1:]+xedges[:-1])
bin_cy = 0.5*(yedges[1:]+yedges[:-1])

X, Y = np.meshgrid(bin_cx, bin_cy)

ax.plot_surface(X, Y, H, cmap=cm.viridis)

cset = ax.contourf(X, Y, H, zdir='z', offset=-0.15)

ax.set_zlim(-0.15, np.amax(H))

plt.show()