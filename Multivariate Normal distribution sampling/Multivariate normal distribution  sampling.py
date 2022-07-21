import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import boost_histogram as bh
import functools, operator
import scipy.interpolate as ip
from timeit import default_timer as timer
from numpy.random import Generator, PCG64DXSM, SFC64, SeedSequence

Samples = 10000000
Histogram_size = 50
Smooth_factor = 1

mean = [0, 0, 0]
#mean = [2, 4]

#cov = [[1, 0.6], [0.6, 2]] 

mat = [[0, 1, 1],
      [ 1, 0, 1],
      [ 1, 1, 0]]


cov = np.cov(mat, rowvar=False)
print(cov)

rng = Generator(PCG64DXSM())
#print(rng.random())

start = timer()

a, b, c = rng.multivariate_normal(mean, cov, Samples).T

mina = np.min(a)
maxa = np.max(a)

minb = np.min(b)
maxb = np.max(b)

minc = np.min(c)
maxc = np.max(c)

end = timer()
print(" Duration multivariate_normal ", end - start)

start = timer()

hist = bh.Histogram(bh.axis.Regular(Histogram_size, mina, maxa),
                    bh.axis.Regular(Histogram_size, minb, maxb),
                    bh.axis.Regular(Histogram_size, minc, maxc))

hist = hist.fill(a, b, c)

hist = hist.project(0, 1) 

# Compute the areas of each bin
areas = functools.reduce(operator.mul, hist.axes.widths)

# Compute the density
density = hist.view() / hist.sum() / areas

end = timer()
print(" Duration bh.Histogram ", end - start)

fig = plt.figure()

ax = fig.add_subplot(projection='3d')

x = np.linspace(minb, maxb, Histogram_size)
y = np.linspace(minc, maxc, Histogram_size)

#f = ip.RectBivariateSpline(x, y, density, kx=4, ky=4)
f = ip.interp2d(x, y, density, kind='cubic', copy=False)

x2 = np.linspace(minb, maxb, Histogram_size * Smooth_factor)
y2 = np.linspace(minc, maxc, Histogram_size * Smooth_factor)

if Smooth_factor > 1: 
    density = f(x2, y2)

start = timer()

X, Y = np.meshgrid(x2, y2)

cp = ax.plot_surface(X, Y, density, cmap=cm.viridis)

mz = np.max(density)

ax.contourf(X, Y, density, 10, zdir='z', offset= -mz/2, cmap=cm.viridis, antialiased=True)
ax.contourf(X, Y, density, 60, zdir='x', offset=  minb, cmap=cm.viridis, antialiased=True)
ax.contourf(X, Y, density, 60, zdir='y', offset=  maxc, cmap=cm.viridis, antialiased=True)

ax.set_zlim(-mz/2, mz)

plt.colorbar(cp, shrink = 0.25)
ax.view_init(27, -21)

plt.title("Bivariate normal distribution sampling, joined density", fontsize=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_zlabel("p(X)")

ax.text2D(0.03, 0.5, "p(Y)", transform=ax.transAxes)

end = timer()
print(" Duration Plot ", end - start)

plt.show()


