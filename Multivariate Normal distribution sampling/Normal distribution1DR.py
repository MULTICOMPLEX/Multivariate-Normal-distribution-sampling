import matplotlib.pyplot as plt
import numpy as np

mu = 0
sigma = 1

x = np.linspace(-4.0, 4.0, 100)

noisex = np.random.normal(mu, sigma, 10000000)

H, xedges = np.histogram(noisex, bins=x, density=True)

bin_cx = 0.5*(xedges[1:]+xedges[:-1])

plt.plot(bin_cx, H, color ='red')


plt.show()