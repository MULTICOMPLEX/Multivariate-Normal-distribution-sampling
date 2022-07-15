import numpy as np
import matplotlib.pyplot as plt
 
# Creating a series of data of in range of -4, 4.
x = np.linspace(-4, 4, 200)

mu = 0
sigma = 1 

noise = np.random.normal(mu, sigma, 10000000)
count, bins = np.histogram(noise, 1000, range=(-4, 4), density=True)

bin_centers = 0.5*(bins[1:]+bins[:-1])

plt.plot(bin_centers, count, color='red') 

#Creating a Function.
def normal_dist(x, mu, sigma):
    prob_density = 1 / np.sqrt(2 * np.pi) * sigma * np.exp(-0.5*((x-mu)/sigma)**2)
    return prob_density
 
#Apply function to the data.
pdf = normal_dist(x, mu, sigma)

 
#Plotting the Results
plt.plot(x, pdf, color = 'green')

plt.xlabel('Data points')
plt.ylabel('Probability Density')

plt.show()