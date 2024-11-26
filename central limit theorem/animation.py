"""Script that models such a statistical experiment: there's a country with population_size
 and you sample sample_size people on any property (height, iq etc) and we see that more we
 make such samples - than closer and closer we approach a normal distribution - and we stabilize."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
population_size = 10000
sample_size = 10
num_samples = 1000

# Generate a population with a non-normal distribution (e.g., exponential, uniform) -- distribution of property
population = np.random.uniform(0, 1, size=population_size)

# Create a figure and axis for the animation
fig, ax = plt.subplots(figsize=(10, 6))


# Function to update the animation
def update(frame):
    ax.clear()

    # Take samples and calculate their means
    sample_means = [np.mean(np.random.choice(population, sample_size)) for _ in range(frame)]
    # Plot the histogram of sample means
    ax.hist(sample_means, bins=50, density=True, alpha=0.6, color='g', edgecolor='black')

    # Plot the theoretical normal distribution
    mu = np.mean(population)
    sigma = np.std(population) / np.sqrt(sample_size)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, y, color='red')

    # Set titles and labels
    ax.set_title("Sampling from Uniform Distribution Approaching Normal Distribution")
    ax.set_xlabel('Sample Means')
    ax.set_ylabel('Density')
    ax.set_xlim(mu - 4 * sigma, mu + 4 * sigma)
    ax.set_ylim(0, y.max() + 1)


# Create the animation
ani = FuncAnimation(fig, update, frames=num_samples, repeat=False, interval=1)

# Show the animation
plt.show()