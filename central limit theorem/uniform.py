"""Here I tried to animate the CLT as it was done in the video of 3Blue1Brown:
https://www.youtube.com/watch?v=IaSGqQa5O-M&t=784s
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.signal import fftconvolve
from matplotlib.animation import FuncAnimation

uniform_dist = stats.uniform(loc=-0.5, scale=1)

delta = 1e-4
big_grid = np.arange(-10, 10, delta)

init = uniform_dist.pdf(big_grid) * delta

sequence_of_convolutions = [init]

for i in range(1, 100):
    new_convolution = fftconvolve(
        init,
        sequence_of_convolutions[-1],
        mode='same',
    )
    sequence_of_convolutions.append(new_convolution)

fig, ax = plt.subplots(figsize=(10, 6))

# Function to update the animation
def update(frame):
    ax.clear()
    ax.plot(big_grid  , sequence_of_convolutions[frame]  / delta)
    ax.set_title("Sampling from Uniform Distribution Approaching Normal Distribution")

ani = FuncAnimation(fig, update, frames=100, repeat=False)  # we see that we approach normal distribution

plt.show()