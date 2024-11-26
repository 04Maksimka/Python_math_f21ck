import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# TODO: доделать convolution самой функции с собой
# Define continuous functions
def gaussian(x, mean=0, std=1):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def uniform(x):
    return 1.0 * (-0.5 < x) * (x < 0.5)


def wedge_func(x):
    return np.clip(-np.abs(x) + 1, 0, 1)


def double_lump(x):
    return 0.45 * np.exp(-6 * (x - 0.5) ** 2) + np.exp(-6 * (x + 0.5) ** 2)


def plot_convolution_evolution(times):
    # Create an array of x values
    x = np.linspace(-5, 5, 1000)
    initial_function = uniform(x)
    # Perform times convolutions
    result = initial_function
    for _ in range(times):
        result = convolve(result, initial_function, mode='same')
    result = result / (np.sqrt(times + 1) * 1 / (2 * np.sqrt(3)))
    # Plotting the initial and final results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Initial Uniform Function')
    plt.plot(x, initial_function)
    plt.xlim(-5, 5)
    plt.xlabel('x')
    plt.ylabel('Amplitude')

    plt.subplot(1, 2, 2)
    plt.title(f'PDF after {times + 1} ')
    plt.plot(x, result)
    plt.xlim(-5, 5)
    plt.xlabel('x')
    plt.ylabel('Amplitude')

    plt.tight_layout()

plot_convolution_evolution(10)
plt.show()
