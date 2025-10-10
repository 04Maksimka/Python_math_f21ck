import numpy as np
import matplotlib.pyplot as plt


# Define the target distribution (e.g., a Gaussian distribution)
def target_distribution(x, mu=0, sigma=1):
    return (np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)) * 0.5
            + 0.5  * np.exp(-0.5 * ((x - mu + 3) / 2 * sigma) ** 2) / (2 * sigma * np.sqrt(2 * np.pi)))


# Define the proposal distribution (e.g., a uniform distribution)
def proposal_distribution(x):
    return 1 / (max_x - min_x)  # Uniform PDF


# Rejection sampling function
def rejection_sampling(target_dist, proposal_dist, min_x, max_x, M, num_samples):
    samples = []
    while len(samples) < num_samples:
        # Step 1: Sample from the proposal distribution
        x = np.random.uniform(min_x, max_x)

        # Step 2: Compute acceptance probability
        alpha = target_dist(x) / (M * proposal_dist(x))

        # Step 3: Accept or reject the sample
        u = np.random.uniform(0, 1)
        if u <= alpha:
            samples.append(x)

    return np.array(samples)


# Parameters
min_x = -8  # Lower bound of the proposal distribution
max_x = 8  # Upper bound of the proposal distribution
M = 5  # Constant to ensure M * q(x) >= p(x)
num_samples = 10000  # Number of samples to generate

# Generate samples using rejection sampling
samples = rejection_sampling(target_distribution, proposal_distribution, min_x, max_x, M, num_samples)

# Plot the results
plt.hist(samples, bins=70, density=True, alpha=0.6, label="Sampled Distribution")
x_values = np.linspace(min_x, max_x, 1000)
plt.plot(x_values, target_distribution(x_values), 'r', label="Target Distribution")
plt.title("Rejection Sampling")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()



