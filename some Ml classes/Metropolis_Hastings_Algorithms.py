import numpy as np
import matplotlib.pyplot as plt


# Define the target distribution (e.g., a normal distribution)
def target_distribution(x, mu=0, sigma=1):
    return (np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)) * 0.5
            + 0.5  * np.exp(-0.5 * ((x - mu + 3) / 2 * sigma) ** 2) / (2 * sigma * np.sqrt(2 * np.pi)))


# Define the proposal distribution (e.g., a normal distribution centered at the current state)
def proposal_distribution(x, sigma_proposal=1):
    return np.random.normal(x, sigma_proposal)


# Metropolis-Hastings algorithm
def metropolis_hastings(target_dist, proposal_dist, n_samples, initial_state, proposal_sigma=1):
    samples = []
    current_state = initial_state

    for _ in range(n_samples):
        # Propose a new state
        proposed_state = proposal_dist(current_state, proposal_sigma)

        # Calculate the acceptance probability
        acceptance_ratio = target_dist(proposed_state) / target_dist(current_state)
        acceptance_probability = min(1, acceptance_ratio)

        # Accept or reject the proposed state
        if np.random.rand() < acceptance_probability:
            current_state = proposed_state

        # Save the current state
        samples.append(current_state)

    return np.array(samples)


# Parameters
n_samples = 100000
initial_state = 0.0
proposal_sigma = 1

# Run the Metropolis-Hastings algorithm
samples = metropolis_hastings(
    target_distribution,
    proposal_distribution,
    n_samples,
    initial_state,
    proposal_sigma,
)

# Plot the results
plt.hist(samples, bins=100, density=True, alpha=0.6, color='g', label='Sampled Distribution')
x = np.linspace(-10, 10, 1000)
plt.plot(x, target_distribution(x), 'r', label='Target Distribution')
plt.title('Metropolis-Hastings Sampling')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()