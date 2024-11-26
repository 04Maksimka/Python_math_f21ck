"""Script with Bertran's paradox see more:
https://en.wikipedia.org/wiki/Bertrand_paradox_(probability)"""
import random
import numpy as np

AMOUNT_OF_CHORDS = 1000
TRI_SIDE = np.sqrt(3)


def probability_2_iid_angles_random_endpoints():
    """When we choose 2 independent points on the circle."""
    positive = 0
    for _ in range(AMOUNT_OF_CHORDS):
        alpha = random.random() * 2 * np.pi
        betta = random.random() * 2 * np.pi
        chord_len = np.sqrt(2) * np.sqrt(1 - np.cos(alpha - betta))
        if chord_len > TRI_SIDE:
            positive += 1

    return positive / AMOUNT_OF_CHORDS


def probability_1_angle_random_endpoints():
    """When we choose 1 independent point on the circle
    and another is fixed."""
    positive = 0
    for _ in range(AMOUNT_OF_CHORDS):
        alpha = random.random() * np.pi
        chord_len = np.sqrt(2) * np.sqrt(1 - np.cos(alpha))
        if chord_len > TRI_SIDE:
            positive += 1

    return positive / AMOUNT_OF_CHORDS


def probability_random_radius():
    """When we choose a center of a chord"""
    positive = 0
    for _ in range(AMOUNT_OF_CHORDS):
        x = random.random()
        chord_len = 2 * np.sqrt(1 - x ** 2)
        if chord_len > TRI_SIDE:
            positive += 1

    return positive / AMOUNT_OF_CHORDS

def probability_random_center():
    """When we choose a center of a chord"""
    pass


if __name__ == '__main__':
    print(probability_2_iid_angles_random_endpoints())
    print(probability_1_angle_random_endpoints())  # we see that these 2 ways:
    # probability_2_iid_angles and probability_1_angle gives the seme probability
    print(probability_random_radius())
