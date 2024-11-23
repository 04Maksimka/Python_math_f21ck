"""Script with Birthday paradox: https://en.wikipedia.org/wiki/Birthday_problem"""
import random

import numpy as np
import matplotlib.pyplot as plt


def birthday_probability(class_size, amount_of_cls: int = 10000):
    """ What's the probability that 2 people will have birthdays in one day

    :param amount_of_cls: amount of classes we measure
    :param class_size: there class_size people in the class
    """
    positive = 0
    for j in range(amount_of_cls):
        cls = []
        for i in range(class_size):
            cls.append(random.randint(1, 365))
        if _check_for_same_elem(cls):
            positive += 1
    return positive / amount_of_cls


def _check_for_same_elem(cls: list):
    flag = True
    if len(set(cls)) == len(cls):
        flag = False
    return flag

def plot_birth_paradox(end: int = 60):
    """
    Plot the graph
    :param end: how many points to plot
    """
    x = np.arange(1, end)
    y = np.array([birthday_probability(i) for i in x])
    plt.title('Birthday paradox')
    plt.plot(x ,y)


if __name__ == '__main__':
    print(birthday_probability(15))
    print(birthday_probability(23))
    print(birthday_probability(27))
    print(birthday_probability(100))
    plot_birth_paradox()  # just wait for 10 sec
    plt.show()
