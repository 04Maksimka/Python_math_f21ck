"""Script with Monti Hole paradox: https://en.wikipedia.org/wiki/Monty_Hall_problem"""
import random


# if we don't change ours chose
def monti_hole_no_change(tries: int):
    """Outputs the probability if you don't switch a chose.

    :param tries: amount of tries
    """
    positive = 0

    for i in range(tries):
        itr = random.randint(1, 3)
        if itr == 1:  # we consider that car always behind the 1 door
            positive += 1

    return positive / tries


# if we change ours chose
def monti_hole_change(tries: int):
    """Outputs the probability if you switch a chose.

    :param tries: amount of tries
    """
    positive = 0

    for i in range(tries):
        itr = random.randint(1, 3)
        if itr == 2:
            new_itr = 1
        elif itr == 3:
            new_itr = 1
        elif itr == 1:
            new_itr = random.choice([2, 3])

        if new_itr == 1:
            positive += 1

    return positive / tries


#  now we have two times more chances to win in this game.
if __name__ == '__main__':
    print(monti_hole_no_change(100))
    print(monti_hole_change(100))
