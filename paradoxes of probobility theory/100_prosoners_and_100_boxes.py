"""
The 100 Prisoners Problem. 
Veritasium's video : https://www.youtube.com/watch?v=iSNsgj1OCLA
Wiki : https://en.wikipedia.org/wiki/100_prisoners_problem
"""
import random

NUM_OF_PRISONERS = 100
ITERATIONS = 1000


def each_prisoner_success(prisoner_id, _boxes):
    """Gets True if prisoner escapes."""
    current_box = prisoner_id
    for _ in range(NUM_OF_PRISONERS // 2):
        if _boxes[current_box - 1] == prisoner_id:
            return True
        current_box = _boxes[current_box - 1]
    return False


def all_prisoners_success(_boxes):
    """Gets True if all prisoners escape."""
    success = True
    for prisoner_id in range(1, NUM_OF_PRISONERS + 1):
        success = success & each_prisoner_success(prisoner_id, _boxes)
    return success


def prisoners_100_probability():
    prisoners_escaped = 0
    for _ in range(ITERATIONS):
        boxes = [i for i in range(1, NUM_OF_PRISONERS + 1)]
        random.shuffle(boxes)
        if all_prisoners_success(boxes):
            prisoners_escaped += 1
    return prisoners_escaped / ITERATIONS


if __name__ == '__main__':
    print(prisoners_100_probability())
