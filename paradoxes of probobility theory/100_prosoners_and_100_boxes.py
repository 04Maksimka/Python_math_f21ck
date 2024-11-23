import random

def each_prisoner_success(prisoner_id, _boxes):
    

    current_box = prisoner_id
    for _ in range(50):
        if _boxes[current_box - 1] == prisoner_id:
            return True
        current_box = _boxes[current_box - 1]
    return False

def all_prisoners_success(_boxes):
    success = True
    for prisoner_id in range(1, 100 + 1):
        success = success & each_prisoner_success(prisoner_id, _boxes)
    return success


prisoners_escaped = 0
for _ in range(100):
    boxes = [i for i in range(1, 100 + 1)]
    random.shuffle(boxes)
    if all_prisoners_success(boxes):
        prisoners_escaped += 1

print(prisoners_escaped / 100)

