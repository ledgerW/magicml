def calculate_cost(storage):
    storage = int(storage)
    if storage > 100:
        rate = 1
    elif storage > 10:
        rate = 2
    else:
        rate = 4

    return rate * storage * 100
