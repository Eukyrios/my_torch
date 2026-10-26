def balance(lines_by_label, rng):
    smallest = min(len(lines) for lines in lines_by_label.values())
    balanced = []
    for lines in lines_by_label.values():
        balanced.extend(rng.sample(lines, smallest))
    rng.shuffle(balanced)
    return balanced
