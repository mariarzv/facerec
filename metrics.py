
# calculate precision value
def calc_precision(tp, fp):
    if tp != 0 or fp != 0:
        return tp/(tp+fp)
    else:
        return 0


# calculate recall value
def calc_recall(tp, fn):
    if tp != 0 or fn != 0:
        return tp / (tp + fn)
    else:
        return 0


# calculate f1 score
def calc_f1_score(tp, fp, fn):
    p = calc_precision(tp, fp)
    r = calc_recall(tp, fn)
    if p != 0 or r != 0:
        return (2 * p * r) / (p + r)
    else:
        return 0

