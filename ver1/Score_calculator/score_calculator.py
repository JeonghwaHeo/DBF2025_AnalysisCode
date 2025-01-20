import numpy as np
import math

def score2(obj2, max_obj2):
    return 1 + (obj2 / max_obj2)

def score3(obj3, max_obj3):
    return 2 + (obj3 / max_obj3)

def score23(score2, score3, weight2):
    return score2 * weight2 + score3 * (1 - weight2)