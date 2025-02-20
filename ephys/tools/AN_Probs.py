"""
AN probs
Given that we have N AN fibers converging on a cell, of which m are effective
what is the probability that we observe NO response?

Binomial problem.

"""
import scipy.stats as ST
import numpy as np

def main():
    probs = []

    Pr = 0.5 # release probability is 1 (repeated trials)
    N = 5
    m = 50

    print(1. - ST.binom.sf(0, m, Pr))

if __name__ == '__main__':
    main()