#Grant White
#December 2019

"""
This program is used for estimating pi.
It randomly and uniformly picks a bunch of points in the unit square and sees
    what percentage of them are in the unit circle. Multiplying that percentage
    by 4 (area of unit square) gives us an estimate for pi.
"""

from random import uniform

def pi(n):
    count = 0

    for _ in range(2**n):
        x = uniform(-1, 1)      #draw from unit square
        y = uniform(-1, 1)

        if x**2 + y**2 < 1:     #compare to unit circle
            count += 1

    return 4 * (count / 2**n)
