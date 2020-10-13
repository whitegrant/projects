#Grant White
#September 2019


"""
This program calculates what the n'th Fibonacci number is.

It calculates the number four different ways.

I wrote this program while learning Python to help me learn functions and using
recursion.
"""

from time import time
from math import sqrt


def naive(n):
    """extremely large temporal complexity"""

    if type(n) != int:
        raise ValueError("Number must be an integer.")
    elif n <= 0:
        raise ValueError("Number be greater than 0.")
    elif n == 1 or n == 2:
        return 1
    else:
        return naive(n - 1) + naive(n - 2)


def memoized(n):
    """good temporal complexity, bad spacial complexity"""

    if type(n) != int:
        raise ValueError("Number must be an integer.")
    elif n <= 0:
        raise ValueError("Number be greater than 0.")
    elif n < len(fib):
        return fib[n]
    else:
        if n == len(fib):
            fib.append(fib[n-1] + fib[n-2])
            return fib[n]
        else:
            fib.append(memoized(n-1) + fib[n-2])
            return fib[n]


def bottom_up(n):
    """good temporal complexity, good spacial complexity"""

    x,y,z = 1, 1, 0

    if type(n) != int:
        raise ValueError("Number must be an integer.")
    elif n <= 0:
        raise ValueError("Number be greater than 0.")
    elif n == 0 or n == 1:
        return 1
    else:
        for i in range(n - 2):
            z = x + y
            x = y
            y = z
        return z


def closed(n):
    """good temporal complexity, good spacial complexity"""
    #NOTE: round-off error occurs with n larger than 70
    return round((1 / sqrt(5)) * (((1 + sqrt(5)) / 2)**n - ((1 - sqrt(5)) / 2)**n))


def time_func(fib_func, n):
    """used to time how long a gived function with a given 'n' takes to run"""

    start = time()
    fib_func(n)
    return round(time() - start, 2)


if __name__ == "__main__":
    fib = [None, 1, 1]
