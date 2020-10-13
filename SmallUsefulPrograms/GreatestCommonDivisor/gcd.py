#Grant White
#September 2019


def euclid(a, b):
    """
    Euclidean Algorithm.
    Used to find the GCD of two integers.
    """

    dvd = a
    dvs = b
    r = None

    while r != 0:
        r = dvd % dvs
        dvd = dvs
        dvs = r

    return dvd


def ex_euclid(a, b):
    """
    Extended Euclidean Algorithm.
    Used to find both the GCD of two integers,
    and x,y where ax+by=gcd(a,b).
    """
    
    dvd = a
    dvs = b
    r = 1
    x = 1
    x_2 = 0
    y = 0
    y_2 = 1

    while r != 0:
        q = dvd // dvs
        r = dvd % dvs
        dvd = dvs
        dvs = r

        x_1 = x_2
        x_2 = x - q * x_2
        x = x_1

        y_1 = y_2
        y_2 = y - q * y_2
        y = y_1

    return dvd, x, y
