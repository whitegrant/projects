def ind(i):
    return 2 * (i+1) + 1

def prime(N):
    #yields all primes less than N
    yield 2

    primes = [True] * (N // 2 - 1)
    for i, prime in enumerate(primes):
        if prime == True:
            for j, possible in enumerate(primes[i:]):
                if possible == True:
                    if ind(j) % ind(i) == 0:
                        primes[j] = False
        yield ind(i)


def prime_sieve(N):
    """Yield all primes that are less than N."""
    yield 2
    integers = [i for i in range(3, N, 2)]

    while len(integers) != 0:
        smallest = integers.pop(0)
        for i, num in enumerate(integers):
            if num % smallest == 0: #not prime
                del integers[i]
        yield smallest  #return prime
