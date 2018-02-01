import math


def factorize(n):
    def prime(n):
        return not [x for x in range(2, int(math.sqrt(n)) + 1) if n % x == 0]
    primes = []
    candidates = range(2, n + 1)
    candidate = 2
    while not primes and candidate in candidates:
        if n % candidate == 0 and prime(candidate):
            primes += [candidate] + factorize(n // candidate)
        candidate += 1
    return primes


if __name__ == '__main__':
    print(factorize(26))