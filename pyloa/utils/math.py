def memoize(func):
    """A function wrapper for a time/space trade-off."""
    table = dict()                      # function specific memoize table
    def wrappingfunction(*args):
        if args not in table:           # args tuple hasn't been seen yet
            table[args] = func(*args)   # envoke func call and store value
        return table[args]              # return stored value
    return wrappingfunction             # return wrappee


@memoize
def s2(n, k):
    """Calculates the Stirling number of the second kind."""
    if n == 0 or n != 0 and n == k:
        return 1
    if k == 0 or n < k:
        return 0
    return k * s2(n-1, k) + s2(n-1, k-1)