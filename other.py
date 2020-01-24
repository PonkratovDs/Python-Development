import functools
import sys


def trace(func=None, *, handle=sys.stdout):
    if func is None:
        return lambda func: trace(func, handle=handle)

    @functools.wraps(func)
    def inner(*args, **kwargs):
        print(func.__name__, args, kwargs)
        return func(*args, **kwargs)
    return inner


@trace(handle=sys.stdout)
def identity(x):
    "i do"
    return x


print(identity('23'), identity.__doc__)

import time

def timethis(func=None, *, n_iter=100):
    if func is None:
        return lambda func: timethis(func, n_iter=n_iter)

    @functools.wraps(func)
    def inner(*args, **kwargs):
        print(func.__name__, end=' ... ')
        acc = float('inf')
        for i in range(n_iter):
            tick = time.perf_counter()
            result = func(*args, **kwargs)
            acc = min(acc, time.perf_counter() - tick)
        print(acc)
        return result
    return inner

#timethis(sum)(range(10 ** 6))

def profiled(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        inner.ncalls += 1
        return func(*args, **kwargs)

    inner.ncalls = 0
    return inner

@profiled
def identi(x):
    return x
'''for i in range(0, 10):
    identity(42)
print(identity.ncalls)'''

def memoized(func):
    cache = {}

    @functools.wraps(func)
    def inner(*args, **kwargs):
        key = args + tuple(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return inner

@memoized
def plus(x, y):
    return x + y
for i in range(0, 10):
    plus(i, 10 - i)

from warnings import warn_explicit

def deprecated(func):
    code = func.__code__
    warn_explicit(
        func.__name__ + ' is deprecated.',
        category=DeprecationWarning,
        filename=code.co_filename,
        lineno=code.co_firstlineno + 1)
    return func

@deprecated
def identit(x):
    x += 1
    return x

def pre(cond, message):
    def wrapper(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            assert cond(*args, **kwargs), message
            return func(*args, **kwargs)
        return inner
    return wrapper
from math import log

@pre(lambda x: x >= 0, 'nefative argument')
def checked_log(x):
    return log(x)

#checked_log(43)

f = functools.partial(sorted, key=lambda p: p[1])
print(f([('a', 34), ('b', 23)]))
