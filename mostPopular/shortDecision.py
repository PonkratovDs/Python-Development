from re import findall
from collections import Counter


def main():
    for i in range(10):
        words = findall(r'\w+', open('besy.txt').read())
        Counter(words).most_common(500)


if __name__ == '__main__':
    import timeit
    setup = """
from __main__ import main
"""
    statements = ['main()']
    for item in statements:
        print (
            '%s execute in %s seconds' %
            (item, min(
                timeit.repeat(
                    item, setup, timeit.default_timer, 5, 1))))
