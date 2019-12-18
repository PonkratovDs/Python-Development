# import sys
import re
import timeit


class File(object):
    def __init__(self, fd):
        self._fd = fd
        self._data = {}

    def pushData(self):
        pattern = re.compile(r'\w+')
        # print(1)

        for line in self._fd:
            words = pattern.findall(line)
            for word in words:
                if word in self._data:
                    self._data[word] += 1
                else:
                    self._data[word] = 1

    def get500FamousWord(self):
        temp = sorted(
            list(
                self._data.items()),
            key=lambda x: x[1],
            reverse=True)
        self._data.clear()
        self._data.update(temp)
        lData = len(self._data)
        flag = 500
        if flag >= lData:
            print(list(self._data.keys()))
        else:
            print(list(self._data.keys())[:flag])


def main():
    for i in range(1, 10):
        with open('besy.txt', 'r') as fd:
            fl = File(fd)
            fl.pushData()
            # fl.get500FamousWord()


if __name__ == '__main__':
    setup = "from __main__ import File, main"
    statement = 'main()'
    print(
        '{} execute in {} seconds'.format(
            statement,
            min(
                timeit.repeat(statement, setup, timeit.default_timer, 10, 1)
            )
        )
    )
