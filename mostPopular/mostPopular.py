
class File:
    def __init__(self, fd):
        self._fd = fd
        self._data = {}

    def pushData(self):
        from re import compile
        pattern = compile(r'\w+')

        def helper(word):
            if self._data.get(word):
                self._data[word] += 1
            else:
                self._data[word] = 1

        def loop():
            for str in self._fd:
                math = pattern.findall(str)
                lM = len(math)
                if lM:
                    for i in range(lM):
                        helper(math[i])
        loop()

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
        fd.close()


if __name__ == '__main__':
    import timeit
    setup = """
from __main__ import File, main
"""
    statements = ['main()']
    for item in statements:
        print (
            '%s execute in %s seconds' %
            (item, min(
                timeit.repeat(
                    item, setup, timeit.default_timer, 5, 1))))
