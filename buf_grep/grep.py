import sys
import os

'''
идея: в начале берем кусок размером в строку в mess. Берем ее за базу динамики. Берем еще такую же часть. Проверяем на наличие \n. Если есть, то
часть до этого символа в дату, а оставшуюся часть в буфер. Смотрим есть ли в дате наши строки. Если нет, то берем еще кусок размером в строку в mess,
удаляя старую базу. Берем еще кусок такого же размера. Ищем. Таким образом у нас не будет случая, что подстрока есть, но мы ее не нашли. Не забываем
с новой строки добавлять в начало буфер
'''


class MatchString:
    def __init__(self, data, length, fd):
        self._data = data
        self._length = length
        self._fd = fd
        self._buffer = ''
        self._numberStr = 1
        self._pos = 0
        self._size = 2 * length
        self._EOF = None
        self._dict_m_string = {}

    def newStr(self):
        pos = self._data.find('\n')
        if pos != -1:
            self._pos = pos
            return True

    def fEOF(self, slice_str):
        if len(slice_str) == 0:
            self._EOF = True
            return True

    def getData(self):
        if self._length <= self._size:
            if self._buffer:
                self._data = self._buffer
                self._numberStr += 1
                self.clearBuffer()
            detuning = self._size - self._length
            slice_str = os.read(self._fd, detuning).decode('utf-8')
            if self.fEOF(slice_str):
                return
            self._data = self._data + slice_str
            if self.newStr():
                self._buffer = self._data[self._pos + 1:]
                self._data = self._data[:self._pos - 1]
            self._length = len(self._data)
        else:
            self.cutOffData()

    def cutOffData(self):
        if self._length < self._size / 2:
            pass  # ничего не делаем, поскольку просто попалась малая строка в файле
        else:
            self._data = self._data[int(self._size / 2):]
            self._length = len(self._data)

    def clearBuffer(self):
        self._buffer = ''

    def findMatchingBracketString(self, m_br):
        if self._data.find(m_br) != -1:
            self._dict_m_string[self._numberStr] = 0
            return True
        return False

    def findMatchingMessageString(self, m_mes):
        if self._data.find(m_mes) != - \
                1 and self._dict_m_string.get(self._numberStr) == 0:
            self._dict_m_string[self._numberStr] = 1
            return True
        return False

    def getEOF(self):
        return self._EOF

    def getDict(self):
        return self._dict_m_string


def main(args):
    m_br, m_mes = '[' + args[1] + ']', args[2]
    len_m_br, len_m_mes = len(m_mes), len(m_br)
    # на случай, если длина строки в скобках будет больше, чем в mes
    len_m_str = len_m_mes if len_m_mes >= len_m_br else len_m_br
    fd = os.open("log.txt", os.O_RDONLY)
    slice_str = os.read(fd, len_m_str).decode('utf-8')
    ms = MatchString(slice_str, len_m_str, fd)
    while True:
        if ms.getEOF():
            break
        ms.getData()
        ms.findMatchingBracketString(m_br)
        ms.findMatchingMessageString(m_mes)
        ms.cutOffData()
    os.close(fd)

    with open("log.txt", 'r') as f:
        for num, line in enumerate(f, 1):
            if ms.getDict().get(num):
                return line.rstrip()
    f.close()


if __name__ == '__main__':
    print(main(sys.argv))
