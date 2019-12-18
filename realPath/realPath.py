import os


class PseudoStack:
    def __init__(self):
        self._items = []

    def isEmpty(self):
        return self._items == []

    def push(self, item):
        self._items.append(item)

    def popIdx(self, idx):
        if not self.isEmpty():
            self._items.pop(idx)
        else:
            raise IndexError

    def pop(self):
        if not self.isEmpty():
            self._items.pop()
        else:
            raise IndexError

    def peek(self):
        if not self.isEmpty():
            return self._items[-1]
        else:
            raise IndexError

    def getStack(self):
        return self._items


class RealPath:
    def __init__(self, path):
        self._path = path
        self._len = len(path)
        self._items = []

    def handlerStack(self):
        st = PseudoStack()
        if self._path[0] == '.':
            self._path = os.getcwd() + self._path[1:]
        for el in self._path:
            st.push(el)
        self.delDir(st)
        self.delDots(st)
        self.delSlash(st)
        return ''.join(self._items)

    def notDotOrSlash(self, idx):
        return self._items[idx] != '.' and self._items[idx] != '/'

    def delDir(self, st):
        count_del = 0
        count_dot = 0
        pos_del = []
        self._items = st.getStack()
        idx = len(self._items) - 1
        while idx > 0:
            if self._items[idx] == '.':
                count_dot += 1
            elif self._items[idx] == '/':
                if self.notDotOrSlash(idx + 1):
                    count_dot = 0
                if count_dot == 2:
                    count_del += 1
                count_dot = 0
            else:
                if count_del:
                    pos_del.append(idx)
                    if self._items[idx - 1] == '/':
                        count_del -= 1

            idx -= 1
        for idx in pos_del:
            st.popIdx(idx)

    def delDots(self, st):
        pos_del = []
        self._items = st.getStack()
        idx = len(self._items) - 1
        while idx > 0:
            if self._items[idx] == '.' and not self.notDotOrSlash(idx - 1):
                pos_del.append(idx)

            idx -= 1

        for idx in pos_del:
            st.popIdx(idx)

    def delSlash(self, st):
        pos_del = []
        self._items = st.getStack()
        idx = len(self._items) - 1
        while idx > 1:
            if self._items[idx] == '/' and self._items[idx - 1] == '/':
                pos_del.append(idx)

            idx -= 1
        for idx in pos_del:
            st.popIdx(idx)
        if st.peek() == '/':
            st.pop()

    def getRealPath(self):
        return self.handlerStack()


def main(path):
    rP = RealPath(path)
    return rP.getRealPath()


if __name__ == "__main__":
    main('./../.././//cd/etc/../pas.sw//.././sdf/.')


'''
class TasRealPathServer:

    def __init__(self, ip='0.0.0.0', port=5555, path='./', timeout=300):
        self.ip = ip
        self.port = port
        self.path = path
        self.timeout = timeout

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.ip, self.port))
        s.listen(1)
        conn, addr = s.accept()
        data = conn.recv(1000000)
        conn.send(self.parser(data))
        conn.close()

    def parser(self, data):
        answer = {}
        for k, v in json.loads(data).items():
            answer[k] = RealPath(v).getRealPath()
        return json.dumps(answer).encode()
'''
