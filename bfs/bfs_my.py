from collections import OrderedDict, deque


class UndirectedGraph:
    """data = {
        k1 : [values1],
        k2 : [values2]
        }"""

    def __init__(self, data):
        self.data = OrderedDict(data)

    def __str__(self):
        _str = ''
        for k in self.data.keys():
            _str += str(k) + ' : ' + self.data[k].__str__()[1:-1] + '\n'
        return _str[:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def bfs(u_g: UndirectedGraph, s: int, r: int):
    inf = float('inf')
    q = deque()
    q.append(s)
    used = [False for _ in range(len(u_g) + 1)]
    used[s] = True
    d = [inf for _ in range(len(u_g) + 1)]
    d[s] = 0
    while q:
        k = q.popleft()
        for v in u_g[k]:
            if not used[v]:
                q.append(v)
                used[v] = True
                if d[v] > d[k] + 1:
                    d[v] = d[k] + 1
    return d[r]


ug = UndirectedGraph({
    1: [2, 3],
    2: [4],
    3: [4, 7],
    4: [5],
    5: [6],
    6: [],
    7: [6]
})
print(ug)

print(bfs(ug, 1, 6))
