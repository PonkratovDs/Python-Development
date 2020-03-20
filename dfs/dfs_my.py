from collections import OrderedDict, deque


class NodeG:

    def __init__(self, k, p=None, c=None):
        self.k = k
        self.p = p
        self.c = c


class UndirectedGraph:
    """data = {
        k1 : [values1],
        k2 : [values2]
        }"""

    def __init__(self, data):
        self.data = OrderedDict(data)
        items = {}
        for k, v in data.items():
            k = NodeG(k)
            items[k] = [NodeG(el) for el in v]
            if items[k]:
                k.c = items[k][0]
                for el in items[k]:
                    el.p = k
        self.vertices = OrderedDict(items)

    def __str__(self):
        _str = ''
        for k in self.data.keys():
            _str += str(k) + ' : ' + self.data[k].__str__()[1:-1] + '\n'
        return _str[:-1]

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, item):
        return self.vertices[item]


def dfs(g: UndirectedGraph, v: int):
    bypass = []
    v = NodeG(v)
    curr_v = v
    while v.c:
        bypass.append(curr_v)
        if curr_v.c is not None:
            curr_v = curr_v.c
            if g.vertices[curr_v.p]:
                curr_v.p.c = g.vertices[curr_v.p].pop(0)
            else:
                curr_v.p.c = None
        else:
            curr_v = curr_v.p
    return bypass


ug = UndirectedGraph({
    1: [2, 7, 8],
    2: [3, 6],
    3: [4, 5],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [9, 12],
    9: [10, 11],
    10: [],
    11: [],
    12: []
})

print(dfs(ug, 0))