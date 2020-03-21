g = [[2, 7, 8], [3, 6], [4, 5], [], [], [], [], [9, 12], [10, 11], [], [], []]
used = [False for _ in range(len(g) + 1)]


def dfs(v: int):
    global g, used
    used[v] = True
    for i in g[v]:
        if not used[i]:
            print(i)
            dfs(i)


