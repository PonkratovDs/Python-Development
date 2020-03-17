class LCA:

    def __init__(self):
        self.lca_h = []
        self.lca_dfs_list = []
        self.lca_first = []
        self.lca_tree = []
        self.lca_dfs_used = []

        self.g = [[]]

    def lca_dfs(self, v: int, h=1):
        self.lca_dfs_used[v] = True
        self.lca_h[v] = True
        self.lca_dfs_list.append(v)
        for i in self.g[v]:
            if not self.lca_dfs_used[i]:
                self.lca_dfs(i, h + 1)
                self.lca_dfs_list.append(v)

    def lca_build_tree(self, i: int, l: int, r: int):
        if l == r:
            self.lca_tree[i] = self.lca_dfs_list[l]
        else:
            m = (l + r) >> 1
            self.lca_build_tree(i + i, l, m)
            self.lca_build_tree(i + i + 1, m + 1, r)
            if self.lca_h[self.lca_tree[i + i]] < self.lca_h[self.lca_tree[i + i + 1]]:
                self.lca_tree[i] = self.lca_tree[i + i]
            else:
                self.lca_tree[i] = self.lca_tree[i + i + 1]


    def lca_prepare(self, root:int):
        n = len(self.g)
        self.lca_h.extend([None for _ in range(n - len(self.lca_h))])
        self.lca_dfs_list.extend([None for _ in range(n - len(self.lca_dfs_list))])
        self.lca_dfs_used.extend([0 for _ in range(n)])

        self.lca_dfs(root)

        m = len(self.lca_dfs_list)
        self.lca_tree.extend([-1 for _ in range(len(self.lca_dfs_list) * 4 + 1)])
        self.lca_build_tree(1, 0, m-1)
        pass
