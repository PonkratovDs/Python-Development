class BNode:

    def __init__(self, key=None):
        self.h = 0
        self.p = None
        self.k = key
        self.left = None
        self.right = None


class BTree:

    def __init__(self, key_root):
        self.root = BNode(key_root)

    def push_node(self, key):
        curr_node = self.root
        parent = curr_node
        h = 0
        while curr_node is not None:
            parent = curr_node
            if key <= curr_node.k:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
            h += 1
        new_node = BNode(key)
        new_node.h = h
        new_node.p = parent
        if key <= parent.k:
            parent.left = new_node
        else:
            parent.right = new_node

    def fill_btree(self, items):
        while items:
            self.push_node(items.pop())

    def find_node(self, key):
        curr_node = self.root
        while curr_node is not None:
            if curr_node.k == key:
                return curr_node
            elif key < curr_node.k:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
        return None

    '''def __str__(self):
        curr_node = self.root
        _str = ''
        while curr_node is not None:
            _str += str(curr_node.k) + '\n'
            if curr_node.left is not None:
                _str += str(curr_node.left.k) + '\t'
            elif curr_node.right is not None:
                _str += str(curr_node.right.k) + '\n' '''


def lca(g: BTree, v1: BNode, v2: BNode):
    if v1 is not None and v2 is not None and g is not None:
        if v1 == g.root or v2 == g.root:
            return None
        elif v1.k <= g.root.k < v2.k or v2.k <= g.root.k < v1.k:
            return g.root
        else:
            if v1.h < v2.h:
                return v1.p
            else:
                return v2.p
    else:
        return None


br = BTree(5)
br.fill_btree([234, 2, 56, 67, 23])
print(lca(br, br.find_node(234), br.find_node(67)).k)
