from os import path, getcwd

#delete this segment
path.realpath('./.s/./././//')
getcwd()
#________

class NodeDLinkedList:

    def __init__(self, key=None, follow=None, prev=None):
        super().__init__()
        self.key = key
        self.follow = follow
        self.prev = prev


class DLinkedList:

    def __init__(self):
        super().__init__()
        self.NIL = NodeDLinkedList()

    def insert(self, key):
        new_node = NodeDLinkedList(key=key, follow=self.NIL)
        if self.NIL.prev is not None:
            new_node.prev = self.NIL.prev
            self.NIL.prev.follow = new_node
        else:
            new_node.prev = self.NIL
            self.NIL.follow = new_node
        self.NIL.prev = new_node

    def delete(self, del_node):
        

    def _back_list_crowl(self):
        curr_node = self.NIL.prev
        while curr_node != self.NIL:
            print(curr_node.key)
            curr_node = curr_node.prev

DLL = DLinkedList()
for i in range(1, 101):
    DLL.insert(i)
DLL._back_list_crowl()
