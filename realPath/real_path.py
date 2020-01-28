from os import path, getcwd

# delete this segment
print('path', path.realpath('.../.s/./././//..'))
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
        # т.к. более вероятно, что с конца будет элемент в реализации ниже
        curr_node = self.NIL.prev
        while curr_node != self.NIL:
            if curr_node == del_node:
                del_node.prev.follow = del_node.follow
                del_node.follow.prev = del_node.prev
                del del_node
                return
            curr_node = curr_node.prev

    def _back_list_crowl(self):
        curr_node = self.NIL.prev
        while curr_node != self.NIL:
            print(curr_node.key)
            curr_node = curr_node.prev

    def __str__(self):
        super().__str__()
        str_ = ''
        curr_node = self.NIL.follow
        while curr_node != self.NIL:
            str_ += str(curr_node.key)
            curr_node = curr_node.follow
        return str_


class RealPath:

    def __init__(self, path):
        super().__init__()
        self.user_path = path
        self._DLL = DLinkedList()
        self._num_remove_segments = 0

    def _fill_DLinkedList(self):
        path = self.user_path
        if path[0] == '.' :
            path = getcwd() + '/' + path
        for idx in range(0, len(path)):
            self._DLL.insert(path[idx])


    def real_path(self):
        self._fill_DLinkedList()
        NIL = self._DLL.NIL

        curr_symbol = self._DLL.NIL.prev
        self._clean_end()
        while curr_symbol != NIL:
            if curr_symbol.key == '/':
                if curr_symbol.prev.key == '.' and curr_symbol.prev.prev.key == '.' and curr_symbol.prev.prev.prev.key == '/':
                    self._two_dots(curr_symbol)
                elif curr_symbol.prev.key == '.' and curr_symbol.prev.prev.key == '/':
                    self._two_slash_and_dot(curr_symbol)
            curr_symbol = curr_symbol.prev




    def _two_dots(self, curr_symbol):
        self._DLL.delete(curr_symbol)
        self._DLL.delete(curr_symbol.prev)
        self._DLL.delete(curr_symbol.prev.prev)
        self._num_remove_segments += 1

    def _two_slash_and_dot(self, curr_symbol):
        self._DLL.delete(curr_symbol)
        self._DLL.delete(curr_symbol.prev)

    def _clean_end(self):
        curr_symbol = self._DLL.NIL.prev
        if curr_symbol.key == '.' and curr_symbol.prev.key == '.':
            self._DLL.delete(curr_symbol)
            self._DLL.delete(curr_symbol.prev)
            self._num_remove_segments += 1



RP = RealPath('.../.s/././../.d./../..')
RP.real_path()
print(RP._DLL)
