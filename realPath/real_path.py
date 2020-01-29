from os import getcwd


class NodeDLinkedList:

    def __init__(self, key=None, follow=None, prev=None):
        super().__init__()
        self.key = key
        self.follow = follow
        self.prev = prev


class DLinkedList:

    def __init__(self):
        super().__init__()
        self.nil = NodeDLinkedList()

    def insert(self, key):
        new_node = NodeDLinkedList(key=key, follow=self.nil)
        if self.nil.prev is not None:
            new_node.prev = self.nil.prev
            self.nil.prev.follow = new_node
        else:
            new_node.prev = self.nil
            self.nil.follow = new_node
        self.nil.prev = new_node

    def delete(self, del_node):
        curr_node = self.nil.prev
        while curr_node != self.nil:
            if curr_node == del_node:
                del_node.prev.follow = del_node.follow
                del_node.follow.prev = del_node.prev
                del del_node
                return
            curr_node = curr_node.prev

    def _back_list_crowl(self):
        curr_node = self.nil.prev
        while curr_node != self.nil:
            print(curr_node.key)
            curr_node = curr_node.prev

    def __str__(self):
        super().__str__()
        str_ = ''
        curr_node = self.nil.follow
        while curr_node != self.nil:
            str_ += str(curr_node.key)
            curr_node = curr_node.follow
        return str_


class RealPath:

    def __init__(self, path):
        super().__init__()
        self.user_path = path
        self._dll = DLinkedList()
        self._num_remove_segments = 0

    def _fill_DLinkedList(self):
        path = self.user_path
        if path[0] == '.':
            path = getcwd() + '/' + path
        for idx in range(0, len(path)):
            self._dll.insert(path[idx])

    def real_path(self):
        self._fill_DLinkedList()
        nil = self._dll.nil

        curr_symbol = self._dll.nil.prev
        self._clean_end()
        while curr_symbol != nil:
            if curr_symbol.key == '/':
                if curr_symbol.prev.key == '.' and curr_symbol.prev.prev.key == '.' and curr_symbol.prev.prev.prev.key == '/':
                    self._two_dots(curr_symbol)
                elif curr_symbol.prev.key == '.' and curr_symbol.prev.prev.key == '/':
                    self._two_slash_and_dot(curr_symbol)
                elif curr_symbol.prev.key == '/':
                    self._dll.delete(curr_symbol)
                elif self._num_remove_segments:
                    self._remove_segment(curr_symbol)
            curr_symbol = curr_symbol.prev
        self._clean_slash(nil)

        return str(self._dll)

    def _two_dots(self, curr_symbol):
        self._dll.delete(curr_symbol)
        self._dll.delete(curr_symbol.prev)
        self._dll.delete(curr_symbol.prev.prev)
        self._num_remove_segments += 1

    def _two_slash_and_dot(self, curr_symbol):
        self._dll.delete(curr_symbol)
        self._dll.delete(curr_symbol.prev)

    def _clean_end(self):
        curr_symbol = self._dll.nil.prev
        if curr_symbol.key == '.' and curr_symbol.prev.key == '.':
            self._dll.delete(curr_symbol)
            self._dll.delete(curr_symbol.prev)
            self._num_remove_segments += 1

    def _remove_segment(self, curr_symbol):
        self._dll.delete(curr_symbol)
        curr_symbol = curr_symbol.prev
        while curr_symbol.key != '/':
            self._dll.delete(curr_symbol)
            curr_symbol = curr_symbol.prev
        self._num_remove_segments -= 1

    def _clean_slash(self, nil):
        curr_symbol = nil.prev
        if curr_symbol.key == '/' and curr_symbol.prev != nil:
            self._dll.delete(curr_symbol)


import random
import string

def get_random_path(size):
    hash_and_dot = ['.', '/', '/./', '/../']
    hash_and_dot.extend(['' for _ in range(100)])
    random_str = ''.join(random.choice(string.ascii_letters + string.digits) + ''.join(random.choices(hash_and_dot, k=10)) for _ in range(size))
    return random.choice([random_str, './' + random_str])
