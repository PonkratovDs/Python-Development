def InsertionSort(ar):
    for j in range(1, len(ar)):
        key = ar[j]
        i = j - 1
        while i >= 0 and ar[i] > key:
            ar[i + 1] = ar[i]
            i -= 1
        ar[i + 1] = key
    return ar


def FindMaxCrossingSubarray(A, low, mid, high):
    leftSum = float('-inf')
    Sum = 0
    maxLeft = 0
    maxRight = 0
    for i in range(low, mid + 1):
        Sum += A[i]
        if Sum > leftSum:
            leftSum = Sum
            maxLeft = i
    rightSum = float('-inf')
    Sum = 0
    for j in range(mid + 2, high + 1):
        Sum += A[j]
        if Sum > rightSum:
            rightSum = Sum
            maxRight = j
    return(maxLeft, maxRight, leftSum + rightSum)


def FindMaximumSubarrray(A, low, high):
    if high == low:
        return (low, high, A[low])

    else:
        mid = (low + high) // 2
        leftLow, leftHigh, leftSum = FindMaximumSubarrray(A, low, mid)
        rightLow, rightHigh, rightSum = FindMaximumSubarrray(A, mid + 1, high)
        crossLow, crossHigh, crossSum = FindMaxCrossingSubarray(
            A, low, mid, high)
        if leftSum >= rightSum and leftSum >= crossSum:
            return leftLow, leftHigh, leftSum
        elif rightSum >= leftSum and rightSum >= crossSum:
            return rightLow, rightHigh, rightSum
        else:
            return crossLow, crossHigh, crossSum


'''A = [24, -1, 30, -2]
print(FindMaximumSubarrray(A, 0, len(A) - 1))'''


class Heap:
    def __init__(self, A):
        self._items = A

    def heapSize(self):
        return len(self._items)

    def left(self, i):
        return self._items[2 * i + 1]

    def right(self, i):
        return self._items[2 * i + 2]

    def maxHeapify(self, i):
        l = 2 * i + 1
        r = 2 * i + 2
        if l < self.heapSize() and self._items[l] > self._items[i]:
            largest = l
        else:
            largest = i
        if r < self.heapSize() and self._items[r] > self._items[largest]:
            largest = r
        if largest != i:
            tmp = self._items[largest]
            self._items[largest] = self._items[i]
            self._items[i] = tmp
            if largest < self.heapSize():
                self.maxHeapify(largest)

    def buildMaxHeap(self):
        for i in reversed(range(0, self.heapSize() // 2)):
            self.maxHeapify(i)

    def heapLength(self):
        return (self.heapSize() - 2) // 2

    def __str__(self):
        for k in range(0, self.heapLength() + 1):
            print(self._items[k])
            if 2 * k + 2 >= self.heapSize():
                print(self.left(k))
            else:
                print(self.left(k), self.right(k))

    def heapSort(self):
        self.buildMaxHeap()
        ansver = []
        for i in reversed(range(0, self.heapSize())):
            tmp = self._items[0]
            self._items[0] = self._items[i]
            self._items[i] = tmp
            ansver.append(tmp)
            self._items.pop(i)
            self.maxHeapify(0)
        self._items = ansver

    def heapPop(self, idx=0):
        return self._items[idx]

    def changeEleminHeap(self, idx, val):
        self._items[idx] = val

    def getHeapArr(self) -> 'arr':
        return self._items


'''h = Heap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
h.heapSort()'''

'''import heapq

x = [4,1,3,2,16,9,10,14,8,7]
heapq.heapify(x)
print(heapq.heappop(x), heapq.heappop(x),end=' mm ')'''


class PriorityQueue():

    def __init__(self, heap: 'Heap class'):
        self.heap = heap
        self.HeapSize = heap.heapSize()

    def heapMaximum(self):
        return self.heap.heapPop()

    def heapExtractMax(self):
        if self.HeapSize < 1:
            raise Exception('Очередь пуста')
        max_ = self.heap.heapPop()
        self.heap.changeEleminHeap(0, self.heap.heapPop(self.HeapSize - 1))
        self.HeapSize -= 1
        self.heap.maxHeapify(0)
        return max_

    def heapIncreaseKey(self, i, key):
        if key < self.heap.heapPop(i):
            raise Exception('Новый ключ меньше текущего')
        self.heap.changeEleminHeap(i, key)
        parIdx = (i - 1) // 2
        while i > 0 and self.heap.heapPop(parIdx) < self.heap.heapPop(i):
            tmp = self.heap.heapPop(i)
            self.heap.changeEleminHeap(i, self.heap.heapPop(parIdx))
            self.heap.changeEleminHeap(parIdx, tmp)
            i = parIdx


'''p = PriorityQueue(h)
h.__str__()

print('max', p.heapExtractMax())'''

#items = [[], [], [], [], []]


class YoungTable:
    def __init__(self, items=[[float('+inf')]]):
        self._items = items

    def len(self):
        return len(self._items) * len(self._items[0])

    def __str__(self):
        str_ = ''
        for i in self._items:
            for j in i:
                str_ = str_ + str(j) + '   '
            str_ += '\n'
        return str_

    def sortTable(self):
        tmp = [
            self._items[j][i] for j in range(
                0, len(
                    self._items)) for i in range(
                0, len(
                    self._items[0]))]
        h = Heap(tmp)
        h.heapSort()
        tmp = h.getHeapArr()
        for j in range(0, len(self._items)):
            for i in range(0, len(self._items[0])):
                self._items[j][i] = tmp.pop()


'''yT = YoungTable([[124, 543, 3], [234, 462, 2]])
yT.sortTable()
print(yT.__str__())'''


def partition(a, p, r):
    x = a[r]
    i = p - 1
    for j in range(p, r):
        if a[j] <= x:
            i += 1
            tmp = a[j]
            a[j] = a[i]
            a[i] = tmp
    tmp = a[i + 1]
    a[i + 1] = a[r]
    a[r] = tmp
    return i + 1


def quick_sort(a, p, r):
    if p < r:
        q = partition(a, p, r)
        quick_sort(a, p, q - 1)
        quick_sort(a, q + 1, r)


class Stack:

    def __init__(self):
        self._items = []
        self._top = -1

    def stack_empty(self) -> 'bool':
        if self._top == -1:
            return True
        return False

    def push(self, val):
        self._top += 1
        self._items.append(val)

    def pop(self):
        if self.stack_empty():
            raise Exception('underflow')
        else:
            self._top -= 1
            return self._items[self._top + 1]


class Queue:

    def __init__(self):
        self._items = []
        self._tail = 0
        self._head = 0

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key):
        return self._items[key]

    def queue_overflow(self):
        if self._head == self._tail + \
                1 or (self._head == 1 and self._tail == len(self) - 1):
            raise Exception('queue overflow')

    def queue_underflow(self):
        if self._head == self._tail and len(self) != 0:
            raise Exception('queue underflow')

    def enqueue(self, x):
        self.queue_overflow()
        self._items.append(x)
        if self._tail == len(self) - 1:
            self._tail = 0
        else:
            self._tail += 1

    def dequeue(self):
        self.queue_underflow()
        x = self[self._head]
        if self._head == len(self) - 1:
            self._head = 0
        else:
            self._head += 1
        return x


class Deque:
    def __init__(self):
        self._items = []
        self._lTail = 0
        self._rTail = 0
        self._lHead = 0
        self._rHead = 0

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key):
        return self._items[key]

    def l_pop(self):
        self._lHead += 1
        if self._lHead == self._lTail:
            raise Exception('deq underflow')
        return self[self._lHead - 1]

    def r_pop(self):
        self._rHead = self._lTail - 1
        self._rTail = self._rHead - 1
        self._rHead += 1
        if self._rHead == self._rTail:
            raise Exception('deq underflow')
        return self[self._rHead - 1]

    def l_push(self, x):
        if self._lHead == self._lTail + \
                1 or (self._lHead == 0 and self._lTail == len(self) - 1):
            raise Exception('deq overflow')
        self._items.append(x)
        if self._lHead == len(self) - 1:
            self._lHead = 0
        else:
            self._lHead += 1


class List:
    def __init__(self, prev=None, next_=None, key=None):
        super().__init__()
        self.prev = prev
        self.next = next_
        self.key = key


'''
class DoubleLinkedList:

    def __init__(self):
        super().__init__()
        self._items = []
        self._head = List()
        self._tail = List()

    def list_search(self, k):
        x = self._head
        while x is not None and x.key != k:
            x = x.next
        return x

    def list_insert(self, x:'List(key)'):
        x.next = self._head
        if self._head is not None:
            self._head.prev = x
        self._head = x
        x.prev = None
        self._items.append(str(x.key))

    def list_delete(self, x):
        if x.prev is not None:
            x.prev.next = x.next
        else:
            self._head = x.next
        if x.next is not None:
            x.next.prev = x.prev

    def __str__(self):
        super().__str__()
        return ' '.join(self._items)

db = DoubleLinkedList()
db.list_insert(List(125))
db.list_insert(List(124))
print('Dl', db.__str__())
db.list_delete(List(122))
print('Dl', db.__str__())
'''


class SentielDoubleLinkedList:
    def __init__(self):
        super().__init__()
        self._nil = List(List(), List(), None)
        self._items = [self._nil]

    def list_insert(self, x):
        self._items.append(x)
        x.next = self._nil.next
        self._nil.next.prev = x
        self._nil.next = x
        x.prev = self._nil

    def list_search(self, k):
        x = self._nil.next
        while x != self._nil and x.key != k:
            x = x.next
        return x

    def __getitem__(self, key):
        return self._items[key]

    def list_delete(self, k):
        self[k].prev.next = self[k].next
        self[k].next.prev = self[k].prev

    def __str__(self):
        super().__str__()
        str_ = ''
        x = self._nil.next
        while x.key is not None:
            str_ += str(x.key) + ' '
            x = x.next
        return str_

    def list_union(self, dl):
        dl.list_delete(0)
        dl[1].next = self._nil
        self._nil.prev = dl[1]
        dl[1].prev = self[1]
        self[1].next = dl[1]
        dl[-1].prev = self[-1]
        self[-1].next = dl[-1]
        self._items.extend(dl._items[1:])
        dl._items = []


'''
db = SentielDoubleLinkedList()
db.list_insert(List(key=125))
db.list_insert(List(key=124))
db.list_insert(List(key=126))
print('sDl', db.__str__())
db.list_delete(1)
print('sDl', db.__str__())
dl = SentielDoubleLinkedList()
dl.list_insert(List(key=25))
dl.list_insert(List(key=24))
dl.list_insert(List(key=23))
db.list_union(dl)
print('sDl', db.__str__())'''


class Tool:
    def __init__(self, key, left, right, p):
        super().__init__()
        self.key = key
        self.left = left
        self.right = right
        self.p = p


class BinTree:
    def __init__(self):
        super().__init__()
        self._items = []
        self._length = 0
        self._len = 0

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key):
        return self._items[key]

    def create_bin_tree(self, arr):
        arr.sort()
        arr.append(None)
        len_ = len(arr)
        if not len_:
            return Tool(None, None, None, None)
        length = (len_ - 1) / 2 - 1 if len_ % 2 else (len_ - 2) / 2 - 1
        self._length = int(length)
        if len_ == 1:
            self._items.append(Tool(arr[0], None, None, None))
            return None
        elif len_ == 2:
            self._items.append(Tool(arr[0], arr[1], None, None))
        else:
            self._items.append(Tool(arr[0], arr[1], arr[2], None))

        for i in range(2, self._length):
            self._items.append(Tool(arr[i], arr[2 * i + 1], arr[2 * i + 2],
                                    arr[i - 1]))
        t = self._length
        if t > 0:
            if 2 * (t + 1) + 1 == len_:
                self._items.append(Tool(arr[t], arr[2 * t + 1], None,
                                        arr[t - 1]))
            elif 2 * (t + 1) + 2 == len_:
                self._items.append(Tool(arr[t], arr[2 * t + 1], arr[2 * t + 2],
                                        arr[t - 1]))
            else:
                print('bad')
        self._len = len(self)
        self._items.append(None)

    def __str__(self):
        super().__str__()
        self._length = len(self)
        str_ = ''
        for t in self._items:
            str_ += 'tool key: ' + str(t.key) +\
                    ' left: ' + str(t.left) + \
                    ' right: ' + str(t.right) + \
                    ' parent: ' + str(t.p) + '\n'
        return str_[:-1]

    def tree_search(self, k, i=0):
        el = self[i]
        if el is None:
            return
        elif el.key == k:
            return el
        elif el.left == k:
            return el
        elif el.right == k:
            return el
        else:
            i += 1
            self.tree_search(k, i)


'''b = BinTree()
b.create_bin_tree([6, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 54, 34, 7, 354, 65])
print(b.tree_search(6))'''


class VerTree:

    def __init__(self, key, child, p):
        super().__init__()
        self.key = key
        self.child = {}
        self.p = p


class BTree:

    def __init__(self):
        self._items = []

    def __getitem__(self, k):
        return self._items[k]

    def create_BTree(self, items):
        items.sort(reverse=True)
        max_idx = len(items) - 1
        while items:
            self._items.append(
                VerTree(
                    items.pop(), {
                        'left': None, 'right': None}, None))
        j = 1 if max_idx % 2 else 2
        i = 0
        while 2 * i + j <= max_idx:
            self[i].child['left'] = self[2 * i + 1]
            self[2 * i + 1].p = self[i]
            if 2 * i + 2 <= max_idx:
                self[i].child['right'] = self[2 * i + 2]
                self[2 * i + 2].p = self[i]
            if i > 0:
                self[i].p = self[i - 1]
            i += 1

    def __str__(self):
        super().__str__()
        str_ = ''
        for t in self._items:
            str_ += 'key: ' + str(t.key)
            if 'left' in t.child:
                if t.child['left'] is not None:
                    str_ += ' left: ' + str(t.child['left'].key)
                else:
                    str_ += ' left: ' + str(t.child['left'])
            if 'right' in t.child:
                if t.child['right'] is not None:
                    str_ += ' right: ' + str(t.child['right'].key)
                else:
                    str_ += ' right: ' + str(t.child['right'])
            if t.p is not None:
                str_ += ' parent: ' + str(t.p.key) + '\n'
            else:
                str_ += ' parent: ' + str(t.p) + '\n'
        return str_[:-1]

    def tree_search(self, x, k):
        if not 'left' in x.child or k == x.key:
            return x
        if k < x.key:
            return self.tree_search(x.child['left'], k)
        else:
            return self.tree_search(x.child['right'], k)


'''b = BTree()
b.create_BTree([214, 346, 8656, 3])
print(b.tree_search(b[0], 214).key)
print(b.__str__())'''


class RbNode:

    def __init__(self, key):
        super().__init__()

        self.key = key
        self.red = True
        self.left = None
        self.right = None
        self.parent = None


class RbTree:

    def __init__(self):
        self.root = None

    def search(self, key):
        current_node = self.root
        while current_node is not None and key != current_node.key:
            if key < current_node.key:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node

    def fix_tree(self, node):
        try:
            while node.parent.red is True and node is not self.root:
                if node.parent == node.parent.parent.left:
                    uncle = node.parent.parent.right
                    if uncle.red:
                        node.parent.red = False
                        uncle.red = False
                        node.parent.parent.red = True
                        node = node.parent.parent
                    else:
                        if node == node.parent.right:
                            node = node.parent
                else:
                    try:
                        uncle = node.parent.parent.left
                        if uncle.red:
                            node.parent.red = False
                            uncle.red = False
                            node.parent.parent.red = True
                    except AttributeError:
                        print("DOES NOT GAVE UNCLE")
                        break
            self.root.red = False
        except AttributeError:
            print("\n\nTree BUILT")

    def insert(self, key):
        node = RbNode(key)
        if self.root is None:
            node.red = False
            self.root = node
            return
        last_node = self.root
        while last_node is not None:
            potential_parent = last_node
            if key < last_node.key:
                last_node = last_node.left
            else:
                last_node = last_node.right
        node.parent = potential_parent
        if key < potential_parent.key:
            node.parent.left = node
        else:
            node.parent.right = node
        node.left = None
        node.right = None
        self.fix_tree(node)

    def del_node(self, key):
        current_node = self.search(key)
        if current_node is None:
            return
        elif current_node.parent is None:
            if current_node == self.root:
                self.root = None
            return
        elif current_node.parent.left == current_node:
            current_node.parent = None
        else:
            current_node.parent = None

    '''def transplant(self, u, v):
        if u == self.root:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def minimum(self, z):
        current_node = z
        while current_node.left is not None:
            current_node = current_node.left
        return current_node

    def delete(self, key):
        z = self.search(key)
        y = z
        y_original_color = y.red
        if z.left is None:
            x = z.right
            self.transplant(z, z.right)
        elif z.right is None:
            x = z.left
            self.transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.red
            x= y.right
            if y.parent == z:
                x.parent = y
            else:
                self.transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self.transplant(z, y)
            y.left = z.left
            z.left.parent = y
            y.red = z.red
        if y_original_color == False:
            self.fix_tree(x)'''


'''t = RbTree()
t.insert(12)
t.insert(45)
t.del_node(415)
print(t.root.key)'''


class AVLNode:

    def __init__(self, key):
        super().__init__()
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
        self._height = 0

    def heighted(self, mt):
        if mt == 'left':
            return self.left._height if self.left else 0
        elif mt == 'right':
            return self.right._height if self.right else 0

    def bfactor(self):
        return self.heighted('right') - self.heighted('left')

    def fixheight(self):
        hl = self.heighted('left')
        hr = self.heighted('right')
        self._height = (hl if hl > hr else hr) + 1

    def rotateright(self):
        q = self.left
        self.left = q.right
        q.right = self
        self.fixheight()
        q.fixheight()
        return q

    def rotateleft(self):
        p = self.right
        self.right = p.left
        p.left = self
        self.fixheight()
        p.fixheight()
        return p

    def balance(self):
        self.fixheight()
        if self.bfactor() == 2:
            if self.right.bfactor() < 0:
                self.right = self.right.rotateright()
            return self.rotateleft()
        if self.bfactor() == -2:
            if self.left.bfactor() > 0:
                self.left = self.left.rotateleft()
            return self.rotateright
        return self


def _isEmpty(func):
    def wrapper(self):
        if self.root is None:
            return


class AVLTree:

    def __init__(self):
        super().__init__()
        self.root = None

    def insert(self, key):
        node = AVLNode(key)
        if self.root is None:
            self.root = node
            return
        last_node = self.root
        while last_node is not None:
            potential_parent = last_node
            if key < last_node.key:
                last_node = last_node.left
            else:
                last_node = last_node.right
        node.parent = potential_parent
        if key < potential_parent.key:
            node.parent.left = node
        else:
            node.parent.right = node
        node.left = None
        node.right = None
        node.balance()

    #@_isEmpty
    def findmin(self, p=None):
        if p is None:
            p = self.root
        return self.findmin(p.left) if p.left else p

    def removemin(self, p=None):
        if p is None:
            p = self.root
        if p.left is None:
            return p.right
        p.left = self.removemin(p.left)
        return p.balance()

    def remove(self, key, p=None):
        if p is None:
            pass


a = AVLTree()
a.insert(125)
a.insert(54)
a.removemin()
print(a.findmin().key)

import functools
import sys


def trace(func=None, *, handle=sys.stdout):
    if func is None:
        return lambda func: trace(func, handle=handle)

    @functools.wraps(func)
    def inner(*args, **kwargs):
        print(func.__name__, args, kwargs)
        return func(*args, **kwargs)
    return inner


@trace(handle=sys.stdout)
def identity(x):
    "i do"
    return x


print(identity('23'), identity.__doc__)
