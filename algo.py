import functools


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


def pre(cond, message):
    def wrapper(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            assert cond(*args, **kwargs), message
            return func(*args, **kwargs)
        return inner
    return wrapper


isEmpty = pre(lambda self, *args: self.root is not None, 'tree is empty')


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

    def initial_node(self, p):
        return self.root if p is None else p

    @isEmpty
    def findmin(self, p=None):
        p = self.initial_node(p)
        return self.findmin(p.left) if p.left else p

    @isEmpty
    def removemin(self, p=None):
        p = self.initial_node(p)
        if p.left is None:
            return p.right
        p.left = self.removemin(p.left)
        return p.balance()

    @isEmpty
    def remove(self, key, p=None):
        p = self.initial_node(p)
        if p.left is None and p.right is None and p.key != key:
            return None
        if key < p.key:
            p.left = self.remove(key, p.left)
        elif key > p.key:
            p.right = self.remove(key, p.right)
        else:
            q = p.left
            r = p.right
            del p
            if not r:
                return q
            min_ = self.findmin(r)
            min_.right = self.removemin(r)
            min_.left = q
            return min_.balance


'''class RBNode:

    def __init__(self, key):
        super().__init__()
        self.key = key
        self.p = None
        self.left = None
        self.right = None
        self.color = 'BLACK'


class RBTree:

    def __init__(self):
        super().__init__()
        self.nil = RBNode(None)
        self.root = None

    def Left_Rotate(self, x):
        if x.right == self.nil:
            return

        y = x.right
        x.right = y.left

        if y.left != self.nil:
            y.left.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x

        x.p = y

    def Right_Rotate(self, y):
        if y.left == self.nil:
            return

        x = y.left
        y.left = x.right

        if x.right != self.nil:
            x.right.p = y
        x.p = y.p
        if y.p == self.nil:
            self.root = x
        elif y == y.p.right:
            y.p.right = x
        else:
            y.p.left = x
        x.right = y

        y.p = x

    def RB_Insert_Fixup(self, z):
        z.color = 'RED'
        while z is not self.root and z.p.color == 'RED':
            if z.p == z.p.p.left:
                y = z.p.p.right
                if y.color == 'RED':
                    z.p.color = 'BLACK'
                    y.color = 'BLACK'
                    z.p.p.color = 'RED'
                    z = z.p.p
                elif z == z.p.right:
                    z = z.p
                    self.Left_Rotate(z)

                    z.p.color = 'BLACK'
                    z.p.p.color = 'RED'
                    self.Right_Rotate(z.p.p)

            else:
                y = z.p.p.left
                if y.color == 'RED':
                    z.p.color = 'BLACK'
                    y.color = 'BLACK'
                    z.p.p.color = 'RED'
                    z = z.p.p
                elif z == z.p.left:
                    z = z.p
                    self.Right_Rotate(z)
                    z.p.color = 'BLACK'
                    z.p.p.color = 'RED'
                    self.Left_Rotate(z.p.p)

        self.root.color = 'BLACK'

    def RB_Insert(self, z: 'RBNode'):
        if self.root is None:
            self.root = z
            self.root.p = self.nil
            self.root.left = self.nil
            self.root.right = self.nil
            return
        y = self.nil
        x = self.root
        while x is not self.nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y

        if z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = 'RED'
        self.RB_Insert_Fixup(z)

rb = RBTree()
rb.RB_Insert(RBNode(12))

rb.RB_Insert(RBNode(11))
rb.RB_Insert(RBNode(98))
rb.RB_Insert(RBNode(3))
rb.RB_Insert(RBNode(12))
rb.RB_Insert(RBNode(12))
rb.RB_Insert(RBNode(12))
rb.RB_Insert(RBNode(12))
rb.RB_Insert(RBNode(12))
rb.RB_Insert(RBNode(12))
rb.RB_Insert(RBNode(12))
rb.RB_Insert(RBNode(12))
rb.RB_Insert(RBNode(12))
rb.RB_Insert(RBNode(12))
rb.RB_Insert(RBNode(12))'''


class NilNode(object):
    def __init__(self):
        self.red = False

    def children_count(self):
        return 0


NIL = NilNode()


class RBNode(object):
    def __init__(self, key):
        self.red = False
        self.parent = None
        self.key = key
        self.left = NIL
        self.right = NIL

    def __str__(self):
        return '{color} {key} Node'.format(
            color='RED' if self.red else 'BLACK', key=self.key)

    def __iter__(self):
        yield self.key
        if self.left != NIL:
            yield from self.left.__iter__()
        if self.right != NIL:
            yield from self.right.__iter__()

    def children_count(self) -> int:
        return sum((int(self.left != NIL), int(self.right != NIL)))

    def has_children(self) -> bool:
        return bool(self.children_count())


class RBTree(object):
    def __init__(self):
        self.root = None
        self.size = 0

    def __iter__(self):
        if not self.root:
            return list()
        yield from self.root.__iter__()

    def insert(self, key):
        self.size += 1
        new_node = RBNode(key)
        if self.root is None:
            new_node.red = False
            self.root = new_node
            return
        currentNode = self.root
        while currentNode != NIL:
            potentialParent = currentNode
            if new_node.key < currentNode.key:
                currentNode = currentNode.left
            else:
                currentNode = currentNode.right
        new_node.parent = potentialParent
        if new_node.key < new_node.parent.key:
            new_node.parent.left = new_node
        else:
            new_node.parent.right = new_node
        self.fix_tree_after_add(new_node)

    def contains(self, key, curr=None):
        if curr is None:
            curr = self.root
        while curr != NIL and key != curr.key:
            if key < curr.key:
                curr = curr.left
            else:
                curr = curr.right
        return curr

    def fix_tree_after_add(self, new_node):
        while new_node.parent.red == True and new_node != self.root:
            if new_node.parent == new_node.parent.parent.left:
                uncle = new_node.parent.parent.right
                if uncle.red:
                    new_node.parent.red = False
                    uncle.red = False
                    new_node.parent.parent.red = True
                    new_node = new_node.parent.parent
                else:
                    if new_node == new_node.parent.right:
                        new_node = new_node.parent
                        self.left_rotate(new_node)
                    new_node.parent.red = False
                    new_node.parent.parent.red = True
                    self.right_rotate(new_node.parent.parent)
            else:
                uncle = new_node.parent.parent.left
                if uncle.red:
                    new_node.parent.red = False
                    uncle.red = False
                    new_node.parent.parent.red = True
                    new_node = new_node.parent.parent
                else:
                    if new_node == new_node.parent.left:
                        new_node = new_node.parent
                        self.right_rotate(new_node)
                    new_node.parent.red = False
                    new_node.parent.parent.red = True
                    self.left_rotate(new_node.parent.parent)
        self.root.red = False

    def left_rotate(self, new_node):
        sibling = new_node.right
        new_node.right = sibling.left
        if sibling.left is not None:
            sibling.left.parent = new_node
        sibling.parent = new_node.parent
        if new_node.parent is None:
            self.root = sibling
        else:
            if new_node == new_node.parent.left:
                new_node.parent.left = sibling
            else:
                new_node.parent.right = sibling
        sibling.left = new_node
        new_node.parent = sibling

    def right_rotate(self, new_node):
        sibling = new_node.left
        new_node.right = sibling.right
        if sibling.right is not None:
            sibling.right.parent = new_node
        sibling.parent = new_node.parent
        if new_node.parent is None:
            self.root = sibling
        else:
            if new_node == new_node.parent.right:
                new_node.parent.right = sibling
            else:
                new_node.parent.left = sibling
        sibling.right = new_node
        new_node.parent = sibling

    def transplant(self, node, tr_node):
        if node.parent == NIL:
            self.root = tr_node
        if node.parent is not None:
            if node == node.parent.left:
                node.parent.left = tr_node
            else:
                node.parent.right = tr_node
            tr_node.parent = node.parent

    def tree_minimum(self, current_node):
        while current_node.left != NIL:
            current_node = current_node.left
        return current_node

    '''def _find_in_order_successor(self, node_to_remove):
        right_node = node_to_remove.right
        left_child_node = right_node.left
        if left_child_node == NIL:
            return right_node
        while left_child_node.left != NIL:
            left_child_node = left_child_node.left
        return left_child_node

    def delete(self, key):
        node_to_remove = self.contains(key)
        if node_to_remove  == NIL:
            return
        if node_to_remove.children_count() == 2:
            successor = self._find_in_order_successor(node_to_remove)
            node_to_remove.key = successor.key  # switch the value
            node_to_remove = successor
        self._remove(node_to_remove)
        self.size -= 1

    def _remove(self, node):
        left_child = node.left
        right_child = node.right
        not_nil_child = left_child if left_child != NIL else right_child
        if node == self.root:
            if not_nil_child != NIL:
                self.root = not_nil_child
                self.root.parent = None
                self.root.red = False
            else:
                self.root = None
        elif node.color:
            if not node.has_children():
                self._remove_leaf(node)
            else:
                raise Exception('Unexpected behavior')
        else:
            if right_child.has_children() or left_child.has_children():
                raise Exception('The red child of a black node with 0 or 1 children'
                                ' cannot have children, otherwise the black height of the tree becomes invalid! ')
            if not_nil_child.red:
                node.key = not_nil_child.key
                node.left = not_nil_child.left
                node.right = not_nil_child.right
            else:
                self._remove_black_node(node)'''

    '''def delete(self, del_node):
        if del_node == NIL:
            return
        elif del_node == self.root:
            if del_node.left:
                del_node.left = self.root
                self.right_rotate(del_node)
                self.delete_fixup(del_node.left)
                return
        tmp = del_node
        tmp_original_color = tmp.red
        if del_node.left == NIL:
            child = del_node.right
            self.transplant(del_node, del_node.right)
        elif del_node.right == NIL:
            child = del_node.left
            self.transplant(del_node, del_node.left)
        else:
            tmp = self.tree_minimum(del_node.right)
            tmp_original_color = tmp.red
            child = tmp.right
            if tmp.parent == del_node:
                child.parent = tmp
            else:
                self.transplant(tmp, tmp.right)
                tmp.right = del_node.right
                tmp.right.parent = tmp
            self.transplant(del_node, tmp)
            tmp.left = del_node.left
            tmp.left.parent = tmp
            tmp.red = del_node.red
        if not tmp_original_color:
            self.delete_fixup(child)

    def delete_fixup(self, child):
        while child != self.root and not child.red:
            if child == child.parent.left:
                brother = child.parent.right
                if brother.red:
                    brother.red = False
                    child.parent.red = True
                    self.left_rotate(child.parent)
                    brother = child.parent.right
                if brother != NIL:
                    if not brother.left.red and not brother.right.red:
                        brother.red = True
                        child = child.parent
                    elif not brother.right.red:
                        brother.left.red = False
                        brother.red = True
                        self.right_rotate(brother)
                        brother = child.parent.right
                    brother.red = child.parent.red
                    child.parent.red = False
                    brother.right.color = False
                    self.left_rotate(child.parent)
                child = self.root
            else:
                brother = child.parent.left
                if brother.red:
                    brother.red = False
                    child.parent.red = True
                    self.right_rotate(child.parent)
                    brother = child.parent.left
                if brother != NIL:
                    if not brother.right.red and not brother.left.red:
                        brother.red = True
                        child = child.parent
                    elif not brother.left.red:
                        brother.right.red = False
                        brother.red = True
                        self.left_rotate(brother)
                        brother = child.parent.left
                    brother.red = child.parent.red
                    child.parent.red = False
                    brother.left.color = False
                    self.right_rotate(child.parent)
                child = self.root'''


'''if __name__ == "__main__":
    tree = RBTree()
    tree.insert(1)
    tree.insert(2)
    tree.insert(4)
    tree.insert(5)
    tree.insert(7)

    #tree.delete(tree.contains(1))
    print(tree.contains(3))
    print(tree.root.key)'''


def memoized_cut_rod(prices, cut_num):
    new_prices = []
    _inf = float('-inf')
    for i in range(0, cut_num + 1):
        new_prices.append(_inf)
    return memoized_cut_rod_aux(prices, cut_num, new_prices)


def memoized_cut_rod_aux(prices, cut_nom, new_prices):
    if new_prices[cut_nom] >= 0:
        return new_prices[cut_nom]
    curr_price = 0 if cut_nom == 0 else float('-inf')
    for curr_nom in range(1, cut_nom + 1):
        curr_price = max(
            curr_nom,
            prices[curr_nom] +
            memoized_cut_rod_aux(
                prices,
                cut_nom -
                curr_nom,
                new_prices))
    new_prices[cut_nom] = curr_price
    return curr_price


def bottom_up_cut_rod(prices, cut_num):
    new_prices = [0]
    mInf = float('-inf')
    for j in range(1, cut_num + 1):
        curr_price = mInf
        for i in range(1, j + 1):
            curr_price = max(curr_price, prices[i] + new_prices[j - i])
        new_prices.append(curr_price)
    return new_prices[cut_num]


def extended_bottom_up_cut_rod(p, n):
    r = [-1 for _ in range(len(p))]
    s = [-1 for _ in range(len(p))]
    mInf = float('-inf')
    for j in range(1, n + 1):
        q = mInf
        for i in range(1, j + 1):
            if q < p[i] + r[j - i]:
                q = p[i] + r[j - i]
                s[j] = i
        r[j] = q
    return r, s


def print_cut_rod_solution(p, n):
    _, s = extended_bottom_up_cut_rod(p, n)
    while n > 0:
        print(s[n])
        n -= s[n]


def calculation_fibonacci(number):
    if number == 0:
        return 0
    elif number == 1:
        return 1
    else:
        return calculation_fibonacci(
            number - 1) + calculation_fibonacci(number - 2)


def memoized_fibonacci(number):
    mInf = float('-inf')
    storage = [mInf for _ in range(number + 1)]
    storage[0] = 0
    storage[1] = 1
    return memoized_fibonacci_aux(number, storage)


def memoized_fibonacci_aux(number, storage):
    if storage[number] >= 0:
        return storage[number]
    else:
        i = 2
        while i < number:
            storage[i] = storage[i - 1] + storage[i - 2]
            i += 1
    return storage[number - 1] + storage[number - 2]


def bottom_up_fibonacci(number):
    storage = [0 for _ in range(number + 1)]
    storage[0] = 0
    storage[1] = 1

    for j in range(2, number + 1):
        storage[j] = storage[j - 1] + storage[j - 2]
    return storage[number]


def lcs_length(X, Y):
    m = len(X)
    n = len(Y)
    b = [[0 for _ in range(0, n)] for _ in range(0, m)]
    c = [[0 for _ in range(0, n + 1)] for _ in range(0, m + 1)]
    for i in range(1, m):
        for j in range(1, n):
            if X[i] == Y[j]:
                c[i][j] = c[i - 1][j - 1] + 1
                b[i][j] = '/'
            elif c[i - 1][j] >= c[i][j - 1]:
                c[i][j] = c[i - 1][j]
                b[i][j] = '|'
            else:
                c[i][j] = c[i][j - 1]
                b[i][j] = '_'
    return c, b


def print_lcs(b, X, i, j):
    if i == 0 or j == 0:
        return
    if b[i][j] == '/':
        print_lcs(b, X, i - 1, j - 1)
        print(X[i])
    elif b[i][j] == '|':
        print_lcs(b, X, i - 1, j)
    else:
        print_lcs(b, X, i, j - 1)


'''c, b = lcs_length('sfsdf', 'sdefdsfds')
X, Y = 'sfsdf', 'sdefdsfds'
print_lcs(b, X, len(X) - 1, len(Y) - 1)'''


def optimal_bst(p, q, n):
    e = [[0 for _ in range(0, n + 1)] for _ in range(1, n + 2)]
    w = [[0 for _ in range(0, n + 1)] for _ in range(1, n + 2)]
    root = [[0 for _ in range(1, n + 1)] for _ in range(1, n + 1)]
    inf = float('inf')

    for i in range(1, n + 2):
        e[i][i - 1] = q[i - 1]
        w[i][i - 1] = q[i - 1]
    for l in range(1, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            e[i][j] = inf
            w[i][j] = w[i][j - 1] + p[j] + q[j]
            for r in range(i, j + 1):
                t = e[i][r - 1] + e[r + 1][j] + w[i][j]
                if t < e[i][j]:
                    e[i][j] = t
                    root[i][j] = r
    return e, root


def inc_seq(items):
    prev_el = items[0]
    tmp = [prev_el]
    max_inc_seq = tmp
    for curr_el in items[1:]:
        if prev_el <= curr_el:
            tmp.append(curr_el)
        else:
            tmp = [curr_el]
        if len(tmp) > len(max_inc_seq):
            max_inc_seq = tmp
        prev_el = curr_el
    return max_inc_seq


def minimal_path_field(size_x, size_y, A: 'arr[size_y - 1][size_x - 1]'):
    inf = float('inf')
    W = [[inf for _ in range(0, size_x)] for _ in range(0, size_y)]
    W[0][0] = A[0][0]
    W[1][0] = W[0][0] + A[1][0]
    W[0][1] = W[0][0] + A[0][1]

    for i in range(1, size_y):
        for j in range(1, size_x):
            W[i][j] = min(W[i - 1][j], W[i][j - 1], W[i - 1][j - 1]) + A[i][j]

    return W[size_y - 1][size_x - 1]

def rec_fib(num):
    storage = [-1 for _ in range(0, num + 1)]
    return rec_fib_aux(num, storage)
def rec_fib_aux(num, storage):
    if storage[num] >= 0:
        return storage[num]
    if num == 0:
        storage[num] = 0
        return 0
    elif num == 1:
        storage[num] = 1
        return 1
    else:
        storage[num] = rec_fib_aux(num - 1, storage) + rec_fib_aux(num - 2, storage)
        return storage[num]

def backpack_task(cost_things, weight_things, max_weight):
    len_ = len(cost_things)
    shift_ = 1
    tmp = []
    things = []
    price = 0
    max_price = 0

    for i in range(len_):
        for j in range(i + shift_, len_):
            price = cost_things[i]
            weight = weight_things[i]
            tmp = [i]
            while weight <= max_weight or j != len_ - 1:
                price += cost_things[j]
                weight += weight_things[j]
                tmp.append(j)
            if max_price < price:
                max_price = price
                things = tmp
            shift_ += 1
        shift_ = 1
    return max_price, things

class NodePQ:

    def __init__(self, freq=None, left=None, right=None, parent=None):
        super().__init__()
        self.freq = freq
        self.left = left
        self.right = right
        self.parent = parent


class PrioritQueue:

    def __init__(self):
        super().__init__()
        self.nill = NodePQ()
        self.root = None

    def insert(self, freq):
        new_node = NodePQ(freq, self.nill, self.nill)
        if self.root is None:
            new_node.parent = self.nill
            self.root = new_node
            return
        current_node = self.root
        parent_node = current_node
        while current_node != self.nill:
            parent_node = current_node
            if current_node.freq > new_node.freq:
                current_node = current_node.left
            else:
                current_node = current_node.right
        if parent_node.freq < new_node.freq:
            parent_node.right = new_node
        else:
            parent_node.left = new_node
        new_node.parent = parent_node

    def node_insert(self, new_node):
        if self.root is None:
            new_node.parent = self.nill
            self.root = new_node
            return
        current_node = self.root
        parent_node = current_node
        while current_node != self.nill:
            parent_node = current_node
            if current_node.freq > new_node.freq:
                current_node = current_node.left
            else:
                current_node = current_node.right
        if parent_node.freq < new_node.freq:
            parent_node.right = new_node
        else:
            parent_node.left = new_node
        new_node.parent = parent_node


    def transplant(self, node, node_tr):
        if node.parent == self.nill:
            self.root = node_tr
        elif node == node.parent.left:
            node.parent.left = node_tr
        else:
            node.parent.right = node_tr
        node_tr.parent = node.parent

    def minimum_list_curr_node(self, curr_node:NodePQ):
        while curr_node.left != self.nill:
            curr_node = curr_node.left
        return curr_node

    def delete(self, del_node:NodePQ):
        if del_node.left == self.nill:
            self.transplant(del_node, del_node.right)
        elif del_node.right == self.nill:
            self.transplant(del_node, del_node.left)
        else:
            min_list_del_node = self.minimum_list_curr_node(del_node.right)
            if min_list_del_node.parent != del_node:
                self.transplant(min_list_del_node, min_list_del_node.right)
                min_list_del_node.right = del_node.right
                min_list_del_node.right.parent = min_list_del_node
            self.transplant(del_node, min_list_del_node)
            min_list_del_node.left = del_node.left
            min_list_del_node.left.parent = min_list_del_node

    def extract_min(self):
        return_node = self.minimum_list_curr_node(self.root)
        self.delete(return_node)
        return return_node

def createPQ_from_iterable(iterable):
    PQ = PrioritQueue()
    for el in iterable:
        PQ.insert(el)
    return PQ

def huffman(iterable):
    n = len(iterable)
    Q = createPQ_from_iterable(iterable)
    for _ in range(0, n):
        z = NodePQ()
        x = Q.extract_min()
        y = Q.extract_min()
        z.left = x
        z.right = y
        z.freq = x.freq + y.freq
        Q.insert(z.freq)
    return Q.extract_min()
