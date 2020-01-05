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


'''db = SentielDoubleLinkedList()
db.list_insert(List(key=125))
db.list_insert(List(key=124))
print('sDl', db.__str__())
db.list_delete(1)
print('sDl', db.__str__())'''
