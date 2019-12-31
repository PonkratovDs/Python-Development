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


A = [24, -1, 30, -2]
print(FindMaximumSubarrray(A, 0, len(A) - 1))


class heap:
    def __init__(self, A):
        self._items = A

    def heapSize(self):
        return len(self._items)

    def Left(self, i):
        return self._items[2 * i + 1]

    def Right(self, i):
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
                print(self.Left(k))
            else:
                print(self.Left(k), self.Right(k))

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


h = heap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
h.heapSort()

'''import heapq

x = [4,1,3,2,16,9,10,14,8,7]
heapq.heapify(x)
print(heapq.heappop(x), heapq.heappop(x),end=' mm ')'''


class priorityQueue():

    def __init__(self, heap:'heap class'):
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
        parIdx = (i-1) // 2
        while i > 0 and self.heap.heapPop(parIdx) < self.heap.heapPop(i):
            tmp = self.heap.heapPop(i)
            self.heap.changeEleminHeap(i, self.heap.heapPop(parIdx))
            self.heap.changeEleminHeap(parIdx, tmp)
            i = parIdx

p = priorityQueue(h)
h.__str__()

print('max', p.heapExtractMax())

