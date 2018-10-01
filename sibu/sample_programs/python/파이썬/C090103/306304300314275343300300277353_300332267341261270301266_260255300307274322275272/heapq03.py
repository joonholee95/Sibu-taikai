#heap 데이타 접근 하기: heappop()을 이용하여 가장 작은 값을 하나씩 끄집어 낸다.

import heapq
from heapData import data
from showHeap import show_tree

"""
힙 데이타 삭제

heapq.heapify(data)
show_tree(data)

print()

for n in range(3):
    min_val = heapq.heappop(data)
    show_tree(data)
    print(min_val)

힙 데이타 수정 heapreplace

heapq.heapify(data)
show_tree(data)

for n in [3, 15]:
    min_val = heapq.heapreplace(data, n)
    print(min_val)
    show_tree(data)
힙의 최대/최소 값 구하기 : nlargest(), nsamllest()
"""

print(heapq.nlargest(1,data))

print(heapq.nsmallest(3,data))