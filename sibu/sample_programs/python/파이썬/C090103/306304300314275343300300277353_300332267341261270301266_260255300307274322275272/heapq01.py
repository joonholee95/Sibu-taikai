# 힙 정렬 알고리즘
"""
Heap 이란 자식노드가 부모노드와 정렬관계를 가지는 트리형 자료 구조

이진 힙의 경우 array나 list를 사용해서 표현할 수 있다. (인덱스를 이용해서 표시할 수 있다.)
이때 수식은 n(인덱스)을 부모로하는 자식 노드의 위치 수식은 2*n+1, 2*n+2

Heap은 최대 힙(max-heap: 부모가 자식보다 크거나 같다),
최소 힙(min-heap: 부모가 자식과 같거나 작다.)이 있다.

파이썬의 heapq모듈은 최소힙(min-heap)으로 구현된 모듈이다.


"""
import heapq
# 힙 생성 : heappush(), heapify()
from showHeap import show_tree
from heapData import data

heap = []

for n in data:
    print ("%3d push :" %n)
    heapq.heappush(heap, n)
    show_tree(heap)



