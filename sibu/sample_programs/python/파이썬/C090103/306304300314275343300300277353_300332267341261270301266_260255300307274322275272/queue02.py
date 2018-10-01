# 우선 순위 큐 구현하기 (우선 순위에 따라서 아이템을 정렬하고,
#                       우선 순위가 가장 높은 아이템을 팝하는 큐를 의미)
# queue.PriorityQueue 클래스를 이용하여 생성 가능(thread 개념을 학습 후 사용하도록 하자)

# heapq 모듈을 응용해서 우선순위를 큐를 구현한다.

import heapq

class PriorityQueue:
    def __init__(self):
        self.list = []
        self.idx = 0 # 입력되는 순서를 나타내느 인덱스

    def put (self, item, priority):
        heapq.heappush(self.list, (priority, self.idx, item))
        self.idx +=1

    def pop(self):
        return heapq.heappop(self.list)[-1]

class Item:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return "Item({!r})".format(self.name)
        #!r은 repr()호출하는 것과 같다. !s은 str()호출하는 것과 같다. !a 은 ascii로 변환


pQ = PriorityQueue()

pQ.put(Item("임꺽정"), 3)
pQ.put(Item("홍길동"), 1)
pQ.put(Item("장길산"), 2)
pQ.put(Item("일지매"), 3)


print(pQ.list)

print(pQ.pop())
print(pQ.pop())
print(pQ.pop())
print(pQ.pop())