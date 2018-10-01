# Deque : 양방향 큐(데크)는 컨테이너 양쪽 (앞뒤)에 아이템을 넣거나 뺄 수 있다.

import collections

deq = collections.deque("Hello python")

print (deq)
print(len(deq))
print(deq[0])
print(deq[-1])

deq.remove('o')
print(deq)


deq.append('k')
print(deq)

deq.appendleft('k')
print(deq)

deq.extendleft('d')
print(deq)

deq1 = collections.deque()
deq1.extend("가나다라마바사")

print(deq1)

deq1.append('자')

print(deq1)

deq1.extendleft("사")

print(deq1)

# 아이템 꺼내기
"""
print("오른쪽에서 부터 꺼내오기")
while True:
    try:
        print (deq1.pop(), end=' ')
    except IndexError:
        break
print()

"""
print("왼쪽에서 부터 꺼내오기")
while True:
    try:
        print (deq1.popleft(), end=' ')
    except IndexError:
        break

