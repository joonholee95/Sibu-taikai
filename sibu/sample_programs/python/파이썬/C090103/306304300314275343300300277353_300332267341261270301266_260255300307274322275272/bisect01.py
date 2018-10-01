# bisect 모듈 : 정렬된 상태로 아이템(데이타)을 추가할 수 있는 모듈
#              데이타가 많은 리스트를 사용할 경우 힙방식보다 시간과 메모리 낭비를 줄일 수 있다.

import bisect
import random

random.seed(1)

"""for i in range(5):
    print ('%5.4f' %random.random(), end=' ') """

print('New Index  List')
print('=== =====  ====')

li = []
for i in range(1,15):
    num = random.randint(1,100)

    pos = bisect.bisect_left(li, num) #아이템이 추가 되었을 때 인덱스값 리턴
    bisect.insort_left(li, num) #리스트를 정렬 상태로 유지시키는 함수
    print('%3d %3d  ' %(num, pos), li)
    # print (num, end=" ")