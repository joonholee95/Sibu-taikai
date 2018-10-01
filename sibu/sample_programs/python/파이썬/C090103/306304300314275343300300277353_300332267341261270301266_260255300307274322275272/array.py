# pprint(pretty printer) : 자료 구조를 사람이 보기 좋게 출력하는 모듈

data = [(1, {"a":"가", "b":"나", "c":"다", "d":"라"}),
        (2, {"e":"마", "f":"바", "g":"사", "h":"아"})
        ]

# pprint모듈에 pprint() 함수를 이용하여 자료구조를 출력해보기

print(data)

from pprint import pprint

pprint(data)


# array : 시퀀스 자료구조를 정의하는데, list와의 차이점은 모든 자료형이 동일하다.

import array

str = "aabcdefgh"

arr =array.array("u", str)  #array(타입코드, 값)

print(arr)

print(array.typecodes)

arr1 = array.array('i', range(5))

print(arr1)

arr1.extend(range(5))

print(arr1)

print(arr1[3:6])

print(list(enumerate(arr1)))


