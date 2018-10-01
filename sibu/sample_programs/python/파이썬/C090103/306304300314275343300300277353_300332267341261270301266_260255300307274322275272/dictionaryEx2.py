# 딕셔너리에서 최소값/최대값/ 정렬

fruits = {
    "사과":300,
    "오렌지":200,
    "바나나":500,
    "배":1000,
    "포도":2000
}

#zip()함수를 이용해서 키(key)와 값(value)을 뒤집는다.
max_fruits = max(zip(fruits.values(), fruits.keys()))
print(max_fruits)

min_fruits = min(zip(fruits.values(),fruits.keys()))
print(min_fruits)

sorted_fruits = sorted(zip(fruits.values(), fruits.keys()))
print(sorted_fruits)

fruits_name = zip(fruits.values(), fruits.keys())
print(max(fruits_name))
fruits_name = zip(fruits.values(), fruits.keys())
print(min(fruits_name))

print(min(fruits)) #키를 비교하여 최소키를 리턴한다.
print(max(fruits))

print(min(fruits.values()))
print(max(fruits.values()))

#zip을 사용하지 않고 키와 값을 동시에 얻겠다.

print(min(fruits, key= lambda n: fruits[n])) #min의 key 함수를 적용한 예

print(max(fruits, key= lambda n: fruits[n]))

#최소 값과 일치하는 key를 얻어오고자 할 때
max_val = max(fruits, key= lambda n: fruits[n])

print(max_val)

fruits1 = {"사과":300, "오렌지":300}

#value가 동일값인 경우 key를 가지고 비교한다.
print(min(zip(fruits1.values(), fruits1.keys())))

