# comprehension : List comprehension, Dictionary comprehension, Set comprehension

# List comprehension : 리스트의 각각의 아이템(원소, 요소)에 어떤 함수를 적용한 후에
#                      그 결과를 받아 새로운 리스트로 만들어 주는 기능

aa_li = [2, 4, 1, 5]

aa_com = [item*3 for item in aa_li]
"""
[item*3 for item in aa_li]

aa_li는 어떤 함수를 적용하려는 대상(리스트)

for ~ 문은 현재 이 리스트 컴프리헨션이 처리하는 방식을 의미
    :aa_li 내부를 순회하면서 각 아이템을 임시변수(item)에 저장

이 저장된 임시변수를 대상으로 함수(item*3)를 적용한 후 그결과를
리스트에 추가(append)한다.

리스트 컴프리헨션의 결과 값으로 새로운 리스트를 반환한다.
원래 있던 aa_li는 변경하지 않는다.

원하는 아이템을 뽑아내고자 할 경우에 리스트 컴프리헨션 뒤에 if 문을 사용할 수 있다.

"""
print(aa_com)

# 딕셔너리 컴프리헨션

# 컴프리헨션을 이용해서 딕셔너리의 value와 key값을 바꾸기
aa_dict = {"홍길동":100, "강호동":200, "유재석":300}

aa_dict_com = {value:key for key, value in aa_dict.items()}

print(aa_dict_com)


#set 컴프리헨션(comprehension)

aa_set = set(range(10))

print(aa_set)
aa_set_com = {i**2 for i in aa_set}

print(aa_set_com)

aa_set_com1 = {i for i in aa_set if i%2 ==0}
print(aa_set_com1)

# 꼭 대상이 set일 필요는 없다. 시퀀스라면 가능하다.
aa_set_com2 = {2**i for i in range(10)}

print(aa_set_com2)

a = dict(one =1, two=2, three=3)
print(a)