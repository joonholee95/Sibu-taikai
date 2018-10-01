

def map_test(li):
	res = []
	for i in li:
		res.append(i*3)
	return res


res = map_test([10,20,30,40])

print(res)


"""
lambda : 함수를 생성할 때 사용하는 예약어 def동일한 기능의 예약어이다.
			보통 한줄로 간결하게 함수를 만들어 사용할 때 사용한다.
            def를 사용할 수 없는 곳에서 사용한다.

사용 형식
	lambda 인수1, 인수2,.... : 인수를 이용한 표현식
"""

def positive(li):
	res = []
	for i in li:
		if i > 0:
			res.append(i)
	return res

print(positive([1,-12,3,0, -3, 7]))


