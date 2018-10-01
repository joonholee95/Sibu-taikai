# ChainMap 클래스 : 여러개의 딕셔너리(매핑데이터)가 있을 때 하나의 딕셔너리로 합쳐서 검색을
#                  할 때 사용한다.
#                  첫번재 매핑데이터에서 검색 한 후에 그다음 매핑데이터에서 검색을 한다.
#                  중복 키가 있으면 첫번째 값을 사용한다.

aa = {"name":"홍길동", 'id':'test', 'email':'abab@naver.com'}
bb = {'order':'김말똥','tel':'010-1234-1234','email':'cccc@naver.com'}

from collections import ChainMap
chain = ChainMap(aa, bb)
print(chain['order'])
print(chain['name'])
print(chain['email'])

print(len(chain))

print(list(chain.keys()))

print(list(chain.values()))


chain['email'] = 'test123@naver.com'
chain['order'] = '강길동'

print(list(chain.values()))

del chain['name']
print(list(chain.values()))

print(aa)
print(bb)
chain['pw'] = 1234
print(aa)

# del chain['tel'] 매핑값을 추가하거나 삭제할 경우 항상 첫번째 매핑데이터에만 영향을 준다.

# ChainMap과 비슷한 기능을 하는 update() 함수가 있다.
aa = {"name":"홍길동", 'id':'test', 'email':'abab@naver.com'}
bb = {'order':'김말똥','tel':'010-1234-1234','email':'cccc@naver.com'}
merge = dict(bb)
merge.update(aa)

print(merge['name'])

print(merge['order'])
