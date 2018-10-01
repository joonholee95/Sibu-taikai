# 딕셔너리 키를 여러값에 매핑하기 (collections.defaultdict)

d = {}

d.setdefault('sel', []).append('02')
d.setdefault('sel', []).append('서울')

print(d)

from collections import defaultdict

d = defaultdict(list)

d['sel'].append('02')
d['sel'].append('서울')

d = defaultdict(set)
d['incheon'].add('032')
d['incheon'].add('인천')

print(d)

color = [('파랑', 3), ('노랑', 2), ('빨강',1),('파랑',4), ('노랑', 5)]

d = defaultdict(list)
for key, val in color:
    d[key].append(val)


li = list(d.items())

print (li)

d = {}
for key, val in color:
    d.setdefault(key, []).append(val)

li = list(d.items())

print(li)

str = "hello hi goodmorning"

d = defaultdict(int)
for key in str:
    d[key] +=1

li = list(d.items())

print(li)




