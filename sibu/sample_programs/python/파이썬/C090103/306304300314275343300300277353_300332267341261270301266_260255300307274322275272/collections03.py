# namedtuple()

aa = ("홍길동", 24, "남")

print(aa)

bb = ("강복녀", 21, "여")
print(bb[0])

for n in [aa, bb]:
    print('%s은(는) %d 세의 %s성 입니다.' %n)


import collections

Person = collections.namedtuple("Person", 'name age gender')

aa = Person(name="강길동", age = "25", gender="남")

bb = Person(name="강길녀", age="21", gender="여")

for n in [aa, bb]:
    print("%s는(은) %s세의 %s성 입니다." %n)


# OrderedDict : 자료의 순서를 기억하는 사전형 클래스

dic = {}

dic["서울"] = "LG트윈스"
dic["대구"] = "삼성라이온즈"
dic["광주"] = "KIA 타이거즈"

for i, j in dic.items():
    print (i, j)

print("------------------")
dic1 = collections.OrderedDict()

dic1["서울"] = "LG트윈스"
dic1["대구"] = "삼성라이온즈"
dic1["광주"] = "KIA 타이거즈"

for i, j in dic1.items():
    print (i, j)

print ("비교를 이용한 표준 사전과 OrderedDict의 차이점")

dic3 = {}

dic3["서울"] = "LG트윈스"
dic3["대구"] = "삼성라이온즈"
dic3["광주"] = "KIA 타이거즈"

dic4 = {}

dic4["서울"] = "LG트윈스"
dic4["광주"] = "KIA 타이거즈"
dic4["대구"] = "삼성라이온즈"

print (dic3 == dic4)


dic5 = collections.OrderedDict()

dic5["서울"] = "LG트윈스"
dic5["대구"] = "삼성라이온즈"
dic5["광주"] = "KIA 타이거즈"

dic6 = collections.OrderedDict()
dic6["서울"] = "LG트윈스"
dic6["광주"] = "KIA 타이거즈"
dic6["대구"] = "삼성라이온즈"

print(dic5==dic6)


