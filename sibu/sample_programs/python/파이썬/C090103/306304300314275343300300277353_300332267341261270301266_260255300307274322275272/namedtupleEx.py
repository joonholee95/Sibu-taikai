#collections.namedtuple():튜플 객체에 이름을 설정하는 함수

from collections import namedtuple

Member = namedtuple('Member', ['email', 'date'])

member1 = Member('asdf@naver.com', '2014-10-04')

print(member1)

print(member1.email)
print(member1.date)

print(len(member1))

email, date = member1

print(email, date)


stock_rec = [
    ("삼성", 10, 100),
    ("현대", 10, 90),
    ("기아", 10, 80)
    ]

def cal_stock(stock):
    tot  = 0
    for n in stock:
        tot += n[1]*n[2]
    return tot

print(cal_stock(stock_rec))


Stock = namedtuple('Stock', ['name', 'amount', 'price'])

def cal_stock(stock):
    tot  = 0
    for n in stock:
        s = Stock(*n)
        tot += s.amount* s.price
    return tot


print (cal_stock(stock_rec))


n1 = ["삼성", 10, 100]
print(*n1)

# namedtuple은 수정할 수 없다

stock2 = Stock('기아차', 100, 500)

print(stock2)

print(stock2.amount)

# stock2.amount = 120 : AttributeError 발생

# namedtuple을 수정하기 위해서는 _replace() 메소드를 사용해야한다.

stock2=stock2._replace(amount=120)

print(stock2)

# _replace()를 활용해서 prototype 튜플을 만들 수 있다.

Stock3 = namedtuple('Stock3', ['name', 'amount', 'price', 'date', 'time'])

# 프로토타입 인스턴스 생성
prototype_stock = Stock3('', 0, 0, None, None)

# 딕셔너리를 Stock3튜플로 전환하는 함수 만들기

def dict_to_stock(dic):
    return prototype_stock._replace(**dic)

aa = {'name':'삼성', 'amount':200, 'price':100}

aa_stock = dict_to_stock(aa)

print(aa_stock)

# 파라미터 앞에 *, ** 의 의미

# *args : 파라미터를 몇개를 사용할 지 모르는 경우
# **args: 딕셔너리 형태의 파라미터의 개수를 모르는 경우

def show_param(*s):
    print (s)
    for p in s:
        print (p)

show_param('a', 21, 1)

def show_param2(**dic):
    print(dic)
    print(dic.keys())
    print(dic.values())

    for key, value in dic.items():
        print ("%s : %s" %(key, value))


show_param2(one=1, two=2, three=3, four=4)

def show_param3(*s, **ss):
    print(s)
    print(ss)

show_param3("a", "b")

show_param3(one=1, two=2)

show_param3("a", "b", one=1, two=2)


ss2 = {'c':1, "d":2, 'e':3, 's':5}

show_param2(**ss2)




