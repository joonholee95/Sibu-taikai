#python ak.py
# -*- coding: euc-kr -*-

def myfunc(x):
    assert type(x) == int, "숫자만 입력하세요"
    return x * 100

rslt = myfunc('가')
print(rslt)