#python ak.py
# -*- coding: euc-kr -*-

def myfunc(x):
    assert type(x) == int, "���ڸ� �Է��ϼ���"
    return x * 100

rslt = myfunc('��')
print(rslt)