# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:14:23 2017

@author: 준호
"""
"사칙연산부터"
print(10+5)
print(10-5)
print(10*5)
print(10/5)

"import 패키지 사용"
import math
"올림"
print(math.ceil(2.2))
"내림"
print(math.floor(2.2))
"자승"
print(math.pow(2,5))
"pie"
print(math.pi)

"문자열"
print('hello')
print('"hello"')

"따음표 2쌍 쓸때는 다른걸로"
print("hello'world'")
print('hello"world"')

"따음표 안에서 스패이스바 눌러줘야 띄어써짐"
print('hello'+'world')
print('hello' + 'world')
print('hello ' + 'world')
print('hello'*3)

"R과는 다르게 0부터 시작"
print('hello'[0])

print("hello world".capitalize())
print("hello world".upper())
print(len("hello world"))
print("hello world".replace("world","yama!"))

"특수문자"
print("ego's\"dad\"")
print('\\')
print('hello\nworld')
print('hello\tworld')
print('\a')

'주의'
print(10+5)
print('10'+'5')

