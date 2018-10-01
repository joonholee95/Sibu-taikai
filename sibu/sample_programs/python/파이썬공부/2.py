# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:40:33 2017

@author: 준호
"""

X=11
Y=12


if X == Y:
    print('hello')
else:
    print('no')

if X == Y:
    print('hello')
else:
    print('no')

    
print(10,10)
print('출력1','출력2')
print("출력1",end='  ')
print('출력2')

x,y=100,200
print(x,y)
y,x=x,y
print(x,y)

v1,*v2=[1,2,3]
print(v1,v2)

*v1,v2=[1,2,3]
print(v1,v2)

print(format(1234567,'3,d'))

"양식문자"
z=300
print('%d +%d = %d' %(x,y,z))
print('%s입니다.점수는%d입니다.' %('임',20))
print('전체%d%%가 %d입니다' %(30,170))

"외부상수"
print('name:{},age:{}'.format('asd',20))
print('name:{0},age:{1}'.format('asd',20))
print('name:{1},age:{0}'.format('asd',20))

str2="이순신"
result=str2[0];print(result)
result=str2[0:3];print(result)
result=str2[:3];print(result)
