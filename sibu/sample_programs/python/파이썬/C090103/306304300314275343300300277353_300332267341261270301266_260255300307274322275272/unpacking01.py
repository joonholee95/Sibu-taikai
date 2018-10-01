# 언패킹(unpacking)

pa = (1,2)

a, b = pa

# a,b,c = pa

print(a)
print(b)


li_data = ["홍길동", 23, "서울",(1980, 11, 21)]
"""
name, age, local, birthday = li_data

print(name)
print(age)
print(local)
print(birthday)

name, age, local, (year, month, day) = li_data

print(year)
print(month)
print(day)

str = "Hello"

a, b, c, d, e = str

print(a)

print(e)
"""

# 특정 값을 무시하거나 *를 이용하여 여러개를 언패킹하기

name, _, local, _ =li_data

print(name)
print(local)

person_info = ("장길산", "jks@naver.com", '010-1234-1234', '02-212-4565')

name, email, *phone = person_info

print(name)
print(email)
print(phone)

pointValue = [10, 5, 12,11,22,14, 12, 15, 10, 10, 15, 14, 15]

*prePoint, curPoint = pointValue

print(prePoint)
print(curPoint)

address = [("우", 234, 123), ("도", "서울"),("도", "경기"), ("우", 123, 234)]

def show_zipcode(num1, num2):
    print("우", num1, num2)

def show_local(str):
    print("도", str)


for key, *arg in address:
    if key == "우":
        show_zipcode(*arg)
    elif key == "도":
        show_local(*arg)


str2 = "홍길동/23/112121212121212121/812541-12545/010-1235-1235/서울"

name, age,*num,local =str2.split("/")

print(name)
print(age)
print(local)


li_data = ["홍길동", 23, "서울",(1980, 11, 21)]

name, *_,(year,*_)=li_data

print(name)
print(year)


