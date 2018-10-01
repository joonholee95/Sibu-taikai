# 슬라이스에 name 설정하기


aa = [ 1,2,5,11,3,6,7,10]

saa = aa[3:6]

record = "김말똥2419911123서울시"

code ="2011 2014 2015 1999 1981"
print(len(code))

birth_year = slice(5,9)
name = slice(10,50,2)

print(record[birth_year])
print(record[5:9])

print(record[name])
print(record[0:3])

scode = slice(0,10)
print(code[scode])
scode = slice(0,10,2)
print(code[scode])


# indices(len)

record1 = "고길동1501012341234서울"

print(len(record1))

print(name.indices(len(record1)))
