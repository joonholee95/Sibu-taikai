# 제너레이터 (generator / yield )
# yield의 리턴값이 generator object
# yield는 generator 생성을 하고 generator는 next()함수를 가지고 있다.

def generatorEx(n):
    for i in range(n):
        yield i ** 2

print(generatorEx(4))

gen = generatorEx(4)

print(gen)

print(next(gen))

print(next(gen))

print(next(gen))

print(next(gen))

gen = generatorEx(3)

print(next(gen))


def countdown(n):
    while n > 0:
        yield n
        n -=1
    print ("end")

cnt = countdown(3)

print(cnt)

print(next(cnt))

print(next(cnt))

print(next(cnt))


for i in generatorEx(5):
    print (i)
