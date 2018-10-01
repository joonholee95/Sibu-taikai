# array의 내용을 파일에 쓰거나 읽기

import array

import binascii

arr = array.array('i', range(5))

print(arr)

f = open("test.txt", 'w+b')
arr.tofile(f)# tofile(파일객체) : 파일객체에 쓰는 함수
f.flush()

with open("test.txt", 'rb') as f1:
    data = f1.read()
    """
    print(binascii.hexlify(data))"""

    f1.seek(0)
    arr2 = array.array('i')
    arr2.fromfile(f1, len(arr))

    print(arr2)





