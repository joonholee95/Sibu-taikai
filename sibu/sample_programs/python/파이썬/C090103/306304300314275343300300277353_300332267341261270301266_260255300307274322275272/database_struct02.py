# 데이터베이스 group by 절과 같은 기능 수행하기

# itertools.groupby() 를 이용하여 필드에 따라 레코드를 묶어주기

records =[
    {"addr":"서울 강남구", "order":'01-1000'},
    {"addr":"서울 강서구", "order":'02-1000'},
    {"addr":"서울 강동구", "order":'03-3000'},
    {"addr":"서울 강북구", "order":'02-1000'},
    {"addr":"서울 강남구", "order":'02-1000'},
    {"addr":"서울 강서구", "order":'01-4000'},
    {"addr":"서울 강남구", "order":'03-1000'},
    {"addr":"서울 강북구", "order":'01-1000'},
    {"addr":"서울 강동구", "order":'02-2000'}
]

from operator import itemgetter
from itertools import groupby
from pprint import pprint
records.sort(key=itemgetter('addr'))

pprint(records)

for addr, items in groupby(records, key=itemgetter('addr')):
    print(addr)
    for item in items:
        print("___", item)
