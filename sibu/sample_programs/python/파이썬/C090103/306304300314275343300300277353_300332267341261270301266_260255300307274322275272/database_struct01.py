# 데이터베이스 자료를 리스트로 저장 후 정렬 시키는 방법

records = [
    {'id':'test', 'pw':'1234', 'name':'홍길동', 'hp':'010-1234-1234'},
    {'id':'test', 'pw':'1234', 'name':'강호동', 'hp':'010-1978-1234'},
    {'id':'test3', 'pw':'1234', 'name':'이경규', 'hp':'010-1334-1234'},
    {'id':'afdf2', 'pw':'1234', 'name':'이문세', 'hp':'010-1245-1234'},
    {'id':'kkine', 'pw':'1234', 'name':'이수근', 'hp':'010-1248-1234'}
]

# 모듈 operator.itemgetter

from operator import itemgetter
from pprint import pprint

rec_by_name = sorted(records, key=itemgetter('name'))

rec_by_id = sorted(records, key=itemgetter('id'))


pprint(rec_by_name)

rec_by_id_tel = sorted(records, key=itemgetter('id', 'hp'), reverse=True)

pprint(rec_by_id_tel)




