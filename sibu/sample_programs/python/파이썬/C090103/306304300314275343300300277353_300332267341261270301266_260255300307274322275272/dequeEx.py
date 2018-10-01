# deque 자료구조 응용
# deque를 이용해서 고정크기의 큐를 생성하기(maxlen = n)

from collections import deque

dQ = deque(maxlen = 4)# 4개의 아이템을 갖는 큐를 생성

dQ.append('aa')

dQ.append('bb')

dQ.append('cc')
dQ.append('dd')

print(dQ)

dQ.append('ee') #새로운 아이템 ee가 추가되면서 aa아이템은 자동으로 삭제
print(dQ)

def search_word(lines, find_word, history):
    previous_lines = deque(maxlen=history)
    for readline in lines:
        if find_word in readline:
            yield readline, previous_lines
        previous_lines.append(readline)

"""
with open('someText.txt') as f:
    fword = search_word(f, "좋은아침", 4)
    print(fword)
    print(next(fword))

    print(next(fword))

    print(next(fword))
"""

if __name__ =='__main__':
    with open('someText.txt') as f:
        for fline, prevTexts in search_word(f,"좋은아침",4):
            for preline in prevTexts:
                print(preline)
            print(fline)
            print("="*10)



