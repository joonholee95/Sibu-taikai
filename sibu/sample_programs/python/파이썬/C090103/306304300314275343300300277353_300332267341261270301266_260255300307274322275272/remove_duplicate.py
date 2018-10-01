# 시퀀스의 중복 없애기

aa = [ 8,1,9,1,5,6,3,1,5,2]
bb = set(aa)
print(bb)


# 시퀀스의 순서를 유지하면서 중복 없애기

def remove_dup(items):
    set_a = set()
    for item in items:
        if item not in set_a:
            yield item
            set_a.add(item)

remove_dup(aa)

print(list(remove_dup(aa)))


