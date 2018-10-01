
# coding: utf-8

# ## NumPy

# ### 다차원 배열 객체 : ndarray

# In[1]:

import numpy as np


# In[2]:

data = np.random.randn(2, 3)
data


# In[3]:

data * 2


# In[4]:

data - data


# In[5]:

data + data


# In[6]:

data.shape, data.dtype, data.ndim


# ### 다양한 방법으로 ndarray 만들기

# In[7]:

something = [1, 2, 2.7, 3.14, 16]
a = np.array(something)
a


# In[8]:

something2 = [[2, 5, 10], [3, 6, 9]]
b = np.array(something2)
b


# In[9]:

b.shape


# In[10]:

a.dtype, b.dtype


# In[11]:

np.ones(5)


# In[12]:

np.zeros((2, 3, 5))


# In[13]:

np.empty((4, 2))


# In[14]:

np.eye(3)


# In[15]:

np.arange(5)


# In[16]:

np.arange(10, 20)


# In[17]:

np.arange(0, 20, 4)


# ### 스칼라 연산

# In[18]:

b


# In[19]:

b.dtype


# In[20]:

1 / b


# In[21]:

(1 / b).dtype


# In[22]:

b  # 원본은 불변


# ### 색인과 슬라이싱

# In[23]:

b[0]


# In[24]:

b[0][2]


# In[25]:

b[0, 2]


# In[26]:

b[1] = [4, 5, 20]
b


# In[27]:

b[1:, :2]


# In[28]:

b[:, 1:]


# In[29]:

b[:, 1:] = 0
b


# ### 불리언 색인

# In[30]:

user = np.array(["길동", "철수", "영희", "길동", "길동", "철수", "하늘"])
c = np.arange(21).reshape((7, 3))


# In[31]:

user


# In[32]:

c


# In[33]:

user == "철수"


# In[34]:

c[user == "철수"]


# In[35]:

c[user != "길동"]


# In[36]:

con = (user == "영희") | (user == "하늘")
c[con]


# In[37]:

c[c > 10] = 0
c


# ### 배열의 축 바꾸기

# In[38]:

d = np.arange(10).reshape((2, 5))
d


# In[39]:

d.T


# In[40]:

np.dot(d.T, d)


# In[41]:

e = np.arange(24).reshape(2, 3, 4)
e


# In[42]:

e.transpose((0, 2, 1))


# In[43]:

e.swapaxes(1, 2)


# ### python list타입과 numpy ndarray의 성능 비교

# In[44]:

import numpy as np
import time


# In[45]:

size = 10000000 # 천만개


# In[46]:

def origin():
    start = time.time()
    x = range(size)
    y = range(size)
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    end = time.time()
    return end - start


# In[47]:

def numpy():
    start = time.time()
    x = np.arange(size)
    y = np.arange(size)
    z = x + y
    end = time.time()
    return end - start


# In[48]:

t1 = origin()
t2 = numpy()
print("original processtime : %.5fs" % t1)
print("numpy processtime : %.5fs" % t2)
print("numpy is %.2f times faster" % (t1/t2))


# ### 불리언 색인

# In[49]:

user = np.array(["길동", "철수", "영희", "길동", "길동", "철수", "하늘"])
c = np.arange(21).reshape((7, 3))


# In[50]:

user


# In[51]:

c


# In[52]:

user == "철수"


# In[53]:

c[user == "철수"]


# In[54]:

c[user != "길동"]


# In[55]:

mixed_condition = (user == "영희") | (user == "하늘")
c[mixed_condition]


# In[56]:

c[c > 10] = 0
c


# ### 전치와 축 바꾸기

# In[57]:

d = np.arange(10).reshape((2, 5))
d


# In[58]:

d.T


# In[59]:

np.dot(d.T, d)


# In[60]:

e = np.arange(24).reshape(2, 3, 4)
e


# In[61]:

e.transpose((0, 2, 1))


# In[62]:

e.swapaxes(1, 2)


# ### 유니버설 함수
# 간단한 기능을 빠르게 수행하는 함수

# In[63]:

a = np.arange(10)
a


# In[64]:

np.sqrt(a)


# In[65]:

np.square(a)


# In[66]:

x = np.arange(0, 20, 2)
x


# In[67]:

y = np.arange(0, 30, 3)
y


# In[68]:

np.maximum(x, y), np.add(x, y)


# In[69]:

z = np.random.randn(5) * 5
z


# In[70]:

np.modf(z)


# #### 자주 쓰이는 단항 유니버설 함수
# abs, sqrt, exp, log, cos, sin, tan
# 
# #### 자주 쓰이는 이항 유니버설 함수
# add, subtract, multiply, divide, power

# ### 적극적인 배열 사용

# In[71]:

points = np.arange(-5, 5, 0.01)
points.shape, points.ndim


# In[72]:

xs, ys = np.meshgrid(points, points)
xs.shape, xs.ndim


# In[73]:

xs, ys


# In[74]:

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


# In[75]:

z = np.sqrt(xs ** 2 + ys ** 2)
z, z.shape, z.ndim


# In[76]:

plt.imshow(z, cmap="gray")
plt.colorbar()
plt.show()


# ### 조건절 표현

# np.where
# 
# x if condition else y
# 
# if condition:
#     x
# else:
#     y

# In[77]:

x = np.array([1, 2, 3, 4, 5])
y = np.array([11, 12, 13, 14, 15])
condition = np.array([True, True, False, True, False])


# In[78]:

# case 1
result = [(xv if c else yv) for xv, yv, c in zip(x, y, condition)]
result


# In[79]:

# case 2
result = []
for i, c in enumerate(condition):
    if c:
        result.append(x[i])
    else:
        result.append(y[i])
result


# In[80]:

# case 3
np.where(condition, x, y)


# In[81]:

a = np.random.randn(1000, 1000)
a.shape, a.ndim, a


# In[82]:

get_ipython().magic('timeit np.where(a > 0, 1, -1)')
np.where(a > 0, 1, -1)


# In[83]:

np.where(a > 0, 1, a)


# #### 주민번호 뒷자리 구하기

# In[84]:

user = np.array(["철수", "길동", "유리", "이슬"])
is_man = np.array([True, True, False, False])
is_adult = np.array([False, True, False, True])


# In[85]:

# not numpy
result = []
for i in range(len(user)):
    if is_man[i] and is_adult[i]:
        result.append(1)
    elif is_man[i]:
        result.append(3)
    elif is_adult[i]:
        result.append(2)
    else:
        result.append(4)
result


# In[86]:

# numpy
np.where(is_man & is_adult, 1,
        np.where(is_man, 3,
                np.where(is_adult, 2, 4)))


# ### 수학, 통계 함수

# In[87]:

arr = np.random.randn(3, 5)
arr


# In[88]:

arr.mean()


# In[89]:

np.mean(arr)


# In[90]:

arr.sum(), np.sum(arr)


# In[91]:

arr.mean(axis=1)


# In[92]:

arr.sum(1)


# In[93]:

arr = np.arange(9).reshape((3, 3))
arr


# In[94]:

arr.cumsum(0)


# In[95]:

arr.cumsum(1)


# sum ,mean, std, var, min, max, cumsum, cumprod

# ### 불리언 배열 함수

# In[98]:

arr = np.random.randn(100)
(arr > 0).sum()


# In[99]:

bools = np.array([True, False, True])


# In[100]:

bools.any()


# In[101]:

bools.all()


# ### 정렬

# In[102]:

arr = np.random.randn(5)
arr


# In[103]:

arr.sort()
arr


# In[104]:

arr = np.random.randn(4, 3)
arr


# In[105]:

arr.sort(1)
arr


# In[106]:

arr = np.random.randint(0, 1000, size=1000)


# In[107]:

arr[:10]


# In[108]:

arr.sort()
arr[:10]


# In[109]:

arr[int(0.1 * len(arr))]


# ### 집합 함수

# In[110]:

user = np.array(["길동", "철수", "영희", "길동", "길동", "철수", "하늘"])
np.unique(user)


# In[111]:

ints = np.array([1,2,3,4,5,4,3,2,1])
np.unique(ints)


# In[112]:

# python version
sorted(set(user))


# In[113]:

np.in1d(ints, [2, 3, 6])


# In[114]:

# python version
result = []
for i in ints:
    if i in [2, 3, 6]:
        result.append(True)
    else:
        result.append(False)
result


# np.intersect1d(x, y) : 교집합
# 
# np.union1d(x, y) : 합집합
# 
# np.setdiff1d(x, y) : 차집합

# ### 배열의 파일 입출력

# In[115]:

arr = np.arange(10)
np.save("1to9", arr)


# In[116]:

np.load("1to9.npy")


# In[117]:

np.savez("arr_list", first=arr, second=arr)


# In[118]:

inventory = np.load("arr_list.npz")
inventory["first"]


# ### 난수 생성

# In[119]:

samples = np.random.randn(4, 4)
samples


# In[120]:

from random import uniform
n = 1000000 #백만개


# In[121]:

get_ipython().magic('timeit [uniform(0, 1) for _ in range(n)]')


# In[122]:

get_ipython().magic('timeit np.random.randn(n)')


# In[123]:

user


# In[126]:

np.random.shuffle(user)


# In[127]:

user


# ### 계단 오르내리기

# #### original 버전

# In[129]:

from random import randint
import matplotlib.pyplot as plt
pos = 0
walk = [pos]
steps = 1000
for i in range(steps):
    if randint(0, 1): # 1 은 True, 0 은 False
        step = 1
    else:
        step = -1
    pos += step
    walk.append(pos)
plt.plot(walk)
plt.title("Random walk with 1 step")
plt.show()


# #### numpy 버전

# In[132]:

nsteps = 1000
records = np.random.randint(0, 2, size=nsteps)
steps = np.where(records == 1, 1, -1)
walk = steps.cumsum()
plt.plot(walk)
plt.title("Random walk with 1 step")
plt.show()


# In[133]:

walk.min(), walk.max()


# In[135]:

(np.abs(walk) >= 15).argmax()


# #### 계단 오르내리기를 1만번하기

# In[138]:

nwalks = 10000
nsteps = 1000
records = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(records == 1, 1, -1)
walks = steps.cumsum(1) # 행은 그대로두고 열끼리 누적합
walks.shape, walks.ndim, walks


# In[139]:

walks.max(), walks.min()


# In[140]:

result = (np.abs(walks) >= 50).any(1)
print("각각의 시뮬레이션에 대해 50계단 이동했는가")
print("시뮬레이션 횟수 :", len(result))
print(result)


# ##### 10,000번 중 50계단 이동한 시뮬레이션의 횟수

# In[141]:

result.sum()


# In[142]:

cross = (np.abs(walks[result]) >= 50).argmax(1)
cross


# In[143]:

cross.mean()


# ### numpy 활용

# #### numpy에서의 `enumerate`

# In[144]:

#python에서의 enumerate
a = [5, -1, 22, 3, 40]
for index, value in enumerate(a):
    print(index, value)


# In[145]:

Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)


# In[146]:

for index in np.ndindex(Z.shape):
    print(index, Z[index])


# #### 2차원 배열에 무작위로 집어넣기

# In[149]:

n = 5
p = 3
Z = np.zeros((n,n))
position = np.random.choice(range(n*n), p, replace=False)
print(position)
np.put(Z, position,1)
print(Z)


# #### 가까운 값 찾기

# In[150]:

Z = np.random.uniform(0,1,10)
print("Z배열\n", Z)
target = 0.5
print("목표 값과의 차이\n", np.abs(Z - target))
print("가장 근접한 값의 위치\n", np.abs(Z - target).argmin())
m = Z[np.abs(Z - target).argmin()]
print("그 값\n", m)

