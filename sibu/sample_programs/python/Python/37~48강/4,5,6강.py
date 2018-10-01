
# coding: utf-8

# In[1]:

from pandas import Series, DataFrame
import pandas as pd
import numpy as np


# In[2]:

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


# ## pandas의 주요 기능
# - 색인, 선택
# - 재색인 (reindex)
# - 중복색인
# - 행, 열 삭제
# - 연산
# - 정렬

# ### 색인, 선택
# Numpy의 ndarray 인덱스와 비슷하나  
# pandas는 인덱스가 꼭 정수일 필요가 없다

# In[3]:

a_ndarray = np.array(["가", "나", "다"])
print(a_ndarray[2])
# print(a_ndarray["가"])


# In[4]:

index = np.nonzero(a_ndarray == "나")[0][0]
index


# In[5]:

a_ndarray[index]


# #### Series

# In[6]:

b_series = Series([1, 2, 3], index=["가", "나", "다"])
b_series


# In[7]:

print(b_series[1])
print(b_series["나"])
print(b_series[0:2]) # 끝부분 제외
print(b_series[["다", "가"]])
print(b_series[[2, 1]])
print(b_series[b_series < 3])


# In[8]:

b_series["가":"다"] # 끝부분 포함


# In[9]:

b_series["가":"나"] = 0
b_series


# #### DataFrame

# In[10]:

c_df = DataFrame(np.arange(9).reshape(3, 3),
                index=["2000", "2001", "2002"],
                columns=["파이썬", "자바", "C"])
c_df

# 파이썬을 0 3 6 이 아닌 0 1 2 로 하고싶다면


# In[11]:

c_df["파이썬"]


# In[12]:

# 행은 인덱스로 선택할 수 없다
try:
    c_df["2001"]
except:
    print("Error!")


# In[13]:

# 슬라이싱으로 행을 선택할 수 있다
c_df[1:3]


# In[14]:

print(c_df["자바"] > 3)
print(type(c_df["자바"] > 3)) # Series 객체


# In[15]:

c_df[c_df["자바"] > 3]


# In[16]:

# 자바 열에 대한 조건을 만족하는 행을 뽑고싶다면
c_df["자바"][c_df["자바"] > 3]


# #### 복잡한 색인을 ix로 깔끔하게

# In[17]:

c_df.ix["2001"]


# In[18]:

c_df.ix["2001", "파이썬"]


# In[19]:

c_df.ix["2001", ["자바", "C"]]


# In[20]:

c_df.ix[c_df["자바"] > 3, "자바"]


# - `df[val]` : 열 선택(불리언 배열, 슬라이스 사용 가능)  
# - `df.ix[val]` : 행 선택  
# - `df.ix[:, val]` : 열 선택  
# - `df.ix[val1, val2]` : 행과 열 선택  

# ### 재색인 (reindex)
# 새로운 색인에 맞는 객체를 **새로** 생성함
# > 기존 객체의 색인을 바꾸는게 아님!

# #### Series

# In[21]:

d_sr = Series(np.random.randn(4), index=["a", "b", "c", "d"])
d_sr


# In[22]:

# ["a", "b", "c", "d"] 를 쉽게 생성하는 리스트 컴프리헨션
print([chr(i) for i in range(97, 101)])
# 혹은
print([chr(i) for i in range(ord("a"), ord("d") + 1)])


# In[23]:

d_sr.reindex(["c", "d", "e", "a", "b"])


# 값이 유지된 채 인덱스의 순서가 바뀌었고 e 가 추가되었다

# In[24]:

d_sr.reindex(["c", "d", "e", "a", "b"], fill_value=0)


# `fill_value`인자로 기본값을 지정해줄 수 있다
# 
# #### 시계열 데이터의 경우에는 단순히 fill_value로 채우는게 아닌 앞 값을 그대로 채워넣어줘야 값이 튀지 않는다

# In[25]:

import matplotlib.pyplot as plt


# In[26]:

e_sr = Series(np.arange(10))
plt.plot(e_sr)
plt.show()


# In[27]:

e_sr.reindex(np.arange(0, 10, 0.5))


# In[28]:

plt.plot(e_sr.reindex(np.arange(0, 10, 0.5), fill_value=0))
plt.show()


# In[29]:

f_sr = e_sr.reindex(np.arange(0, 10, 0.5), method="ffill") # forward fill
f_sr[:3]


# In[30]:

plt.plot(f_sr)
plt.show()


# #### DataFrame

# In[31]:

g_df = DataFrame(np.arange(12).reshape(4, 3),
                index=["a", "b", "d", "f"],
                columns=["한국", "미국", "영국"])
g_df


# In[32]:

# 기본적으로 행이 재색인된다
g_df.reindex(["a", "b", "c", "d"])


# In[33]:

g_df.reindex(columns=["미국", "한국", "독일"])


# In[34]:

g_df.reindex(index=["a", "b", "c", "d"],
             columns=["미국", "한국", "독일"],
            method="ffill")
# 보간은 행에서만 일어난다


# #### 앞에서 배운 ix를 통해 간결하게 재색인 할 수 있다

# In[35]:

g_df.ix[["a", "b", "c", "d"], ["미국", "한국", "독일"]]


# ### 중복색인
# 색인 값은 꼭 유일하지 않아도 된다.  
# 하지만 웬만하면 중복된 색인은 피하는게 좋다.

# #### Series

# In[36]:

h_sr = Series(range(4), index=["가", "나", "가", "나"])
h_sr


# In[37]:

h_sr.index


# In[38]:

h_sr.index.is_unique


# In[39]:

h_sr["가"]


# In[40]:

Series(range(4), index=list("abcd"))["b"]


# #### DataFrame

# In[41]:

i_df = DataFrame(np.random.randn(5, 2), index=["가", "가", "다", "나", "다"])
i_df


# In[42]:

i_df.ix["다"]


# 재색인을 하고싶다면?

# In[43]:

i_df.index = [1, 2, 3, 4, 5]
i_df


# 인덱스 오브젝트가 변경불가능한 것이지
# 색인은 언제나 변경가능하다

# ### 행 열 삭제
# **drop** 메서드를 사용

# #### Series

# In[44]:

j_sr = Series(range(5), index=["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ"])
j_sr


# In[45]:

j_sr.drop("ㄹ")


# In[46]:

j_sr.drop(["ㄱ", "ㅁ"])


# 아까 삭제했던 `ㄹ`인덱스가 남아있다.  
# `j_sr`을 확인해보면 원본 객체는 바뀌지 않았음을 알 수 있다

# In[47]:

j_sr


# `drop`연산이 적용된 객체를 활용하려면 변수에 대입해줘야한다

# In[48]:

j_sr = j_sr.drop("ㄹ") # 변수가 꼭 자기자신일 필요는 없다
j_sr


# #### DataFrame

# In[49]:

k_df = DataFrame(np.random.randint(10, size=(3, 5)),
         index=["2000", "2001", "2002"],
         columns=["서울", "부산", "제주", "강원", "대전"])
k_df


# In[50]:

k_df.drop("2001")


# 열을 삭제할 때는 바로 삭제할 순 없다.

# In[51]:

try:
    k_df.drop("부산")
except:
    print("error!")


# In[52]:

k_df.drop("부산", axis=1)


# 열과 행을 동시에 삭제하려면 체이닝해준다

# In[53]:

k_df.drop("2001").drop("제주", axis=1)


# ### 연산

# #### Series

# In[54]:

tmp1 = np.arange(4)
tmp2 = np.arange(6)
l_sr = Series(tmp1, index=["2000", "2001", "2002", "2003"])
m_sr = Series(tmp2, index=[str(i) for i in range(1998, 2004)])
print(l_sr)
print()
print(m_sr)


# In[55]:

l_sr + m_sr


# In[56]:

type(np.nan)


# In[57]:

l_sr.add(m_sr)


# #### DataFrame

# In[58]:

n_df = DataFrame(np.arange(6).reshape(2, 3),
                index=["2000", "2001"],
                columns=["python", "java", "c"])
o_df = DataFrame(np.arange(12).reshape(3, 4),
                index=["2000", "2002", "2004"],
                columns=["python", "matlab", "java", "c++"])


# In[59]:

n_df


# In[60]:

o_df


# In[61]:

n_df + o_df
# n_df.add(o_df)


# null값들을 채우고 싶다면 reindex와 마찬가지로 fill_value를 지정해준다.  
# 다른 점은 위 DataFrame에서 null값이 fill_value로 채워지는게 아닌 null값을 0으로 처리해 더해준다는 것이다.

# In[62]:

n_df.add(o_df, fill_value=0)


# 그럼에도 null값이 남아있는 이유는 참고할 값이 없기때문

# In[63]:

p_df = DataFrame(np.arange(6).reshape(2, 3),
                columns=list("가나다"))
q_df = DataFrame(np.arange(12).reshape(3, 4),
                columns=list("가나다라"))


# In[64]:

p_df


# In[65]:

q_df


# In[66]:

p_df.sub(q_df)


# In[67]:

p_df.sub(q_df, fill_value=0)


# 산술 연산 메서드에는
# - add : 덧셈
# - sub : 뺄셈
# - div : 나눗셈
# - mul : 곱셈

# #### Series와 DataFrame
# 브로드캐스팅에 대한 이해 필요

# #### numpy의 브로드캐스팅

# In[68]:

np.arange(4)


# In[69]:

np.arange(4) * 2


# In[70]:

r = np.arange(15).reshape(5, 3)
r


# In[71]:

r[0]


# In[72]:

r - r[0]


# In[73]:

r[0].repeat(5).reshape(3, 5).T


# In[74]:

r[0].repeat(5)


# In[75]:

r - r[0].repeat(5).reshape(3, 5).T


# #### Series와  DataFrame간의 연산

# In[76]:

s = DataFrame(np.arange(15).reshape(5, 3),
             index=[str(i) for i in range(2000, 2005)],
             columns=["서울", "부산", "인천"])
s


# In[77]:

t = s.ix[0] # 행을 선택, 반환된 t는 Series
print(type(t))
print(t)


# In[78]:

s - t


# 색인이 어긋난 연산일 경우

# In[79]:

u = Series(range(4), index=["서울", "제주", "경기", "부산"])
u


# In[80]:

s + u # reindex와 함께 브로드캐스팅


# 행이 아닌 열을 뽑아서 연산하면?

# In[81]:

s


# In[82]:

v = s["인천"]
v


# In[83]:

s - v


# In[84]:

s.sub(v, axis=0) # 아래가 아닌 옆으로 일어난 브로드캐스팅


# ### 정렬

# #### Series

# In[85]:

w = Series(range(5), index=list("다마나가라")) # ["다", "마", "나", "가", "라"]
w


# In[86]:

w.sort_index()


# #### DataFrame

# In[87]:

x = DataFrame(np.arange(16).reshape(4, 4),
             index=list("다나라가"),
             columns=list("cabd"))
x


# 행을 기준으로

# In[88]:

x.sort_index()


# 열을 기준으로

# In[89]:

x.sort_index(axis=1)


# In[90]:

x.sort_index().sort_index(axis=1)


# In[91]:

x.sort_index(ascending=False)


# #### 값에 따른 정렬

# #### Series

# In[92]:

y = Series(np.random.randint(-10, 10, size=5))
y


# In[93]:

y.sort_values()


# In[94]:

y


# In[95]:

y.sort_values(ascending=False)


# null값이 있다면

# In[96]:

z = Series([-5, np.nan, 1, 2, np.nan])
z


# In[97]:

type(np.nan)


# In[98]:

z.sort_values()


# In[99]:

z.sort_values(ascending=False)


# #### DataFrame

# In[100]:

aa = DataFrame([[3, 0], [-1, -3], [6, 7]],
              index=["2000", "2001", "2002"],
              columns=list("가나"))
aa


# In[101]:

try:
    aa.sort_values()
except BaseException as e:
    print(e)


# In[102]:

aa.sort_values(by="가")


# In[103]:

aa.sort_values(by="2000", axis=1)


# #### 랭크
# 정렬과 비슷하나 값에 순위를 매겨서 알려줌

# #### Series

# In[104]:

ab = Series([3, 1, 4, 0, 5, 4, 3, 2])
ab


# In[105]:

ab.rank()


# 같은 값을 가졌을 경우(e.g. 3이 각각 4번째와 5번째 순위에 위치함)  
# 평균 순위를 매김

# In[106]:

Series([1, 2, 3, 3, 3, 4]).rank()


# In[107]:

ab.rank(method="first")


# method로는 average, min, max, first가 있다.

# In[108]:

ab.rank(method="max")


# In[109]:

ab.rank(method="min")


# In[110]:

ab.rank(ascending=False)


# #### DataFrame

# In[111]:

data = {
    "철수" : [15, 13, 11],
    "영희" : [13, 14, 15],
    "민수" : [10, 9, 12]
}
ac = DataFrame(data, index=["2000", "2001", "2002"])
ac.name = "달리기 기록"
ac


# In[112]:

ac.rank() # 기본적으로 열


# ##### 해석
# 민수는 2001년 기록이 가장좋고  
# 영희는 2000년, 철수는 2002년 기록이 가장 좋다

# In[113]:

import matplotlib.pyplot as plt


# In[114]:

ad = ac.rank()


# In[115]:

plt.plot(ad)
plt.show()


# In[116]:

plt.plot(ad.ix[:, 0], label="민수") # ad.ix[:, 0].name
plt.plot(ad.ix[:, 1], label="영희", linestyle="--")
plt.plot(ad.ix[:, 2], label="철수", linestyle=":")
plt.title("기록 순위 비교 그래프", fontsize=16)
plt.xlabel("연도", fontsize=14)
plt.ylabel("순위", fontsize=14)
plt.xlim([1999.9, 2002.1])
plt.ylim([0.9, 3.1])
plt.xticks([2000, 2001, 2002], ["2000년", "2001년", "2002년"] )
plt.yticks([1, 2, 3])
plt.legend()
plt.show()


# 연도 별로 비교하고 싶다면

# In[117]:

ac


# In[118]:

ac.rank(axis=1)


# #### applymap

# In[119]:

ac.applymap(lambda x: "%d초"%x)

