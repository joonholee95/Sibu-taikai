
# coding: utf-8

# # Pandas

# - 이름을 가진 축에 따라 데이터 정렬
# - 다양한 방식으로 저장된 데이터 처리
# - 시계열, 비시계열 데이터를 통합적으로 처리
# - 축의 이름으로 데이터 연산
# - 누락된 데이터의 유연한 처리

# In[1]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np


# `pd.Series`와 `pd.DataFrame`보다는 직접 import해 사용하는 것이 편리함

# ## 주요 자료 구조
# - Series
# - DataFrame
# - index object

# ### Series
# 특징 : 1차원 배열, index(색인)

# In[2]:


a = [42, -9, 5, 10, 2]
series_object = Series(a)


# In[3]:


series_object


# In[4]:


series_object.values


# In[5]:


type(series_object.values)


# In[6]:


series_object.index


# In[7]:


b = ["안녕하세요", 20000, 5.0, [1, 2, 3], (-1, 0)]
series_object2 = Series(b)


# In[8]:


series_object2


# #### 색인의 이름 지정

# In[9]:


s_obj = Series(b, index=["문자", "정수", "실수", "리스트", "튜플"])
s_obj


# In[10]:


s_obj.index


# In[11]:


s_obj["실수"]


# In[12]:


s_obj["정수"] = 42
s_obj["정수"]


# In[13]:


s_obj[["실수", "튜플", "문자"]]


# In[14]:


s_obj[["실수", "정수"]] = 3.141592, -5
s_obj[["실수", "정수"]]


# In[15]:


# index의 이름 변경
s_obj.index = ["string", "int", "float", "list", "tuple"] # 갯수가 맞지 않으면 miss match
s_obj

# 되돌리기
s_obj.index = ["문자", "정수", "실수", "리스트", "튜플"]


# #### pandas의 불리언 배열과 산술연산

# In[16]:


s_obj2 = Series([3, -6, 9, 4])
s_obj2 > 0 # Numpy와 비슷


# In[17]:


s_obj2[s_obj2 > 3]


# In[18]:


np.power(s_obj2, 3)


# In[19]:


s_obj2 * -1


# #### python의 dict와 비슷한 pandas의 Series
# index -> key  
# values -> values

# In[20]:


s_obj


# In[21]:


dict_obj = {
    "문자" : "안녕하세요",
    "정수" : -5,
    "실수" : 3.14159,
    "리스트" : [1, 2, 3],
    "튜플" : (-1, 0)
}
dict_obj


# In[22]:


print("문자" in dict_obj)
print("문자" in s_obj)
print("딕셔너리" in s_obj)


# In[23]:


s_obj["딕셔너리"] = {"key" : "value"}
s_obj


# #### 딕셔너리로부터 Series객체 생성

# In[24]:


income = {
    "홍길동" : 200 * 10000,
    "김철수" : 10 * 10000,
    "윤아름" : 123 * 10000,
    "박영희" : 5 * 10000
}
s_obj3 = Series(income)
s_obj3 # index값이 순서대로 들어가지 않았다.


# In[25]:


name = ["최갑을", "김철수", "윤아름", "박영희"] # list를 index인자로 전달하면 순서가 유지된다
s_obj4 = Series(income, index=name) # index값이 살짝 다르다
s_obj4


# NaN = not a number, 누락된 값  
# pandas에서는 누락된 값을 `isnull`과 `notnull`함수로 찾을 수 있다

# In[26]:


pd.isnull(s_obj4)


# In[27]:


pd.notnull(s_obj4)


# NumPy에서 그랬던 것처럼 Series객체의 함수로도 있다.

# In[28]:


s_obj4.isnull()


# #### index가 다른 Series의 산술 연산

# In[29]:


print(s_obj3)
print(s_obj4)


# In[30]:


s_obj3 - s_obj4


# #### name 속성

# In[31]:


s_obj4


# In[32]:


s_obj4.name = "수입"
s_obj4


# In[33]:


s_obj4.index.name = "이름"
s_obj4


# ### DataFrame
# 특징 : 스프레드시트와 비슷, 여러 개의 열과 행으로 구성  
# Series가 모인거라고 생각하면 편함

# #### DataFrame객체의 생성
# - 같은 길이의 리스트가 담긴 딕셔너리
# - Numpy 배열

# In[34]:


data = {
    "구" : ["마포구", "마포구", "양천구", "양천구", "은평구"],
    "연도" : [2015, 2016, 2015, 2016, 2014],
    "소득" : [12.5, 13.3, 16.5, 14.9, 10.1]
}
seoul_income = DataFrame(data)


# In[35]:


seoul_income # 아까와 마찬가지로 순서가 유지되지는 않음


# `columns`인자로 열의 이름을 지정해줄 수 있고, `index`인자로 색인을 지정할 수 있다

# In[36]:


DataFrame(data, columns=["연도", "구", "소득"], index=["1번", "2번", "3번", "4번", "5번"])


# In[37]:


# 참고할 파이썬 팁 : 리스트 컴프리헨션
["%d번"%(i+1) for i in range(5)]


# data와 매칭되지 않는 값을 주면 NaN값이 저장된다.

# In[38]:


seoul2 = DataFrame(data, columns=["연도", "구", "소득", "인구"])
seoul2


# 중첩 딕셔너리를 사용한 DataFrame생성

# In[39]:


star = {
    "Python" : {
        2015 : 3.5,
        2016 : 3.6
    },
    "Assembly" : {
        2014 : 0.1,
        2015 : 0.2,
        2016 : 0.1
    }
}
popularity = DataFrame(star)
popularity


# In[40]:


# NumPy의 전치행렬과 비슷
popularity.T


# In[41]:


DataFrame(star, index=[2015, 2016, 2017])


# #### DataFrame의 name, values

# In[42]:


popularity.index.name = "연도"
popularity.columns.name = "언어"
popularity


# In[43]:


popularity.values


# In[44]:


# NumPy를 쓰지 않았지만 타입을 확인해보면...
type(popularity.values)


# In[45]:


# ndarray니까 shape, dtype도 확인할 수 있다
popularity.values.shape, popularity.values.dtype


# In[46]:


# 값이 하나로 통일되어있지않다면
seoul2.values


# 그 외에 DataFrame을 생성할 수 있는 인자 값으로는  
# - 2차원 ndarray
# - Series의 딕셔너리
# - 중첩된 딕셔너리
# - 2차원 리스트 등등
# 

# In[47]:


a = Series([1, 2, 3], index=["a", "b", "c"])
a.name = "1번"
b = Series([4, 5, 6], index=["a", "b", "d"])
b.name = "2번"
DataFrame([a, b]).T


# #### DataFrame의 칼럼(열)

# In[48]:


seoul2.columns


# DataFrame의 칼럼은 인덱스와 속성으로 접근할 수 있다.

# In[49]:


seoul2["구"]


# In[50]:


type(seoul2["구"]) # Series객체가 반환된다.


# In[51]:


seoul2.구 # 한글 속성이라 조금 어색해 보일 수 있다.


# #### DataFrame의 로우(행)

# In[52]:


seoul2.ix[1]


# #### 칼럼의 연산

# In[53]:


seoul2["인구"] = 500000
seoul2


# In[54]:


seoul2["인구"] = [i * 10000 for i in range(1, 6)]
seoul2


# In[55]:


# numpy와 함께 쓰겠다면
seoul2["인구"] = np.arange(5, 0, -1) * 10000
seoul2


# 색인이 지정된 Series의 대입

# In[56]:


data = Series([5000, 6000], index=[1, 3])
seoul2["인구"] = data
seoul2 # index가 지정되지 않으면 NaN


# In[57]:


del seoul2["인구"]
seoul2


# ### Index Object
# 특징 : 메타데이터를 저장하는 객체  
# `Series` 혹은 `DataFrame`에서 객체를 생성할 때 내부적으로 `Index Object(색인 객체)`로 변환됨

# In[58]:


vals = [1, 2, 3, 4]
cols = ["일", "이", "삼", "사"]
obj = Series(vals, index=cols)
obj


# In[59]:


index = obj.index
index


# In[60]:


index[2:]


# #### `Index Object`는 `immutable(변경불가능)`하다

# In[61]:


index[2] = "Three"


# #### append, delete, drop, insert를 쓰면 새로운 index object를 반환한다

# In[62]:


index.append("오")


# In[63]:


pd.Index(["오"])


# In[64]:


index.append(pd.Index(["오"]))


# In[65]:


index # 원본 객체는 그대로다


# In[66]:


# delete
print(index.delete(2))

# drop
print(index.drop("이"))

# insert
print(index.insert(2, "영"))


# #### 그렇기에 `pandas`를 쓰면서 안전하게 쓸 수 있다.

# In[67]:


index2 = pd.Index([1, 2, 3, 4]) # index가 아닌 Index
index2


# In[68]:


a = Series([-1, 0, 1, 0], index=index)
a


# In[69]:


a.index is index


# In[70]:


# == 으로 비교하면 불리언 배열이 반환된다.
a.index == index


# #### 여러가지 Index Object

# - Index : 일반적인 인덱스
# - Int64Index : 정수값 인덱스
# - DatetimeIndex : 타임스탬프 인덱스
# - PeriodIndex : 기간 인덱스
# 
# DatetimeIndex, PeriodIndex는 시계열 데이터에서 굉장히 자주 쓰인다.

# In[71]:


from datetime import datetime
a = datetime(2000,1,1)
b = datetime(2000,1,2)
c = datetime(2000,1,3)
date_index = pd.Index([a, b, c])
date_index


# In[72]:


d = Series([1, 2, 3], index=date_index)
d


# In[73]:


import matplotlib.pyplot as plt
plt.plot(d)
plt.show()


# In[74]:


e = pd.period_range("2000-02-01", "2000-02-28", freq="D")
e


# In[75]:


e[2:13]

