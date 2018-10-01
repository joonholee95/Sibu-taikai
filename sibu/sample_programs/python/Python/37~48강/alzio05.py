
# coding: utf-8

# In[1]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


# ### 값 체크

# In[2]:


a = Series(list("가나라다다나라다"))
a


# In[3]:


a.isin(["나", "라"])


# In[4]:


mask = a.isin(["나", "라"])
a[mask]


# In[5]:


a[~mask]


# - `unique` : 유일한 값만 반환, set과 비슷
# - `value_counts` : 각 값들이 몇개나 있는지
# - `isin` : 각 원소가 인자값에 속하는지

# #### DataFrame

# In[6]:


b = DataFrame(
    [[0, 1, 2], [1, 1, 0], [3, 2, 1]],
    index=list("가나다"),
    columns=list("abc")
)
b


# #### isin

# In[7]:


b.isin([1, 2])


# In[8]:


b[b.isin([1, 2])]


# #### unique와 value_counts

# In[9]:


try:
    b.unique()
except Exception as e:
    print(e)
    
try:
    b.value_counts()
except Exception as e:
    print(e)


# In[10]:


b.values
b.values.flatten()
pd.unique(b.values.flatten())


# In[11]:


b


# In[12]:


b.apply(pd.value_counts)


# ## 누락 데이터 처리

# 누락된 데이터: null, NaN (`np.nan`)
# - `dropna`: 누락된 데이터가 있으면 제외
# - `fillna`: 누락된 데이터를 채움
# - `isnull`: 누락된 데이터인지 판별
# - `notnull`: `isnull`의 반대

# In[13]:


b.apply(pd.value_counts).fillna(0)


# In[14]:


c = Series(["NaN", np.nan, "seoul", "busan"])
c


# In[15]:


c.isnull()


# `None`도 null값으로 취급

# In[16]:


c[2] = None
c


# In[17]:


c.isnull()


# ### null값 제외하기

# #### Series

# In[18]:


d = Series([1, np.nan, 3, np.nan, 2])
d


# In[19]:


d.dropna()


# In[20]:


mask = d.notnull()
d[mask]


# #### DataFrame
# - 행이나 열에 하나라도 null값이면 제외
# - 행이나 열이 모두 null값이어야 제외

# In[21]:


e = DataFrame([
    [1, np.nan, 3],
    [np.nan, 7, 5],
    [np.nan, np.nan, np.nan],
    [-1, 2, 0]
])
e


# In[22]:


e.dropna()


# In[23]:


e.dropna(how="all")


# 컬럼을 기준으로 하려면

# In[24]:


e[0] = np.nan
e


# In[25]:


e.dropna(axis=1, how="all")


# `NaN이 아닌 값`이 몇개 이상인 행만 보고싶다면 `thresh`인자 사용

# In[26]:


e.dropna(thresh=1)


# In[27]:


e.dropna(thresh=2)


# ### null값 채우기

# In[28]:


f = DataFrame(np.random.rand(6, 4))
f.ix[1:2, 1:2] = None
f.ix[2:4, 2:3] = None
f


# In[29]:


f.fillna(0)


# 칼럼마다 다른 값으로 채울 수 있다.

# In[30]:


f.fillna({1: '1번', 2: '2번', 3: '3번'})


# 앞에서 배웠던 `ffill`, `bfill` 보간 메서드를 사용할 수 있다.

# In[31]:


f.fillna(method='ffill')


# In[32]:


f.fillna(method='bfill')


# In[33]:


f.fillna(method='ffill', limit=2)


# #### `fillna`의 활용

# In[34]:


g = Series([5, np.nan, 7, 8, np.nan, 10])
g


# In[35]:


plt.plot(g.fillna(0), label='real')
plt.plot(range(5, 11), linestyle='--', c='red', label='expect')
plt.ylim(0, 12)
plt.legend()
plt.show();


# In[36]:


g.fillna(g.mean())


# In[37]:


plt.plot(g.fillna(g.mean()), label='real')
plt.plot(range(5, 11), linestyle='--', c='red', label='expect')
plt.ylim(0, 12)
plt.legend()
plt.show();


# ## 계층적 색인
# **매우 핵심적인 기능**
# 
# |  부서  | 이름 | 성과 |
# |:------:|:----:|:----:|
# |  인사  | 김xx |   5  |
# |        | 박xx |   4  |
# |        | 이xx |   5  |
# |  재무  | 최xx |   2  |
# |        | 남xx |   4  |
# | 마케팅 | 정xx |   5  |
# |        | 한xx |   3  |

# ### Series

# In[38]:


a = Series([5, 4, 5, 2, 4, 5, 3],
          index=[['인사'] * 3 + ['재무'] * 2 + ['마케팅'] * 2,
                ['김xx', '박xx', '이xx', '최xx', '남xx', '정xx', '한xx']])
a


# In[39]:


a = Series(np.random.randn(7),
          index=[['a'] * 3 + ['b'] * 2 + ['c'] * 2,
                ['가', '나', '다', '가', '나', '나', '다']])
a


# In[40]:


a.index


# 원래 `Series`객체는 행의 index로만 접근할 수 있었지만

# In[41]:


a['b']


# In[42]:


a['a':'b']


# In[43]:


a[['a', 'c']]


# In[44]:


a[:, '나']


# 뭔가 `DataFrame`과 비슷한 것 같다.  
# 그래서 있는게 `unstack()`메서드

# In[45]:


a


# In[46]:


a.unstack()


# In[47]:


type(a.unstack())


# In[48]:


a.unstack().stack()


# ### DataFrame

# In[49]:


b = DataFrame(np.arange(15).reshape(5, 3),
             index=[['a', 'a', 'a', 'b', 'b'],
                   ['가', '나', '다', '가', '나']],
             columns=[['서울', '서울', '부산'],
                     ['남', '여', '여']])
b


# In[50]:


b.index
b.columns


# In[51]:


b.index.names = ['1차', '2차']
b.columns.names = ['도시', '성별']


# In[52]:


b


# In[53]:


b['서울']


# In[54]:


b.ix['a']


# #### 계층적 색인을 활용한 통계

# In[55]:


b.sum()


# In[56]:


b


# In[57]:


b.sum(level="1차")


# In[58]:


b.mean(level="도시", axis=1)


# In[59]:


b.sum(level="성별", axis=1).sum(level="2차")


# #### 일반적인 DataFrame에서 계층적 색인 추가

# In[60]:


c = DataFrame({
    "학년": [1, 1, 1, 2, 2, 3, 3],
    "반": [2, 5, 2, 1, 5, 4, 3],
    "이름": list("가나다라마바사"),
    "점수": [95, 85, 100, 20, 86, 0, 77],
})
c


# `Series`의 `stack()`처럼

# In[61]:


d = c.set_index(["학년", "반"])
d


# In[62]:


d.mean(level="학년")


#  

#  

#  

#  

#  

#  

# # Pandas로 데이터 save, load 하기

# ## CSV

# ### 읽기

# - `df = pd.read_csv("파일명.csv")`
# - `df = pd.read_csv("파일명.csv", sep=";")`  
# 정규표현식도 가능
# - `df = pd.read_csv("파일명.csv", header=None)`  
# 칼럼의 이름으로 사용할 행의 번호
# - `df = pd.read_csv("파일명.csv", header=["부서", "이름", "성과"])`
# - `df = pd.read_csv("파일명.csv", header=None, names=["부서", "이름", "성과"])`
# - `df = pd.read_csv("파일명.csv", index_col=["1차", "2차"])`  
# 색인으로 사용할 열의 번호나 이름
# - `df = pd.read_csv("파일명.csv", skiprows=[1, 5])`  
# 무시할 행의 개수 혹은 번호가 담긴 리스트
# - `df = pd.read_csv("파일명.csv", na_values=["없음", "누락", -1])`  
# null값으로 변환할 문자들
# - `df = pd.read_csv("파일명.csv", parse_dates=[0])`  
# `datetime`으로 변환할 열의 번호
# - `df = pd.read_csv("파일명.csv", nrows=5)`  
# 읽어올 행의 수
# 

# ### 쓰기

# - `df.to_csv("저장할파일명.csv")`
# - `df.to_csv("저장할파일명.csv", sep=";")`

# ## 엑셀
# xlrd, openpyxl 패키지 설치 필요
# > `pip install xlrd openpyxl`

# ```py
# excel_file = pd.ExcelFile("파일명.xls")
# df = excel_file.parse("시트이름")
# ```

# ## HTML

# In[ ]:


import requests  # pip install requests 로 설치 가능
url = "https://api.github.com/repos/pydata/pandas/milestones/50/labels"
res = requests.get(url)
res


# In[ ]:


r = res.json()
r[:3]


# In[ ]:


df = pd.DataFrame(r)
df.head()

