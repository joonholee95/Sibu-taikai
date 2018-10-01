
# coding: utf-8

# In[1]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


# ## pandas에서의 통계
# 누락된 데이터를 유연하게 처리하며 각종 통계처리와 데이터 축소 작업을 쉽게 할 수 있다.

# ### 기초

# In[3]:


a = DataFrame(
    [[-5.4, 3], [np.nan, np.nan], [-2, np.nan], [2.8, 1.4]],
    index=list("가나다라"),
    columns=list("ab")
)
a


# In[4]:


a.sum()
type(a.sum())


# In[5]:


a.sum(axis=1)


# In[6]:


a.mean(axis=1)


# `nan`값을 포함해 계산하고 싶다면

# In[7]:


a.mean(axis=1, skipna=False)


# In[8]:


a.idxmax()


# In[9]:


a.idxmax(axis=1)


# 거의 모든 메서드에는 `axis`인자가 있다.

# In[10]:


a
a.describe()


# 그 외에도 다양한 통계 메서드가 존재한다.
# - count : nan값을 제외한 값의 수
# - min, max
# - **argmin, argmax** (Series) : 최소, 최대 값의 위치
# - quantile : 0%부터 100%까지의 분위수
# - var, std
# - cumsum : 누적합
# - **diff** : 산술 차 - 시계열 데이터 처리 시 유용
# - **pct_change** : 퍼센트 변화율

# #### argmax

# In[11]:


type(a["a"])
a["a"].argmax()


# #### quantile

# In[12]:


# quantile bottom 10%
a.quantile(0.1, axis=1)


# ##### quantile example

# In[13]:


np.random.seed(42)
score = DataFrame(np.random.randint(0, 100, size=(4, 3)),
               index=["영희","철수","하늘","민석"],
               columns=["영어", "수학", "파이썬"])
score


# In[14]:


pivot = score.quantile(0.5)
pivot


# In[15]:


mask = score > pivot
mask


# In[16]:


score[mask]


# In[17]:


score[mask] = "Pass"
score[~mask] = "Fail"
score


# #### diff

# In[18]:


b = DataFrame(np.arange(1, 17).reshape(4, 4), index=list("가나다라"), columns=list("abcd"))
b


# In[19]:


b.diff()


# In[20]:


b.diff(axis=1)


# In[21]:


b.diff(axis=1).ix[:, 1:]


# In[22]:


b.diff(axis=1).ix[:, 1:].rename(columns={"b":"a-b", "c":"b-c", "d":"c-d"})


# #### pct_change

# In[23]:


b


# In[24]:


b.pct_change()


# ### 상관관계와 공분산

# 보다 이해하기 쉽게 야후의 금융 데이터를 불러온다.  
# `pip install pandas-datareader`를 통해 실습에 필요한 각종 데이터를 담고있는 패키지를 설치할 수 있다.

# In[25]:


from pandas_datareader import data


# In[49]:


c = {}
from datetime import datetime
for symbol in ["FB", "TWTR", "GOOG", "MSFT"]:
    c[symbol] = data.DataReader(symbol, "google",datetime(2015,1,1),datetime(2016,1,1))
    print(symbol, ": load finish")


# In[50]:


# df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
for symbol in c:
    c[symbol] = c[symbol].rename(columns={"Close" : "Adj Close"})


# In[51]:


type(c)
len(c)


# In[52]:


type(c["FB"])


# In[53]:


c["FB"].head()


# In[54]:


plt.plot(c["FB"]["Adj Close"])
plt.show();
plt.close();


# In[55]:


fig, subplots = plt.subplots(2, 2, figsize=(13, 7))
for symbol, subplot in zip(c.keys(), subplots.flatten()):
    subplot.set_title(symbol)
    subplot.plot(c[symbol]["Adj Close"])
plt.show();
plt.close(fig);


# In[56]:


for symbol in c.keys():
    plt.plot(c[symbol]["Adj Close"], label=symbol)
plt.legend()
plt.show();
plt.close();


# #### 각 회사들의 종가를 하나의 DataFrame으로

# In[57]:


# dictionary comprehension
{symbol: df["Adj Close"].head() for symbol, df in c.items()}


# In[58]:


d = DataFrame({symbol: df["Adj Close"] for symbol, df in c.items()})
d.head()


# In[59]:


plt.plot(d)
plt.ylim([1, 120])
plt.show();


# In[60]:


for symbol in d.columns:
    plt.plot(d[symbol], label=symbol)
plt.ylim([1, 120])
plt.legend()
plt.show();


# #### 퍼센트 변화율로
# 기업들의 주가가 달라도 퍼센트로 표시한다면 한번에 확인할 수 있음

# 방금 배웠던 통계 메서드 중 `퍼센트 변화율 메서드` 적용

# In[62]:


e = d.pct_change()
e.tail()


# In[63]:


e.describe()


# In[64]:


e.head()


# In[65]:


e = e.ix[1:, :]
e.head()


# In[66]:


(e < 0).sum()


# #### 상관관계 `corr` 공분산 `cov`
# nan값이 없고  
# 정렬되어있고  
# 두 개의 `Series`

# In[67]:


e.notnull().all()


# ##### 페이스북과 구글의 주가로 본 상관관계

# In[70]:


e["FB"].corr(e["GOOG"])


# In[71]:


e.GOOG.corr(e.MSFT)


# In[72]:


fig, subplots = plt.subplots(2, 2, figsize=(13, 7))
for symbol, subplot in zip(c.keys(), subplots.flatten()):
    subplot.set_title(symbol)
    subplot.plot(c[symbol]["Adj Close"])
plt.show();
plt.close(fig);


# In[73]:


e.corr()


# ##### 산점도그래프를 통한 시각화

# In[74]:


plt.scatter(x=e.GOOG, y=e.MSFT)
plt.show();


# In[75]:


plt.scatter(x=e.GOOG, y=e.TWTR)
plt.show();


# 굳이 `pct_change()`를 통해 반환된 데이터프레임으로 분석하는 이유  
# 공분산보다 상관관계를 써야하는 이유

# #### DataFrame의 상관관계 메서드
# **corrwith**

# In[76]:


e.corrwith(e)


# In[77]:


e.corrwith(e.FB)


# In[78]:


e.corr().ix[0, :]


# ### 값 추출

# #### Series
# #### unique

# In[79]:


f = Series(list("가나라다다나라다"))
f


# In[80]:


f.unique()


# In[81]:


tmp = f.unique()
print("정렬 전", tmp)
tmp.sort()
print("정렬 후", tmp)


# In[82]:


f.duplicated()


# In[83]:


f[~f.duplicated()]


# #### 값 count

# In[84]:


f.value_counts()


# In[85]:


Series(np.random.randint(1, 11, size=1000)).value_counts()


# In[86]:


Series(np.random.randint(1, 11, size=1000)).value_counts(sort=False)


# In[87]:


f.count()


# In[88]:


f.size


# In[89]:


pd.value_counts(list("abfeadbacd"))


# #### 값 체크

# In[ ]:


f.isin(["나", "라"])


# In[ ]:


mask = f.isin(["나", "라"])
f[mask]


# In[ ]:


f[~mask]


# - `unique` : 유일한 값만 반환, set과 비슷
# - `value_counts` : 각 값들이 몇개나 있는지
# - `isin` : 각 원소가 인자값에 속하는지

# #### DataFrame

# In[ ]:


g = DataFrame(
    [[0, 1, 2], [1, 1, 0], [3, 2, 1]],
    index=list("가나다"),
    columns=list("abc")
)
g


# #### isin

# In[ ]:


g.isin([1, 2])


# In[ ]:


g[g.isin([1, 2])]


# #### unique와 value_counts

# In[ ]:


try:
    g.unique()
except Exception as e:
    print(e)
    
try:
    g.value_counts()
except Exception as e:
    print(e)


# DataFrame에서는 unique가 있을 필요가 없다.

# In[ ]:


g.values
g.values.flatten()
pd.unique(g.values.flatten())


# In[ ]:


g.apply(pd.value_counts)


# ### 누락 데이터 처리

# ### 계층적 색인
