
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

# In[ ]:




# In[11]:

type(a["a"])
a["a"].argmax()


# #### quantile

# In[12]:

# quantile bottom 10%
a.quantile(0.1, axis=1)


# ##### quantile example

# In[18]:

np.random.seed(42)
score = DataFrame(np.random.randint(0, 100, size=(4, 3)),
               index=["영희","철수","하늘","민석"],
               columns=["영어", "수학", "파이썬"])
score


# In[19]:

pivot = score.quantile(0.5)
pivot


# In[20]:

mask = score > pivot
mask


# In[21]:

score[mask]


# In[22]:

score[mask] = "Pass"
score[~mask] = "Fail"
score


# #### diff

# In[23]:

b = DataFrame(np.arange(1, 17).reshape(4, 4), index=list("가나다라"), columns=list("abcd"))
b


# In[24]:

b.diff()


# In[25]:

b.diff(axis=1)


# In[26]:

b.diff(axis=1).ix[:, 1:]


# In[27]:

b.diff(axis=1).ix[:, 1:].rename(columns={"b":"a-b", "c":"b-c", "d":"c-d"})


# #### pct_change

# In[28]:

b


# In[29]:

b.pct_change()


# ### 상관관계와 공분산

# 보다 이해하기 쉽게 야후의 금융 데이터를 불러온다.  
# `pip install pandas-datareader`를 통해 실습에 필요한 각종 데이터를 담고있는 패키지를 설치할 수 있다.

# In[30]:

from pandas_datareader import data


# In[31]:

c = {}
from datetime import datetime
for symbol in ["FB", "TWTR", "GOOG", "MSFT"]:
    c[symbol] = data.DataReader(symbol, "yahoo",datetime(2015,1,1),datetime(2016,1,1))
    print(symbol, ": load finish")


# In[32]:

type(c)
len(c)


# In[33]:

type(c["FB"])


# In[34]:

c["FB"].head()


# In[35]:

plt.plot(c["FB"]["Adj Close"])
plt.show();
plt.close();
