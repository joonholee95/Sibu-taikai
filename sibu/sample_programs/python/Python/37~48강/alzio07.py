
# coding: utf-8

# In[1]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


# # pandas로 데이터 다루기

# ## 변형
# - 중복 제거
# - 매핑
# - 치환
# - 특이값 다루기

# ### 중복 제거

# In[2]:


df = DataFrame({'key1': list("가가가나나다다다"), 'key2': list("13122414")})
df


# In[3]:


df.duplicated()


# In[4]:


df.drop_duplicates()


# In[5]:


df.drop_duplicates(keep='last')


# In[6]:


df.drop_duplicates('key1')


# In[7]:


df.drop_duplicates('key2')


# ### 매핑

# In[8]:


data = {
    '이름': list("가나다라마"),
    '점수': np.arange(5) * 10
}
df = DataFrame(data)
df


# In[9]:


gender = {
    "나": "여",
    "마": "여",
    "가": "남",
    "라": "남",
    "다": "여",
}


# In[10]:


df['성별'] = df['이름'].map(gender)
df


# In[11]:


df['점수'] = df['점수'].map("{}점".format)
df


# ### 치환

# In[12]:


sr = Series([2000, -1, 2001, 2002, -1, 0, 2003])
sr


# In[13]:


sr.replace(-1, np.nan)


# In[14]:


sr.replace([-1, 0], np.nan)


# In[15]:


sr.replace({-1: np.nan, 0: 2000})


# ### 특이값 다루기

# In[16]:


np.random.seed(1)
df = DataFrame(np.random.randn(1000, 5))
df.describe()


# In[17]:


plt.hist(df[2], bins=100)
plt.show();


# In[18]:


abs(df[2]) > 2.5


# In[19]:


df[abs(df[2]) > 2.5]


# In[20]:


df[abs(df[2]) > 2.5] = np.sign(df) * 2.5


# In[21]:


plt.hist(df[2], bins=100)
plt.show();


# # matplotlib로 그래프 그리기

# In[22]:


from numpy.random import randn


# ## 기본

# In[23]:


plt.plot(randn(50).cumsum())
plt.show();


# ## 서브플롯

# In[24]:


fig, axes = plt.subplots(2, 2)
axes[0, 0].hist(randn(100), color='k')
axes[0, 1].plot(randn(100).cumsum(), c='red')
axes[1, 0].scatter(randn(100), randn(100), color='green')
axes[1, 1].bar(np.arange(100), np.random.randint(1, 100, size=100), color='blue')
plt.show();


# ## 서브플롯간 간격 조절

# In[25]:


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
axes[0, 0].hist(randn(1000), color='k')
axes[1, 0].hist(randn(1000), color='r')
axes[0, 1].hist(randn(1000), color='g')
axes[1, 1].hist(randn(1000), color='b')
plt.subplots_adjust(wspace=0, hspace=0)
plt.show();


# ## 크기, 마커, 선 스타일

# In[26]:


plt.figure(figsize=(12, 6))
plt.plot(randn(50).cumsum(), linestyle='-', color='k', marker='*')
plt.plot(randn(50).cumsum(), linestyle='--', color='r', marker='o')
plt.plot(randn(50).cumsum(), linestyle='-.', color='g', marker='v')
plt.plot(randn(50).cumsum(), linestyle=':', color='b', marker='D')
plt.show();


# ## 눈금

# In[27]:


plt.plot(randn(500).cumsum())
plt.show();


# In[28]:


plt.plot(randn(500).cumsum())
plt.xticks([0, 250, 500])
plt.show();


# In[29]:


plt.plot(randn(500).cumsum())
plt.xticks([0, 250, 500], ["2000년 1월 1일", "7월 1일", "2001년 1월 1일"], rotation=45)
plt.show();


# ## 라벨, 범례

# In[30]:


plt.plot(randn(50).cumsum(), linestyle='-', color='k', label="검정")
plt.plot(randn(50).cumsum(), linestyle='--', color='r', label="빨강")
plt.plot(randn(50).cumsum(), linestyle='-.', color='g', label="초록")
plt.xlabel("스텝")
plt.ylabel("값")
plt.title("제목")
plt.legend(loc="lower left")
plt.show();


# ## 저장

# In[32]:


fig = plt.figure()
plt.plot(randn(50).cumsum(), linestyle='-', color='k', label="검정")
plt.plot(randn(50).cumsum(), linestyle='--', color='r', label="빨강")
plt.plot(randn(50).cumsum(), linestyle='-.', color='g', label="초록")
plt.xlabel("스텝")
plt.ylabel("값")
plt.title("제목")
plt.legend(loc="lower left")
plt.savefig("test.jpg", dpi=300);
plt.show();


# ![](test.jpg)
