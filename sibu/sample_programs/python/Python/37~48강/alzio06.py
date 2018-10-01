
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

# ## 병합

# ### merge
# SQL의 join과 비슷, 키를 이용해 데이터를 합친다.

# In[2]:


a = DataFrame({
    'key': list("가나가다나나"),
    'score': range(6)
})
b = DataFrame({
    'key': list("라나가"),
    'date': ['2/1', '3/14', '8/15']
})
a
b


# In[3]:


pd.merge(a, b, on='key')


# `how`키워드 인자로 left outer join, right outer join, full outer join 선택

# In[4]:


pd.merge(a, b, how='left')


# In[5]:


pd.merge(a, b, how='right')


# In[6]:


pd.merge(a, b, how='outer')


# - left, right, how, on
# - suffixes: 열 이름이 겹치면 겹치는 열 뒤에 붙여줄 접미어
# - left_index, right_index: 기준이 될 key가 색인 일 경우

# ### concat
# 이어붙이기

# #### Series

# In[7]:


a = Series([0, 1, 2], index=['나', '가', '라'])
b = Series([4, 5], index=['다', '마'])
c = Series([3, 6, 7], index=['바', '사', '아'])
pd.concat([a, b, c])


# #### DataFrame

# In[8]:


a = DataFrame(np.arange(8).reshape(4, 2),
             index=['가', '나', '다', '라'],
             columns=['score', 'data'])
b = DataFrame(np.arange(5, 9).reshape(2, 2),
             index=['가', '다'],
             columns=['city', 'age'])
pd.concat([a, b])


# In[ ]:


pd.concat([a, b], axis=1)


# In[ ]:


pd.concat([a, b], axis=1, keys=['key1', 'key2'])


# In[ ]:


pd.concat([a, b], axis=1, keys=['key1', 'key2'])['key1']


# #### 색인 무시

# In[ ]:


a = DataFrame(np.random.randn(3, 4), columns=list('abcd'))
b = DataFrame(np.random.randn(2, 3), columns=list('bda'))
a
b


# In[ ]:


pd.concat([a, b], ignore_index=True)


# In[ ]:




