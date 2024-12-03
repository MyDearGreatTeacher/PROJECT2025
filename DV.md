# Python 資料視覺化(DATA Visulization)報告
- [1.Matplotlib](#1)
- [2.Seaborn(Colab 已有支援)](#2)
- [3.Plotly](#3)
- [4.bokeh](#4)
- Pygal
  - [Pygal:Beautiful python charting](https://www.pygal.org/en/stable/) 
- pyechart
  - [A Python Echarts Plotting Library](https://pyecharts.org/) 
- mplfinance 股票資料視覺化

# 1
- [Matplotlib: Visualization with Python](https://matplotlib.org/)
## Matplotlib 常見的統計圖形
  - plot()折線圖   了解資料趨勢
  - scatter()氣泡圖   matplotlib.pyplot.scatter() 了解資料相關度
  - bar()柱狀圖
  - barh()條形圖
  - hist()直方圖
  - pie()圓餅圖
  - polar()極線圖
  - stem()——用於繪製棉棒圖
  - boxplot()箱型圖
  - errorbar()誤差棒
- [Writing mathematical expressions](https://matplotlib.org/stable/tutorials/text/mathtext.html)
- [Plot Mathematical Expressions in Python using Matplotlib](https://www.geeksforgeeks.org/plot-mathematical-expressions-in-python-using-matplotlib/)

## Elements of a Figure
- [解說圖形組成](https://github.com/PacktPublishing/Matplotlib-3.0-Cookbook/blob/master/Chapter01/Chapter%201%20-%20Anatomy%20of%20Matplotlib.ipynb)

## 官方範例
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5, 0.1)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
```
## 使用plot()畫折線圖
```pyhton
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)    # 建立含100個元素的陣列
y1 = np.sin(x)                       # sin函數
y2 = np.cos(x)                      # cos函數
```
- 基本線條
```python
plt.plot(x, y)                      
plt.show()
```
- 設定線條寬度
```python
plt.plot(x, y1, lw = 2)             # 線條寬度是 2
plt.plot(x, y2, linewidth = 5)      # 線條寬度是 5                
plt.show()
```
- 設定線條顏色
```python
plt.plot(x, y1, color='c')          # 設定青色cyan            
plt.plot(x, y2, color='r')          # 設定紅色red
plt.show()
```
- legend()
```python
plt.plot(x, y1, label='Sin')                    
plt.plot(x, y2, label='Cos')
plt.legend()
plt.grid()                          # 顯示格線
plt.show()
```
### 線條的樣式
```python
import matplotlib.pyplot as plt

d1 = [1, 2, 3, 4, 5, 6, 7, 8]           
d2 = [1, 3, 6, 10, 15, 21, 28, 36]     
d3 = [1, 4, 9, 16, 25, 36, 49, 64]     
d4 = [1, 7, 15, 26, 40, 57, 77, 100]  

plt.plot(d1, linestyle = 'solid')       # 預設實線
plt.plot(d2, linestyle = 'dotted')      # 虛點樣式
plt.plot(d3, ls = 'dashed')             # 虛線樣式
plt.plot(d4, ls = 'dashdot')            # 虛線點樣式
plt.show()
```
### 節點的樣式
```python
import matplotlib.pyplot as plt

d1 = [1, 2, 3, 4, 5, 6, 7, 8]           
d2 = [1, 3, 6, 10, 15, 21, 28, 36]      
d3 = [1, 4, 9, 16, 25, 36, 49, 64]      
d4 = [1, 7, 15, 26, 40, 57, 77, 100]    

seq = [1, 2, 3, 4, 5, 6, 7, 8]
plt.plot(seq,d1,'-',marker='x')
plt.plot(seq,d2,'-',marker='o')
plt.plot(seq,d3,'-',marker='^')
plt.plot(seq,d4,'-',marker='s') 
plt.show()
```
- [matplotlib.markers](https://matplotlib.org/stable/api/markers_api.html)

### 標題 | x 軸 | y 軸
```python
import matplotlib.pyplot as plt

temperature = [23, 22, 20, 24, 22, 22, 23, 20, 17, 18,
               20, 20, 16, 14, 14, 20, 20, 20, 15, 14,
               14, 14, 14, 16, 16, 16, 18, 21, 21, 20,
               16]
x = [x for x in range(1,len(temperature)+1)]        
plt.plot(x, temperature)
plt.title("Weather Report", fontsize=24)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()
```
## [scatter](https://matplotlib.org/stable/plot_types/basic/scatter_plot.html#sphx-glr-plot-types-basic-scatter-plot-py) 
```
import matplotlib.pyplot as plt
import numpy as np

# make the data
np.random.seed(3)
x = 4 + np.random.normal(0, 2, 24)
y = 4 + np.random.normal(0, 2, len(x))

# size and color:
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
```
## 產生滿足機率分布的直方圖
```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

# 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
normal_samples = np.random.normal(size = 100000) 

# 生成 100000 組介於 0 與 1 之間均勻分配隨機變數
uniform_samples = np.random.uniform(size = 100000) 

plt.hist(normal_samples)
plt.show()
plt.hist(uniform_samples)
plt.show()
```


## 多表並陳(subplot、subplots)
- 程式範例 [建立多個子圖表 ( subplot、subplots )](https://steam.oxxostudio.tw/category/python/example/matplotlib-subplot.html)

- [subplot(row, column, index)](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplot.html)
```python
import matplotlib.pyplot as plt
x = [1,2,3,4,5]
y = [5,4,3,2,1]
fig = plt.figure()
plt.subplot(221)
plt.plot(x)
plt.subplot(224)
plt.plot(y)
plt.show()
```
- [matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots)
```python
import matplotlib.pyplot as plt
x = [1,2,3,4,5]
y = [5,4,3,2,1]
fig, ax = plt.subplots(2,2)
ax[0][0].plot(x)
ax[1][1].plot(y)
plt.show()
```
# 2
## seaborn data visualization 資料視覺化
- [User guide and tutorial](https://seaborn.pydata.org/tutorial.html)
- [Example gallery](https://seaborn.pydata.org/examples/index.html)
- [An introduction to seaborn](https://seaborn.pydata.org/tutorial/introduction.html#multivariate-views-on-complex-datasets)

### 參考資料:[[第 19 天] 資料視覺化（2）Seaborn](https://ithelp.ithome.com.tw/articles/10186624)
```
直方圖（Histogram）:distplot() 方法
散佈圖（Scatter plot）:joinplot() 方法
線圖（Line plot）:factorplot() 方法
長條圖（Bar plot）:countplot() 方法
盒鬚圖（Box plot）:boxplot() 方法
```
```python
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
sns.distplot(normal_samples)
```
```python
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

sns.jointplot(x = "speed", y = "dist", data = cars_df)
```
```python
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]

cars_df = pd.DataFrame(
    {"speed": speed,
     "dist": dist
    }
)

sns.factorplot(data = cars_df, x="speed", y="dist", ci = None)
```
```python
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cyl = [6 ,6 ,4 ,6 ,8 ,6 ,8 ,4 ,4 ,6 ,6 ,8 ,8 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,4 ,8 ,8 ,8 ,8 ,4 ,4 ,4 ,8 ,6 ,8 ,4]
cyl_df = pd.DataFrame({"cyl": cyl})

sns.countplot(x = "cyl", data=cyl_df)
```
```python
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

normal_samples = np.random.normal(size = 100000) # 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
sns.boxplot(normal_samples)
```
# 3
- [Plotly: Data Apps for Production](https://plotly.com/)
  - [Plotly Open Source Graphing Library for Python](https://plotly.com/python/)
  - [示範](https://plotly.com/python/)

### 範例學習
- 參考資料 [Creating and Updating Figures in Python](https://plotly.com/python/creating-and-updating-figures/)
- 作業練習:[[Day 22] Python 視覺化解釋數據 - Plotly Expres](https://ithelp.ithome.com.tw/articles/10277258)

- Figures As Dictionaries
```python
!pip install plotly

fig = dict({
    "data": [{"type": "bar",
              "x": [1, 2, 3],
              "y": [1, 3, 2]}],
    "layout": {"title": {"text": "A Figure Specified By Python Dictionary"}}
})

# To display the figure defined by this dict, use the low-level plotly.io.show function
import plotly.io as pio
pio.show(fig)
```
- Figures as Graph Objects

```python
import plotly.graph_objects as go

fig = go.Figure(
    data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
    layout=go.Layout(
        title=go.layout.Title(text="A Figure Specified By A Graph Object")
    )
)

fig.show()
```


```python
import plotly.graph_objects as go

dict_of_fig = dict({
    "data": [{"type": "bar",
              "x": [1, 2, 3],
              "y": [1, 3, 2]}],
    "layout": {"title": {"text": "A Figure Specified By A Graph Object With A Dictionary"}}
})

fig = go.Figure(dict_of_fig)

fig.show()
```
- Converting Graph Objects To Dictionaries and JSON

```python
import plotly.graph_objects as go

fig = go.Figure(
    data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
    layout=go.Layout(height=600, width=800)
)

fig.layout.template = None # to slim down the output

print("Dictionary Representation of A Graph Object:\n\n" + str(fig.to_dict()))
print("\n\n")
print("JSON Representation of A Graph Object:\n\n" + str(fig.to_json()))
print("\n\n")
```

# 4
## bokeh參考資料
- [Bokeh documentation](https://docs.bokeh.org/en/latest/)
- https://docs.bokeh.org/en/latest/docs/user_guide/basic/scatters.html

```python
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
output_notebook()

from bokeh.plotting import figure, show

p = figure(width=400, height=400)

# add a circle renderer with a size, color, and alpha
p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)

# show the results
show(p)
```
```python
from bokeh.plotting import Histogram, show
import numpy as np

# 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
normal_samples = np.random.normal(size = 100000)


hist = Histogram(normal_samples)
show(hist)
```
