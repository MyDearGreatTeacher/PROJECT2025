# 舊版資料
- https://github.com/MyDearGreatTeacher/2022_1_courses/tree/main/%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E5%B0%8E%E8%AB%96
- 課程內容
  - 1.pandas的資料結構:series vs DataFrame
  - 2.[資料匯入與清洗(data cleaning)](#2)
  - 3.資料分析:連接_合併和重塑
    - [2023教材](https://github.com/MyDearGreatTeacher/ML202302/blob/main/%E8%B3%87%E6%96%99%E7%A7%91%E5%AD%B8/3_3_%E8%B3%87%E6%96%99%E8%99%95%E7%90%86_%E9%80%A3%E6%8E%A5_%E5%90%88%E4%BD%B5%E5%92%8C%E9%87%8D%E5%A1%91.md) 
  - 4.pandas存取SQLITE資料庫
    - [[Pandas教學]快速掌握Pandas套件讀寫SQLite資料庫的重要方法](https://www.learncodewithmike.com/2021/05/pandas-and-sqlite.html) 
  - 5.進階分析:資料聚合和分組
    - [2023教材](https://github.com/MyDearGreatTeacher/ML202302/blob/main/%E8%B3%87%E6%96%99%E7%A7%91%E5%AD%B8/3_4_%E8%B3%87%E6%96%99%E8%81%9A%E5%90%88%E5%92%8C%E5%88%86%E7%B5%84.md)
   - 6.pandas資料分析小專題:股票分析
     - 參考經典教材 第十三章 資料分析範例
# pandas
- 經典教材[Python 資料分析, 3/e ](https://www.tenlong.com.tw/products/9786263244177?list_name=srh)
  - [GITHUB](https://github.com/wesm/pydata-book) 
  - [中譯](https://github.com/LearnXu/pydata-notebook/tree/master/)
  - 第十三章 資料分析範例
    - 14.1 USA.gov Data from Bitly（USA.gov資料集）
    - 14.2 MovieLens 1M Dataset（MovieLens 1M資料集）
    - 14.3 US Baby Names 1880–2010（1880年至2010年美國嬰兒姓名）
    - 14.4 USDA Food Database（USDA食品資料庫）
    - 14.5 2012 Federal Election Commission Database（2012聯邦選舉委員會資料庫）
- [Pandas 資料分析實戰：使用 Python 進行高效能資料處理及分析 (Learning pandas : High-performance data manipulation and analysis in Python, 2/e) Michael Heydt ](https://www.tenlong.com.tw/products/9789864343898)
  - [GITHUB](https://github.com/PacktPublishing/Learning-Pandas-Second-Edition) 
  - 一定要教  Chapter 15：歷史股價分析
- [Python 資料分析必備套件！Pandas 資料清理、重塑、過濾、視覺化 (Pandas 1.x Cookbook, 2/e) Matt Harrison、Theodore Petrou](https://www.tenlong.com.tw/products/9789863126898?)
- [深入淺出 Pandas：利用 Python 進行數據處理與分析 李慶輝　著](https://www.tenlong.com.tw/products/9787111685456)

## 1_pandas Data Structures| pandas的資料結構 
- series vs DataFrame


### 1_1_series的運算 see [CHAPTER 5 Getting Started with pandas](https://github.com/LearnXu/pydata-notebook/blob/master/Chapter-05/5.1%20Introduction%20to%20pandas%20Data%20Structures%EF%BC%88pandas%E7%9A%84%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%EF%BC%89.ipynb)
- 建立series
  - 使用pandas.Series() 
  - 使用字典資料型態傳入pandas.Series() 
- 搜尋滿足條件的資料


#### (1)使用[pandas.Series()](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) 建立Series
```python
import pandas as pd

obj = pd.Series([41, 27, -25, 13,21])
obj
```


```python
obj.values
```


```python
obj.index
```

#### (2)使用[pandas.Series()](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) 建立Series
```python
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
```
```python
obj2.values
```


```python
obj2.index
```

#### 透過index(索引)進行搜尋

```python
obj2['a']
```


```python
obj2[['c', 'a', 'd']]
```
#### 透過index(索引)進行設定

```python
obj2['d'] = 6
```


```python
obj2[['c', 'a', 'd']]
```
#### 更多運算
```python
obj2[obj2 > 0]
```


```python
obj2 * 2
```

```python
import numpy as np
np.exp(obj2)
```

#### 利用dict來創建series
```python
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon':16000, 'Utah': 5000}

obj3 = pd.Series(sdata)
obj3
```


```python
states = ['California', 'Ohio', 'Oregon', 'Texas']

obj4 = pd.Series(sdata, index=states)
obj4
```
#### 使用pandas的isnull和notnull函數檢測MISSING Value(缺失資料)
```python
pd.isnull(obj4)
```


```python
pd.notnull(obj4)
```

```python
obj4.isnull()
```

#### Series自動排序(Data alignment features)
- Series自動按index label來排序


```python
obj3
```

```python
obj4
```


```python
obj3 + obj4
```

#### series  `name屬性`的運用
- series自身和它的index都有一個叫name的屬性

```python
obj4.name = 'population'

obj4.index.name = 'state'

obj4
```
#### 更改 series的index

```python
obj
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj

```
#### 加分作業:研讀底下參考資料並完成實務演練

[Pandas 資料分析實戰：使用 Python 進行高效能資料處理及分析 (Learning pandas : High-performance data manipulation and analysis in Python, 2/e) Michael Heydt ](https://www.tenlong.com.tw/products/9789864343898)
  - [GITHUB](https://github.com/PacktPublishing/Learning-Pandas-Second-Edition) 
  - Chapter 3：用序列表示單變數資料
    - 3.1 設定pandas
    - 3.2 建立序列
    - 3.3 .index及.values屬性
    - 3.4 序列的大小及形狀
    - 3.5 在序列建立時指定索引
    - 3.6 頭、尾、選取
    - 3.7 以索引標籤或位置提取序列值
    - 3.8 把序列切割成子集合
    - 3.9 利用索引標籤實現對齊
    - 3.10 執行布林選擇
    - 3.11 將序列重新索引
    - 3.12 原地修改序列


## 1_2_DataFrame的運算 see [CHAPTER 5 Getting Started with pandas](https://github.com/LearnXu/pydata-notebook/blob/master/Chapter-05/5.1%20Introduction%20to%20pandas%20Data%20Structures%EF%BC%88pandas%E7%9A%84%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%EF%BC%89.ipynb)
- 建立DataFrame

- 搜尋滿足條件的資料

### (1)建立DataFrame: 使用python dict 資料型態 + dict裡的value是list 資料型態
```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'], 
        'year': [2000, 2001, 2002, 2001, 2002, 2003], 
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}

frame = pd.DataFrame(data)

frame
```

### 顯示資料的技術
```python
frame.head()
```


```python
pd.DataFrame(data, columns=['year', 'state', 'pop'])
```
### missing value
```python
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'], 
                      index=['one', 'two', 'three', 'four', 'five', 'six'])
frame2

frame2.columns
```

## 資料提取
- 提取一column(列)
- 提取一row(行)
### 從DataFrame提取一列(底下兩種提取方法)== > 回傳給你一個series
```python
frame2['state']
frame2.year
```
### 從DataFrame提取一row(行)

```python
frame2.loc['three']
```
### 資料變更(賦值)的幾種範例
```python
frame2['debt'] = 16.5
frame2
```

```python
frame2['debt'] = np.arange(6.)
frame2
```

```python
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2
## 特別注意結果
```

```python
frame2['eastern'] = frame2.state == 'Ohio'
frame2
```

#### 刪除資料
```python
del frame2['eastern']

frame2.columns
```

### (2)建立Data Frame的情境 == >
```python
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

frame3 = pd.DataFrame(pop)
frame3
```

```python

```
### (3)建立Data Frame的情境 == > 使用python dict + series

```python
pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
pd.DataFrame(pdata)
```


```python
frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3
frame3.values
frame2.values
```
#### 加分作業:研讀底下參考資料並完成實務演練

[Pandas 資料分析實戰：使用 Python 進行高效能資料處理及分析 (Learning pandas : High-performance data manipulation and analysis in Python, 2/e) Michael Heydt ](https://www.tenlong.com.tw/products/9789864343898)
  - [GITHUB](https://github.com/PacktPublishing/Learning-Pandas-Second-Edition) 
  - Chapter 4：用資料框表示表格及多變數資料
    - 4.1 設定pandas
    - 4.2 建立資料框物件
    - 4.3 存取資料框的資料
    - 4.4 利用布林選擇選取列
    - 4.5 跨越行與列進行選取
  - Chapter 5：操控資料框結構
    - 5.1 設定pandas
    - 行columns的各種運算
      - 5.2 重新命名行Renaming columns
      - 5.3 利用[]及.insert()增加新行
      - 5.4 利用擴展增加新行
      - 5.5 利用串連增加新行
      - 5.6 改變行的順序
      - 5.7 取代行的內容
      - 5.8 刪除行
    - 列columns的各種運算
      - 5.9 附加新列
      - 5.10 列的串連
      - 5.11 經由擴展增加及取代列
      - 5.12 使用.drop()移除列
      - 5.13 利用布林選擇移除列
      - 5.14 使用切割移除列

# 2.
- pandas資料匯入與資料清理(Data cleaning)
- [經典:Python 資料分析, 2/e](https://www.tenlong.com.tw/products/9789864769254)
  - [GITHUB](https://github.com/wesm/pydata-book) 
  - [中譯](https://github.com/LearnXu/pydata-notebook/tree/master/)
  - 第六章 資料載入、儲存和檔案格式
  - 第七章 資料整理和前處理
- [Pandas 資料分析實戰：使用 Python 進行高效能資料處理及分析 (Learning pandas : High-performance data manipulation and analysis in Python, 2/e) Michael Heydt ](https://www.tenlong.com.tw/products/9789864343898)
  - [GITHUB](https://github.com/PacktPublishing/Learning-Pandas-Second-Edition) 
  - [Ch9](https://github.com/PacktPublishing/Learning-Pandas-Second-Edition/blob/master/Chapter09/09_Accessing_Data.ipynb)
  - Chapter 9：存取資料
  - 9.1 設定pandas
  - 9.2 處理CSV及文字/表格格式的資料
  - 9.3 讀寫Excel格式資料
  - 9.4 讀寫JSON檔案
  - 9.5 從網站讀取HTML資料
  - 9.6 讀寫HDF5格式檔案
  - 9.7 存取網站上的CSV資料
  - 9.8 讀寫SQL資料庫
  - 9.9 從遠端資料服務讀取資料


## 延伸學習:存取資料庫
- [Pandas讀寫MySQL資料庫](https://codertw.com/%E8%B3%87%E6%96%99%E5%BA%AB/16156/)
- [使用Python從Mysql抓取每日股價資料與使用pandas進行分析](https://sites.google.com/site/zsgititit/home/python-cheng-shi-she-ji/shi-yongpython-congmysql-zhua-qu-mei-ri-gu-jia-zi-liao-yu-shi-yongpandas-jin-xing-fen-xi)
- [如何用Pandas連接到PostgreSQL資料庫讀寫數據](https://medium.com/@phoebehuang.pcs04g/use-pandas-link-to-postgresql-6cfc24a930f1)

## 1_讀寫CSV檔案 
- see 9.2 處理CSV及文字/表格格式的資料 
- [pandas.read_table](https://pandas.pydata.org/docs/reference/api/pandas.read_table.html)
- 讀取CSV [pandas.read_csv()](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
  - 熟悉各種讀取參數用法  [pandas.read_csv参数详解](https://www.cnblogs.com/datablog/p/6127000.html)
  - index_col:指定使用某欄位當作索引
  - nrows：僅讀取⼀定的⾏數
  - skiprows：跳過⼀定的⾏數
  - skipfooter：尾部有固定的⾏數永不讀取
  - skip_blank_lines：空⾏跳過
- 寫入CSV [pandas.DataFrame.to_csv()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html#pandas-dataframe-to-csv)

#### 先下載遠端資料到Google Colab 
```
!wget https://raw.githubusercontent.com/PacktPublishing/Learning-Pandas-Second-Edition/master/data/msft.csv
```
#### 檢視資料
```
!head -n 5 msft.csv 
```
#### Reading a CSV into a DataFrame
```
msft = pd.read_csv("./msft.csv")
msft[:5]
```

#### Specifying the index column when reading a CSV file
```python
# use column 0 as the index
msft = pd.read_csv("./msft.csv", index_col=0)
msft[:5]
```

```python
df3 = pd.read_csv("./msft.csv", usecols=['Date', 'Close'])
df3[:5]
```
#### 寫入CSV Saving a DataFrame to a CSV ==> [pandas.DataFrame.to_csv()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html#pandas-dataframe-to-csv)
```PYTHON
# read in data only in the Date and Close columns
# and index by the Date column
df2 = pd.read_csv("./msft.csv", 
                  usecols=['Date', 'Close'], 
                  index_col=['Date'])
df2[:5]
```
```python
# save df2 to a new csv file
# also specify naming the index as date
df2.to_csv("./msft_A999168.csv", index_label='date')
```
```
# view the start of the file just saved
!head -n 5 ./msft_A999168.csv
```

### 各式csv的讀取 
- 去除頭部說明文字
- 去除底部說明文字
- 讀取部分欄位
- 讀取部分資料
  - [參考程式 GITHUB](https://github.com/PacktPublishing/Learning-Pandas-Second-Edition) 

### 2_讀寫excel檔案 Reading and writing data in Excel format
- [下載excel檔案](https://github.com/PacktPublishing/Learning-Pandas-Second-Edition/blob/master/data/stocks.xlsx)
- 再upload到Google Colab
- xlsx vs xls 檔案差異
- 讀取excel [pandas.read_excel()](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html)
  - 熟悉各種讀取參數用法  
  - [(最新Pandas.read_excel()全参数详解（案例实操，如何利用python导入excel）)](https://zhuanlan.zhihu.com/p/142972462)
  - [[Pandas教學]5個實用的Pandas讀取Excel檔案資料技巧](https://www.learncodewithmike.com/2020/12/read-excel-file-using-pandas.html)
  - [Python pandas.ExcelWriter用法及代碼示例](https://vimsky.com/zh-tw/examples/usage/python-pandas.ExcelWriter.html)

- 寫入excel [pandas.DataFrame.to_excel()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html)
  - [Python pandas.DataFrame.to_excel用法及代碼示例](https://vimsky.com/zh-tw/examples/usage/python-pandas.DataFrame.to_excel.html)
```python
df = pd.read_excel("./stocks.xlsx")
df[:5]
```

### 讀取不同試算表
```python
# read from the aapl worksheet
aapl = pd.read_excel("./stocks.xlsx", sheet_name='aapl')
aapl[:5]
```

```python
# save to an .XLS file, in worksheet 'Sheet1'
df.to_excel("./stocks2.xls")
```


```python

```

### 3_讀寫 JSON 檔案
- [JSON](https://zh.wikipedia.org/wiki/JSON)
- XML vs JSON
- Reading and writing JSON files

```python

```
### 4_讀取網頁表格資料 
- [pandas.read_html()](https://pandas.pydata.org/docs/reference/api/pandas.read_html.html)
  - [[Pandas教學]掌握Pandas DataFrame讀取網頁表格的實作技巧](https://www.learncodewithmike.com/2020/11/read-html-table-using-pandas.html)  
- [pandas.DataFrame.to_html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_html.html)
  - [[Pandas教學]利用Pandas套件的to_html方法在網頁快速顯示資料分析結果](https://www.learncodewithmike.com/2021/07/pandas-to-html.html) 


# 6.股票分析

- [Pandas 資料分析實戰：使用 Python 進行高效能資料處理及分析 (Learning pandas : High-performance data manipulation and analysis in Python, 2/e) Michael Heydt ](https://www.tenlong.com.tw/products/9789864343898)
  - [GITHUB](https://github.com/PacktPublishing/Learning-Pandas-Second-Edition) 
  - [Ch15](https://github.com/PacktPublishing/Learning-Pandas-Second-Edition/blob/master/Chapter06/06_Working%20with%20Indexes.ipynb)
```
Chapter 15：歷史股價分析
15.1 設定IPython筆記本
15.2 從Google取得與組織股票資料
15.3 繪製股價時間序列的圖
15.4 繪製成交量序列的圖
15.5 計算簡易的每日收盤價變化百分比
15.6 計算簡易的股票每日累積報酬率
15.7 將每日報酬率重新取樣為每月報酬率
15.8 分析報酬率分布
15.9 移動平均計算
15.10 比較股票之間的平均每日報酬率
15.11 依每日收盤價的變化百分比找出股票相關性
15.12 計算股票波動率
15.13 決定風險相對於期望報酬率的關係
```
## 15.1 設定IPython筆記本
```python
# Commented out IPython magic to ensure Python compatibility.
# import numpy and pandas
import numpy as np
import pandas as pd

# used for dates
import datetime
from datetime import datetime, date

# Set formattign options
pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 60)

# bring in matplotlib for graphics
import matplotlib.pyplot as plt
# %matplotlib inline
```
## 要先更新版本

!pip install --upgrade pandas-datareader

## 15.2 從Google取得與組織股票資料
```
# import data reader package
import pandas_datareader as pdr

# read data from Yahoo! Finance for a specific 
# stock specified by ticker and between the start and end dates
def get_stock_data(ticker, start, end):
    # read the data
    data = pdr.data.DataReader(ticker, 'yahoo', start, end)

    # rename this column
    data.insert(0, "Ticker", ticker)
    return data

# request the three years of data for MSFT
start = datetime(2012, 1, 1)
end = datetime(2014, 12, 31)
get_stock_data("MSFT", start, end)[:5]

# gets data for multiple stocks
# tickers: a list of stock symbols to fetch
# start and end are the start end end dates
def get_data_for_multiple_stocks(tickers, start, end):
    # we return a dictionary
    stocks = dict()
    # loop through all the tickers
    for ticker in tickers:
        # get the data for the specific ticker
        s = get_stock_data(ticker, start, end)
        # add it to the dictionary
        stocks[ticker] = s
    # return the dictionary
    return stocks

# get the data for all the stocks that we want
raw = get_data_for_multiple_stocks(
    ["MSFT", "AAPL", "GE", "IBM", "AA", "DAL", "UAL", "PEP", "KO"],
    start, end)

# take a peek at the data for MSFT
raw['MSFT'][:5]
```
## 15.3 繪製股價時間序列的圖
```
# given the dictionary of data frames,
# pivots a given column into values with column
# names being the stock symbols
def pivot_tickers_to_columns(raw, column):
    items = []
    # loop through all dictionary keys
    for key in raw:
        # get the data for the key
        data = raw[key]
        # extract just the column specified
        subset = data[["Ticker", column]]
        # add to items
        items.append(subset)
    
    # concatenate all the items
    combined = pd.concat(items)
    # reset the index
    ri = combined.reset_index()
    # return the pivot
    return ri.pivot("Date", "Ticker", column)

# do the pivot
close_px = pivot_tickers_to_columns(raw, "Close")
# peek at the result
close_px[:5]

```
## Plotting time-series prices
```

# plot the closing prices of AAPL
close_px['AAPL'].plot();

# plot the closing prices of MSFT
close_px['MSFT'].plot();

# plot MSFT vs AAPL on the same chart
close_px[['MSFT', 'AAPL']].plot();

```
## Plotting volume series data
```
# pivot the volume data into columns
volumes = pivot_tickers_to_columns(raw, "Volume")
volumes.tail()

# plot the volume for MSFT
msft_volume = volumes[["MSFT"]]
plt.bar(msft_volume.index, msft_volume["MSFT"])
plt.gcf().set_size_inches(15,8)

# draw the price history on the top
top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
top.plot(close_px['MSFT'].index, close_px['MSFT'], 
         label='MSFT Close')
plt.title('MSFT Close Price 2012 - 2014')
plt.legend(loc=2)

# and the volume along the bottom
bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
bottom.bar(msft_volume.index, msft_volume['MSFT'])
plt.title('Microsoft Trading Volume 2012 - 2014')
plt.subplots_adjust(hspace=0.75)
plt.gcf().set_size_inches(15,8)
```

## Calculating simple daily percentage change
```
# calculate daily percentage change
daily_pc = close_px / close_px.shift(1) - 1
daily_pc[:5]

# check the percentage on 2012-01-05
close_px.loc['2012-01-05']['AAPL'] / \
    close_px.loc['2012-01-04']['AAPL'] -1

# plot daily percentage change for AAPL
daily_pc["AAPL"].plot();

"""# Calculating simple daily cumulative returns"""

# calculate daily cumulative return
daily_cr = (1 + daily_pc).cumprod()
daily_cr[:5]

# plot all the cumulative returns to get an idea 
# of the relative performance of all the stocks
daily_cr.plot(figsize=(8,6))
plt.legend(loc=2);

"""# Resampling data from daily to monthly returns"""

# resample to end of month and forward fill values
monthly = close_px.asfreq('M').ffill()
monthly[:5]

# calculate the monthly percentage changes
monthly_pc = monthly / monthly.shift(1) - 1
monthly_pc[:5]

# calculate monthly cumulative return
monthly_cr = (1 + monthly_pc).cumprod()
monthly_cr[:5]

# plot the monthly cumulative returns
monthly_cr.plot(figsize=(12,6))
plt.legend(loc=2);

"""# Analyzing distribution of returns"""

# histogram of the daily percentage change for AAPL
aapl = daily_pc['AAPL']
aapl.hist(bins=50);

# matrix of all stocks daily % changes histograms
daily_pc.hist(bins=50, figsize=(8,6));

"""# Performing moving average calculation"""

# extract just MSFT close
msft_close = close_px[['MSFT']]['MSFT']
# calculate the 30 and 90 day rolling means
ma_30 = msft_close.rolling(window=30).mean()
ma_90 = msft_close.rolling(window=90).mean()
# compose into a DataFrame that can be plotted
result = pd.DataFrame({'Close': msft_close, 
                       '30_MA_Close': ma_30,
                       '90_MA_Close': ma_90})
# plot all the series against each other
result.plot(title="MSFT Close Price")
plt.gcf().set_size_inches(12,8)

"""# Comparision of average daily returns across stocks"""

# plot the daily percentage change of MSFT vs AAPL
plt.scatter(daily_pc['MSFT'], daily_pc['AAPL'])
plt.xlabel('MSFT')
plt.ylabel('AAPL');

# demonstrate perfect correlation
plt.scatter(daily_pc['MSFT'], daily_pc['MSFT']);

from pandas.plotting import scatter_matrix
# plot the scatter of daily price changed for ALL stocks
scatter_matrix(daily_pc, diagonal='kde', figsize=(12,12));

"""# Correlation of stocks based upon daily percentage change of closing price"""

# calculate the correlation between all the stocks relative
# to daily percentage change
corrs = daily_pc.corr()
corrs

# plot a heatmap of the correlations
plt.imshow(corrs, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corrs)), corrs.columns)
plt.yticks(range(len(corrs)), corrs.columns)
plt.gcf().set_size_inches(8,8)

"""# Volatility"""

# 75 period minimum
min_periods = 75
# calculate the volatility
vol = daily_pc.rolling(window=min_periods).std() * \
        np.sqrt(min_periods)
# plot it
vol.plot(figsize=(10, 8));

"""# Determining risk relative to expected returns"""

# generate a scatter of the mean vs std of daily % change
plt.scatter(daily_pc.mean(), daily_pc.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')

# this adds fancy labels to each dot, with an arrow too
for label, x, y in zip(daily_pc.columns, 
                       daily_pc.mean(), 
                       daily_pc.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (30, -30),
        textcoords = 'offset points', ha = 'right', 
        va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', 
                    fc = 'yellow', 
                    alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', 
                          connectionstyle = 'arc3,rad=0'))

# set ranges and scales for good presentation
plt.xlim(-0.001, 0.003)
plt.ylim(0.005, 0.0275)

# set size
plt.gcf().set_size_inches(8,8)
```
