# 信用贷款违约风险预测
项目简述：依据客户的信用卡信息、分期付款信息、信用局信息等数据预测客户贷款是否会违约。
训练集共30W+条记录，测试集接近5W条记录，数据特征主要包含用户的个人属性，包括性别、职业、是否有车、是否有房、房子面积、家庭信息、贷款金额等信息。

## 1.初识数据
项目第一步：认识数据
```python
# 项目：信用贷款违约风险预测

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1.初识数据
application_train = pd.read_csv('application_train.csv')
application_test = pd.read_csv('application_test.csv')
print("application_train.shape:",application_train.shape)
print("application_test.shape:",application_test.shape)
print("application_train demo:")
print(application_train.head())
print("application_test demo:")
print(application_test.head())
```
结果：<br>
application_train.shape: (307511, 122)<br>
application_test.shape: (48744, 121)<br>
![初识数据](https://img-blog.csdnimg.cn/20200518122049729.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

当然，测试集没有是否违约这一列；而且看到有NaN。编辑器选的不好结果就会展现不完全，看一下训练集和测试集截图吧：<br>
![训练集初始数据](https://img-blog.csdnimg.cn/20200518122753760.png)<br>
![测试集初始数据](https://img-blog.csdnimg.cn/20200518123312944.png)<br>
可以看到目标变量在第二列”TARGET“，数据处理存在NaN，还有”N“”Y"“M”“F"等符号表示一定含义的特征值，在数据处理阶段需要进行编码。<br>
好了，认识数据之后，开始进行数据质量探索咯~<br>
## 2.数据质量探索
数据质量探索主要针对缺失值和异常值进行检测<br>
### 2.1 合并训练集和测试集
数据处理之前最好将训练集和测试集拼接在一块处理，减少工作量。<br>
```python
# 2.数据质量探索（异常值、缺失值）
# 2.1 将训练集和测试集拼接在一块处理，减少工作量
n_train = application_train.shape[0]
n_test = application_test.shape[0]
y_train = application_train['TARGET']
all_data = pd.concat([application_train,application_test],axis=0)
print("all data shape:{}".format(all_data.shape))
```
结果显示：all data shape:(356255, 122)
### 2.2 缺失值检测
```python
# 2.2 缺失值检测
# 封装一个缺失值检测函数
def missing_values_table(df):
    mis_val = df.isnull().sum() # 总缺失量
    mis_val_percent = 100 * mis_val/len(df) # 缺失值百分比
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) # 将缺失值整理为一个新的表
    mis_val_table_remane = mis_val_table.rename(columns = {0: 'MissingValues', 1: 'PercentMissingValues'})  # 重新命名缺失值表的列名
    mis_val_table_remane = mis_val_table_remane[mis_val_table_remane.iloc[:,1] != 0].sort_values('PercentMissingValues', ascending=False).round(1) # 按照第二列排序，且不含缺失率=0的
    print("Your selected dataframe has "+str(df.shape[1]) + "columns. \n There are" + str(mis_val_table_remane.shape[0]) + "columns that have missing values.")
    return mis_val_table_remane
# 计算缺失率
all_data_missing = missing_values_table(application_train)
print(all_data_missing.head(10))
# 2.2 缺失值处理（根据分析，本项目暂且不做缺失值处理）
```
结果显示：共67列特征含有缺失值。失率最高的是"COMMONAREA_MEDI"、“COMMONAREA_AVG”、“COMMONAREA_MODE”三列（均为房屋信息，缺失率也相同，可能是都没有房子信息造成同时缺失）缺失率为69.9%。打印一下缺失率表前10行如下：<br>
![缺失率](https://img-blog.csdnimg.cn/20200518172711599.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据对缺失率表进行分析发现：<br>
（1）53个特征都是与房屋相关的信息，缺失率完全相等，我们可以推断这些用户没有房子，没有这些数据，因此这里默认为nan不对其进行处理<br>
（2）剩余有缺失值的特征中，与贷款违约可能关联的特征有：OCCUPATION_TYPE（客户职业类型）缺失率最高为31.3%；AMT_GOODS_PRICE（消费者获得贷款的商品价格）缺失率0.1%<br>
OCCUPATION_TYPE（客户职业类型）缺失的原因分析：用户在输入时没有填写（没有职业或不想填写），这里不做任何处理，因为不管是使用众数还是均值填充，可能会改变模型预测的结果<br>
AMT_GOODS_PRICE（消费者获得贷款的商品价格）缺失率足够小，这里不做处理<br>
由于后续在特征工程中，需要剔除掉不相关的特征，所以这里挑选出缺失率高的特征进行处理，其他的特征缺失率较低、且在特征选择中可能被剔除掉，所以这里不做过多的分析和处理。<br>
### 2.3 异常值检测
根据个人经验，异常值主要是针对数值型数据进行检测，可以分为以下几种方法：<br>
（1）描述性统计<br>
（2）画箱线图<br>
（3）3sigma原则<br>
当然，这里需要对每一种特征含义进行解读，比如“用户年龄”，可以使用简单的描述性统计，如超过200岁，我们认为其异常；针对不同的特征选择不同的检验方法。<br>
通过对所有特征含义的分析，我将其分为用户基本信息和贷款信息两大类，接下来分别对两类信息的异常值进行检测。<br>
#### 第一：对用户基本信息进行检测<br>
用户基本信息包括：年龄、工作年限、车龄、拥有孩子数量、收入、房子信息，对于年龄、工作年限、车龄、拥有孩子数量这几个特征适合进行简单的描述性统计去检测异常值，这种方法快速简单，对于年龄、工作年限、车龄等
特征能够很好的检测，我们也不用花太多时间去处理。<br>
对于收入信息，不像年龄信息我们根据常识能判断异常值，所以对收入画箱线图进行异常值检测和处理。<br>
##### （1）用户年龄DAYS_BIRTH异常值检验
```python
# 2.3 异常值检测
# 2.3.1 用户基本信息检测：年龄/工作年限/车龄/拥有孩子数量/收入/房子信息
# 描述性统计：年龄/工作年限/车龄/拥有孩子数量/收入
# （1）年龄DAYS_BIRTH
print(all_data['DAYS_BIRTH'].head())  # 查看一下数据格式
```
结果显示如下图，发现所有的数据都是负数，然后查看数据描述文档，这里显示的是用户申请该笔贷款之前已经活了多少天，负号表示申请贷款前的时间。如“-9461”表示该用户申请贷款前已经生活了9461天<br>
![客户年龄](https://img-blog.csdnimg.cn/2020051817304323.png)

这里需要对年龄进行转换一下，我直接转化为以年为单位：<br>
```python
print((all_data['DAYS_BIRTH'] / -365).describe())  # 去掉负号，且将天数转化为年
```
结果显示：共356255条记录，说明年龄没有缺失值。另外，用户的年龄均值为43.9；年龄最大的用户：69岁；年龄最小的用户20岁。可以认为没有异常值<br>
![客户年龄转化](https://img-blog.csdnimg.cn/2020051817320567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>

##### （2）工作年限DAYS_EMPLOYED异常值检验
先查看工作年限取值情况：<br>
```python
# （2）工作年限DAYS_EMPLOYED
print(all_data['DAYS_EMPLOYED'].head())
```
![工作年限](https://img-blog.csdnimg.cn/20200518173600872.png)<br>
同用户年龄处理方式相同：<br>
```python
print((all_data['DAYS_EMPLOYED'] / -365).describe())
```
结果：<br>
![工作年限描述](https://img-blog.csdnimg.cn/20200518173711256.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
同样发现没有缺失值。另外，工作年限最长是49年，最小值却是个负数，说明原表中是正数，我们来查看一番:<br>
```python
sort_DayEmployed = all_data['DAYS_EMPLOYED'].sort_values(ascending=False) # 降序工作年限
print(sort_DayEmployed.head(10)) 
```
![工龄取值](https://img-blog.csdnimg.cn/20200518173923857.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据结果，再查看一下原表发现，工作年限这一列存在一个正数取值“365243”，表示申请贷款之后算起工作年限为365243天，换算下来是1000年，很明显属于异常值。<br>
这里我们将“365243”异常值替换成“nan”，再来看看工作年限的分布情况：<br>
```python
all_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)           # 暂且使用nan替换掉异常值
(all_data['DAYS_EMPLOYED'] /(-365)).plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')
plt.ylabel('Ferquency')
plt.grid()
plt.show()
```
替换掉异常值之后，用户已工作的年限DAYS_EMPLOYED分布如下图所示，显然没有异常了：<br>
![工龄分布直方图](https://img-blog.csdnimg.cn/20200518174057597.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
可以看出，用户工作年限大部分处在0~10年间，0~5年最多。<br>
##### （3）车龄OWN_CAR_AGE异常值检验
```python
print(all_data['OWN_CAR_AGE'].head(10))
```
![车龄格式](https://img-blog.csdnimg.cn/20200518174306837.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
可以看出nan值很多，我们分析原因应该是没有车~~~<br>
```python
print(all_data['OWN_CAR_AGE'].describe())
```
![车龄描述性统计](https://img-blog.csdnimg.cn/20200518174357208.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
结果显示：用户最小车龄为0，认为其没有车；用户最大车龄91，根据刚才对用户年龄的统计，发现最大的用户为69岁，说明91为异常值。<br>
查看一下异常值“91”多不多：<br>
```python
# 处理车龄OWN_CAR_AGE异常值
all_data.reset_index(inplace=True)  # 重置一下索引，便于查找缺失值所在的位置
sort_OWN_CAR_AGE = all_data['OWN_CAR_AGE'].sort_values(ascending=False)
print(sort_OWN_CAR_AGE.head())
```
![车龄排序](https://img-blog.csdnimg.cn/20200518180739817.png)<br>
显然，91年的车龄有2条记录，还有一条记录是74，超过了用户最大年龄，也标为异常值。3条记录对于整体数据的影响不大，这里直接删除这3条记录：<br>
```python
for i in (321542,294131,271741): # 注意：顺序一定不能错，先删除索引大的行，对小索引没有影响
    all_data = all_data.drop(i,axis=0)
print(all_data.shape)
```
删除3条记录之后数据形状：(356252, 123)  #这里我新增加了一列索引哦<br>
##### （4）拥有孩子数量CNT_CHILDREN异常值检验
原理同上，这里就直接上代码：<br>
```python
print(all_data['CNT_CHILDREN'].head())
```
![孩子数](https://img-blog.csdnimg.cn/20200518183414227.png)<br>
```python
print(all_data['CNT_CHILDREN'].describe())
```
![孩子数描述](https://img-blog.csdnimg.cn/20200518183603706.png)<br>
用户拥有孩子最大孩子数为20，最小为0，我们认为其没有异常值。另外这里也可以看出该特征没有缺失值。<br>
##### （5）收入AMT_INCOME_TOTAL异常值检验
由于我们不能像判断年龄一样判断用户的收入，因此这里利用箱线图对异常值进行检测并处理。<br>
首先，我这里包装了一个箱线图检测异常值的函数：<br>
```python
# 2.4.1.2 箱线图：收入AMT_INCOME_TOTAL异常值检测
# （1）包装一个箱线图函数，剔除掉Ql-1.5(Qu-Ql)和Qu+1.5(Qu-Ql)以外的数据
def outliers(data, col_name):
    # data : 接收pandas数据
    # col_name : pandas列名
    def box_plot_outlier(data_ser):
        # data_ser：接收pandas.Series格式
        quantileSpace = 1.5 * (data_ser.quantile(0.75) - data_ser.quantile(0.25))  # 1.5倍的分位数间距
        outliersLOW = data_ser.quantile(0.25) - quantileSpace   # 正常值下边界
        outliersUp = data_ser.quantile(0.75) + quantileSpace
        rule_low = (data_ser < outliersLOW)  # 小于下边界的异常值
        rule_up = (data_ser > outliersUp)
        return (rule_low, rule_up),(outliersLOW,outliersUp)

    data_new = data.copy()
    data_series = data_new[col_name] # 某一列
    rule, value = box_plot_outlier(data_series)
    # 取异常值的索引：
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]] # 选择的某列数据的行数shape[0]，rule[0]是小于正常值的异常值，rule[1]是大于正常值的异常值
    print("Delete number is:{}".format(len(index)))
    data_new = data_new.drop(index,axis=0)
    data_new.reset_index(drop=True,inplace=True) # 删除了异常值后，其索引还在原表中，需要重置索引，将异常值索引删除掉
    print("Now column number is:{}".format(data_new.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]  # 正常值下边界索引
    outliersDataL = data_series.iloc[index_low]
    print("Description of data less than the lower bond is:")
    print(pd.Series(outliersDataL).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliersDataU = data_series.iloc[index_up]
    print("Description of data Larger than the upper bond is:")
    print(pd.Series(outliersDataU).describe())
    # 可视化
    fig,ax = plt.subplots(1,2,figsize=(10,8)) #fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)
    sns.boxplot(y=data[col_name], data=data, palette='Set3',ax=ax[0])  # palette是色系设置；ax=ax[0]表示将原数据画在左边（1行2列的画图窗口哟）
    sns.boxplot(y=data_new[col_name], data=data_new, palette='Set3',ax=ax[1]) # 将删除掉异常值的图画在右边
    plt.grid()
    plt.show()
    return data_new # 返回删除掉异常值的新数据
```
调用函数，对收入进行检测并处理：<br>
```python
# （2）检测收入AMT_INCOME_TOTAL异常值，并进行处理
all_data = outliers(all_data,'AMT_INCOME_TOTAL')
```
处理结果：<br>
![收入处理结果](https://img-blog.csdnimg.cn/20200518190747964.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据结果显示，共删除了16757条记录，剩余339495条记录。<br>
用户收入异常值处理前后箱线图变化如下：<br>
![箱线图](https://img-blog.csdnimg.cn/20200518190957670.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
#### 第二：对贷款信息基本信息进行检测<br>
贷款基本信息：贷款信用额AMT_CREDIT/贷款年金AMT_ANNUITY/获得贷款的商品价格AMT_GOODS_PRICE，这三个特征使用箱线图比较合适。<br>
```python
# 2.4.2 贷款信息检测：贷款信用额AMT_CREDIT/贷款年金AMT_ANNUITY/获得贷款的商品价格AMT_GOODS_PRICE
for col in ('AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE'):
    all_data = outliers(all_data,col)
print(all_data.shape)
```
对贷款信用额AMT_CREDIT异常值检测及处理结果如下：<br>
![贷款信用额描述](https://img-blog.csdnimg.cn/20200518191931449.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据结果，贷款信用额AMT_CREDIT共删除7477条记录。处理前后箱线图如下：<br>
![贷款信用额](https://img-blog.csdnimg.cn/2020051819162763.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
对贷款年金AMT_ANNUITY异常值检测及处理结果如下：<br>
![贷款年金](https://img-blog.csdnimg.cn/20200518192039893.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据结果，贷款年金AMT_ANNUITY共删除6245条记录。处理前后箱线图如下：<br>
![贷款年金箱线图](https://img-blog.csdnimg.cn/20200518192053390.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
对获得贷款的商品价格AMT_GOODS_PRICE异常值检测及处理结果如下：<br>
![获得贷款的商品价格](https://img-blog.csdnimg.cn/20200518192145231.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据结果，获得贷款的商品价格AMT_GOODS_PRICE共删除1606条记录。处理前后箱线图如下：<br>
![获得贷款商品价格箱线图](https://img-blog.csdnimg.cn/20200518192210569.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
再次查看一下数据集的形状：<br>
```python
print(all_data.shape)
```
结果显示：(324167, 123)<br>
至此，我们已经对数据主要特征的质量进行了探索，分别对缺失值和异常值进行了检测和处理。接下来，开始构建违约用户的画像！<br>

## 3.违约用户画像探索
该阶段的主要目标是查看违约用户和非违约用户的特征分布情况，对违约用户的画像建立一个基本的了解，为后续特征工程打下基础。比如数据集里有很多字段，包括性别、年龄、工作年限等，那么是
违约用户在性别上有没有差异？在年龄上有没有差异？查看数据分布情况，一方面能够帮助我们认识数据，另一方面能够对违约用户做一个大概的理解，有助于后续特征工程构建及建模。<br>

### 3.1 违约用户的性别差异
在对违约用户进行画像之前，将训练集和测试集分开来：<br>
```python
# 3.1 违约用户的性别CODE_GENDER差异
# 3.1.1 对违约用户进行画像之前，将训练集和测试集分开来
train_new = all_data[all_data['TARGET'].notnull()]  # TARGET不为空的则是训练集
train_y = train_new['TARGET'] # 保存下清洗完数据的预测变量，拟合模型时要用
test_new = all_data[~all_data['TARGET'].notnull()]  # TARGET为空的则是测试集
test_new = test_new.drop(['TARGET'], axis=1)
print('After data cleaning, train_new.shape:',train_new.shape)
print('After data cleaning, test_new.shape:',test_new.shape)
```
![清洗完数据的训练集和测试集](https://img-blog.csdnimg.cn/20200518223256821.png)<br>
可以看到清洗完数据之后，训练集为281211条记录；测试集为42956条记录。<br>
接下来可以进行用户画像了，为了减少重复性的工作，这里封装两个函数，来分别画出条形图和概率密度图，以便于对用户画像有一个直观的分析：<br>
```python
# 将画图操作封装为一个函数，接下来对用户特征可视化会更加便捷
# 第1个：条形图函数
def plot_stats(feature, label_rotation=False, horizontal_layout=True):
    temp = train_new[feature].value_counts() #value_counts()是查看某列有多少不同取值，是Series拥有的方法，返回的也是Series类型
    df1 = pd.DataFrame({feature:temp.index, 'Number of contracts':temp.values}) # 将选择的某列的不同取值存在df1表中
    # 计算每一个特征不同取值的数量分布、及每一种取值下违约用户的百分比（需要用分组聚合函数groupby)
    # 使用pandas中的groupby进行分组聚合时，若对需要聚合的单列使用双中括号，则输出时会带有列标签
    # 若对需要聚合的单列使用单中括号，则输出时不会带有列标签，末尾会单独输出一行属性列
    cal_perc = train_new[[feature, 'TARGET']].groupby([feature], as_index=False).mean()  # 对输入的feature列进行分组聚合
    cal_perc.sort_values(by = 'TARGET', ascending=False, inplace=True)  # 根据“TARGET”进行降序排列违约用户的百分比
    if(horizontal_layout):  # 如果horizontal_layout（垂直排列）为真
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6)) # 画图时两列
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x=feature, y="Number of contracts", data=df1) #画条形图（每一种类下的违约用户的百分比）
    if(label_rotation): # 如果坐标刻度标签旋转为真
        s.set_xticklabels(s.get_xticklabels(), rotation=90) # 将横坐标刻度标签旋转90度
    s = sns.barplot(ax= ax2, x=feature, y='TARGET', order=cal_perc[feature], data=cal_perc)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)  # tick_params()定义参数刻度线样式：axis='both'表示对横纵坐标都设置，默认也是both；which='major'表示设置主刻度线，默认也是major
    plt.grid()
    plt.show()

# 第2个：概率密度图函数
def plot_distritution(var):
    # 输入var: 特征，可以是列表
    i = 0
    t1 = train_new.loc[train_new['TARGET'] != 0]  # t1:是违约用户
    t0 = train_new.loc[train_new['TARGET'] == 0]  # t0:不是违约用户
    # 画图
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,2,figsize=(12,12))
    for feature in var:
        i += 1
        plt.subplot(2,2,i)
        sns.kdeplot(t1[feature], bw=0.5, label='TARGET = 1') # 核密度估计图(估计密度函数)
        sns.kdeplot(t0[feature], bw=0.5, label='TARGET = 0')
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid()
    plt.show()
```
统计违约用户中男性和女性的人数及其各自占比，直接调用画条形图函数实现即可：<br>
```python
# 3.1.2 违约用户的性别CODE_GENDER差异
plot_stats('CODE_GENDER')
```
![违约用户性别差异](https://img-blog.csdnimg.cn/20200518224221880.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>

根据条形图结果发现：男性用户较少，但是在违约用户中，男性占比大于10%，女性占比7%，说明男性比女性发生违约的几率高。<br>
### 3.2 用户的年龄与是否违约的关系
年龄属于连续型变量，因此不能使用条形图展示，这里使用概率密度图查看用户的年龄按照是否违约划分的概率密度图：<br>
```python
# 3.2 用户的年龄DAYS_BIRTH与是否违约的关系
plt.figure(figsize=(10,8))
# 非违约用户年龄的概率密度图
sns.kdeplot(train_new.loc[train_new['TARGET']==0,'DAYS_BIRTH']/(-365), label='target==0')
# 违约用户年龄的概率密度图
sns.kdeplot(train_new.loc[all_data['TARGET']==1,'DAYS_BIRTH']/(-365), label='target==1')
plt.xlabel('Age(year)')
plt.ylabel('Density')
plt.grid()
plt.show()
```
![用户年龄按照是否违约划分的密度分布](https://img-blog.csdnimg.cn/20200518224456169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据年龄按照是否违约分类的概率密度图，橙色线代表违约用户(TARGET=1)的年龄分布；蓝色线代表非违约用户(TARGET=0)的年龄分布。<br>
从图中可以看出，违约用户中，年轻群体分布更多，所以我们推断用户的年龄越小，违约的可能性越大。<br>
为证明我们的推断，进一步我们对年龄进行分组，查看每一年龄段用户群体的违约概率:<br>
```python
# 计算每一年龄段的用户违约率，并可视化
age_data = train_new[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / (-365)  # 新增一列，将年龄转化为以年为单位的格式
age_data['YEARS_BIRTH'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20,70,num=11)) # cut()表示按照参数bins设置的间隔进行分组/分割；np.linspace(20,70,num=11)表示将区间[20,70]分成10组（因为最小年龄20，最大年龄69）
age_groups = age_data.groupby('YEARS_BIRTH').mean()
plt.figure(figsize=(8,8))
plt.bar(age_groups.index.astype(str), 100*age_groups['TARGET'])
plt.xticks(rotation = 75)
plt.xlabel('Age Group(years)')
plt.ylabel('Failure to Repay(%)')
plt.grid()
plt.show()
```
![每年龄段违约率](https://img-blog.csdnimg.cn/20200518231515306.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据每个年龄段用户违约率可以看出：确实是用户年龄越小，违约率越高，验证了上一步的推断。<br>

### 3.3 违约用户与贷款类型的关系
贷款类型属于离散型变量，直接调用条形图展示违约用户贷款类型的分布情况：<br>
```python
# 3.3 违约用户与贷款类型NAME_CONTRACT_TYPE的关系
plot_stats('NAME_CONTRACT_TYPE')
```
![违约用户贷款类型](https://img-blog.csdnimg.cn/2020051908430336.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据用户贷款类型分布情况来看：现金贷款的违约率更高。<br>
### 3.4 违约用户与是否有车/是否有房的关系
对特征是否有车FLAG_OWN_CAR、是否有房FLAG_OWN_REALTY进行同样的操作：<br>
```python
# 3.4 违约用户与是否有车FLAG_OWN_CAR/是否有房FLAG_OWN_REALTY的关系
plot_stats('FLAG_OWN_CAR')
plot_stats('FLAG_OWN_REALTY')
```
![违约用户是否有车](https://img-blog.csdnimg.cn/20200519090330711.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
![违约用户是否有房](https://img-blog.csdnimg.cn/20200519090511214.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据用户是否有车/是否有房的分布情况来看：没有房和没有车的用户违约率略高，但差异不是很大。<br>
### 3.5 违约用户与家庭类型的关系
```python
# 3.5 违约用户与家庭类型NAME_FAMILY_STATUS的关系
plot_stats('NAME_FAMILY_STATUS',True,True)
```
![违约用户家庭类型分布](https://img-blog.csdnimg.cn/2020051909490119.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据用户的家庭分布情况来看：申请贷款的用户大部分为已婚人士；单身和世俗结婚的违约率比较高；寡居的违约率最低。<br>

### 3.6 违约用户与家庭孩子数量的关系
```python
# 3.6 违约用户与家庭孩子数量CNT_CHILDREN的关系
plot_stats('CNT_CHILDREN')
```
![孩子数量分布](https://img-blog.csdnimg.cn/20200519095715847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据孩子分布情况来看：拥有孩子数量越多，违约率越高。<br>
当然，其中孩子数量为9和11的用户违约率达到了100%，与样本数量太少可能有关。<br>

### 3.7 违约用户与收入类型的关系
```python
# 3.7 违约用户与收入类型NAME_INCOME_TYPE的关系
plot_stats('NAME_INCOME_TYPE',False,False)
```
![收入类型](https://img-blog.csdnimg.cn/20200519100345329.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据用户收入类型分布情况来看：申请贷款的用户中没有工作的用户量很少，但是其违约率却最高，因此对该类用户放款应谨慎。<br>

### 3.8 违约用户与职业类型的关系
```python
# 3.8 违约用户与职业类型OCCUPATION_TYPE的关系
plot_stats('OCCUPATION_TYPE',True,True)
```
![职业类型](https://img-blog.csdnimg.cn/20200519100755414.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据用户职业类型分布情况来看：申请贷款用户中职业类型为廉价劳动力的用户很少，但是其违约率最高；同时职业收入相对较低、不稳定的用户违约率更高，如：廉价劳动力、司机、理发师等，像会计、高技术员工等违约率较低。<br>

### 3.9 违约用户与学历的关系
```python
# 3.9 违约用户与学历NAME_EDUCATION_TYPE的关系
plot_stats('NAME_EDUCATION_TYPE',True)
```
![学历](https://img-blog.csdnimg.cn/20200519101237631.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据用户的学历分布情况来看： 贷款申请人学历大部分为中学，学历越低违约率越高。<br>

## 4.特征工程
特征工程主要包括三方面：<br>
（1）对离散型变量的取值进行编码：Label Encoding / One-Hot Encoding<br>
（2）特征选择：Filter过滤 / Wrapper包装 / Embedded嵌入 <br>
（3）特征抽取：增加新的特征<br>
### 4.1 编码
根据违约用户画像，这里对影响较大的离散特征进行编码，主要包括：性别、是否有车、是否有房、贷款类型、职业、
学历、家庭状况(单身/已婚/...)、住房情况(租房/与父母同住/...)<br>
```python
# 4. 特征工程
# 4.1 特征取值编码
cols = ('CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_CONTRACT_TYPE','OCCUPATION_TYPE',
        'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE')
for feature in cols:
    LE1 = LabelEncoder()
    LE1.fit(list(all_data[feature].values)) # 将每一列的特征取值作为一个列表塞进编码字典中进行编码
    all_data[feature] = LE1.transform(list(all_data[feature].values)) # 将每一列特征值转化为编码字典中的索引，对应fit就能知道每一个值编码后的结果
```
编码完成，我们看几列数据检验一下是否编码成功：<br>
![性别](https://img-blog.csdnimg.cn/20200519110914396.png)
![车和房](https://img-blog.csdnimg.cn/20200519111053889.png)<br>
OK，编码成功！

### 4.2 特征选择
特征选择这一步，根据常识，我们先删除不必要的离散型特征，减少编码工作的同时，降低模型的复杂度。<br>
```python
# 4.2 特征选择
drop_feature = ('WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','NAME_TYPE_SUITE',
                'FONDKAPREMONT_MODE','EMERGENCYSTATE_MODE')
for f1 in drop_feature:
    all_data = all_data.drop([f1], axis=1)
print("all data shape:{}".format(all_data.shape))
```
OK删除了5列，剩余118列：all data shape:(324167, 118)<br>
接下来，通过相关系数矩阵删除掉相关系数>0.9的特征：<br>
```python
# 计算变量间相关性，剔除相关系数大于0.9的特征
train_y_new = all_data['TARGET']  # 对all_data不断进行更新中，更新下目标变量的存储
all_data_new = all_data.drop(['TARGET'], axis=1)
threshold = 0.9
corr_matrix = all_data_new.corr().abs()  # 计算相关系数矩阵
print(corr_matrix.head())
```
![相关系数](https://img-blog.csdnimg.cn/20200519113342550.png)<br>
```python
# 选择上三角阵处理
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)) # np.triu()获取上三角矩阵，k表示对角线起始位置
print(upper.head())
```
![上三角阵](https://img-blog.csdnimg.cn/20200519113632106.png)<br>
```python
# 删除系数大于0.9的特征
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %s columns to remove.'%(len(to_drop)))
all_data_new = all_data_new.drop(columns=to_drop)
print("all_data_New shape:{}".format(all_data_new.shape))
```
结果显示删除了34个特征：<br>
![处理结果](https://img-blog.csdnimg.cn/20200519114038748.png)<br>

### 4.3 特征抽取
根据对数据集及用户画像结果的分析，现在建立如下特征：<br>
（1）Credit_Income_Percent：贷款金额/客户收入。预期该值越大，用户违约的可能性越大<br>
（2）Annuity_Income_Percent：每年还款金额/客户收入。预期该值越大，用户违约的可能性越大<br>
（3）Credit_Term：每年还款金额/贷款金额，指还款周期，预期还款周期越小，用户短期资金压力较大，违约的可能性越大<br>
（4）Days_Employed_Percent：用户工作时间/用户年龄<br>
（5）Income_Per_Child：客户收入/孩子数量。如果家庭孩子较多，违约的可能性可能更高<br>
```python
#  4.3 特征抽取
all_data_copy = all_data.copy()  # 后续想对比一下特征选择前后模型的误差，因此这里将经过特征选择的数据和未经过特征选择的数据分开做
all_data_new_copy = all_data_new.copy()
#  （1）Credit_Income_Percent：贷款金额/客户收入
all_data_copy['Credit_Income_Percent'] = all_data_copy['AMT_CREDIT'] / all_data_copy['AMT_INCOME_TOTAL']
all_data_new_copy['Credit_Income_Percent'] = all_data_new_copy['AMT_CREDIT'] / all_data_new_copy['AMT_INCOME_TOTAL']
#  （2）Annuity_Income_Percent：每年还款金额/客户收入
all_data_copy['Annuity_Income_Percent'] = all_data_copy['AMT_ANNUITY'] / all_data_copy['AMT_INCOME_TOTAL']
all_data_new_copy['Annuity_Income_Percent'] = all_data_new_copy['AMT_ANNUITY'] / all_data_new_copy['AMT_INCOME_TOTAL']
# （3）Credit_Term：每年还款金额/贷款金额
all_data_copy['Credit_Term'] = all_data_copy['AMT_ANNUITY'] / all_data_copy['AMT_CREDIT']
all_data_new_copy['Credit_Term'] = all_data_new_copy['AMT_ANNUITY'] / all_data_new_copy['AMT_CREDIT']
# （4）Days_Employed_Percent：用户工作时间/用户年龄
all_data_copy['Days_Employed_Percent'] = all_data_copy['DAYS_EMPLOYED'] / all_data_copy['DAYS_BIRTH']
all_data_new_copy['Days_Employed_Percent'] = all_data_new_copy['DAYS_EMPLOYED'] / all_data_new_copy['DAYS_BIRTH']
# （5）Income_Per_Child：客户收入/孩子数量
all_data_copy['Income_Per_Child'] = all_data_copy['AMT_INCOME_TOTAL'] / all_data_copy['CNT_CHILDREN']
all_data_new_copy['Income_Per_Child'] = all_data_new_copy['AMT_INCOME_TOTAL'] / all_data_new_copy['CNT_CHILDREN']
```
特征抽取完成，接下来看看新特征与用户是否违约的关系：<br>
```python
# 查看新特征与是否违约的关系（分布图）
plt.figure(figsize=(12,20))
for n, newFeature in enumerate(['Credit_Income_Percent','Annuity_Income_Percent',
                             'Credit_Term','Days_Employed_Percent','Income_Per_Child']):
    plt.subplot(5,1,n+1)
    sns.kdeplot(all_data_copy.loc[all_data_copy['TARGET']==0, newFeature], label='target==0') # 非违约用户
    sns.kdeplot(all_data_copy.loc[all_data_copy['TARGET']==1, newFeature], label='target==1')
    plt.title('Distribution of %s by Target Value'%newFeature)
    plt.xlabel('%s'%newFeature)
    plt.ylabel('Density')
    plt.grid()
plt.tight_layout(h_pad=2.5)
plt.show()
```
![新特征密度分布](https://img-blog.csdnimg.cn/20200519122731651.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
从上图可以看到，除了Credit_Term：每年还款金额/贷款金额和Days_Employed_Percent：用户工作时间/用户年龄两个特征有点差异之外，
其他特征与用户是否违约之间的关系不是很大哈。不过没关系，属于正常现象。之后我们可以将新特征放到模型中，对比一下模型前后的误差。<br>
至此，特征工程就完成了！

## 5.建模及预测
经过数据清洗和特征工程步骤后，我们得到数据集包括：<br>
（1）all_data：异常值和缺失值、编码<br>
（2）all_data_new：异常值和缺失值、编码、特征选择<br>
（3）all_data_copy：异常值和缺失值、编码、特征抽取<br>
（4）all_data_new_copy：异常值和缺失值、编码、特征选择、特征抽取<br>
本节所有的操作我们都针对以上几组情况进行对比<br>
这里我选择lgb模型，将模型训练过程封装在一个函数中：<br>
```python
def model(features, test_features, encoding='ohe', n_folds=10):
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    # Extract the labels for training
    labels = features['TARGET']

    # Remove the ids and target
    features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns=['SK_ID_CURR'])

    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join='inner', axis=1)

        # No categorical indices to record
        cat_indices = 'auto'

    # Integer label encoding
    elif encoding == 'le':

        # Create a label encoder
        label_encoder = LabelEncoder()

        # List for storing categorical indices
        cat_indices = []

        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)

    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    # Extract feature names
    feature_names = list(features.columns)

    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        # Create the model
        model = lgb.LGBMClassifier(n_estimators=1000, objective='binary',
                                   class_weight='balanced', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   subsample=0.8, n_jobs=-1, random_state=50)

        # Train the model
        model.fit(train_features, train_labels, eval_metric='auc',
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], categorical_feature=cat_indices,
                  early_stopping_rounds=100, verbose=200)

        # Record the best iteration
        best_iteration = model.best_iteration_

        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits

        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration=best_iteration)[:, 1]

        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return submission, feature_importances, metrics
```
这里先对all_data中训练集的模型进行训练：<br>
```python
# 调用函数
submission, fi, metrics1 = model(train_1, test_1)
```
![训练1](https://img-blog.csdnimg.cn/2020051914405126.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)

![训练11](https://img-blog.csdnimg.cn/20200519144159983.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70))<br>
打印一下训练结果：<br>
```python
print('Baseline metrics1')
print(metrics1)
```
![训练11结果](https://img-blog.csdnimg.cn/20200519144244314.png)<br>
```python
submission2, fi2, metrics2 = model(train_2, test_2)print('Baseline metrics2')
print(metrics2)

```
Training Data Shape:  (281211, 96)<br>
Testing Data Shape:  (42956, 96)<br>
...<br>
![训练2](https://img-blog.csdnimg.cn/20200519150023530.png)<br>
```python
submission3, fi3, metrics3 = model(train_3, test_3)
print('Baseline metrics3')
print(metrics2)
```
Training Data Shape:  (281211, 135)<br>
Testing Data Shape:  (42956, 135)<br>
...<br>
![训练3结果](https://img-blog.csdnimg.cn/20200519152722846.png)<br>
```python
submission4, fi4, metrics4 = model(train_4, test_4)
print('Baseline metrics4')
print(metrics4)
```
Training Data Shape:  (281211, 101)<br>
Testing Data Shape:  (42956, 101)<br>
...<br>
![训练4结果](https://img-blog.csdnimg.cn/20200519163523894.png)<br>
对比以上四种情况可以看出：同时进行了特征选择和特征抽取的训练集模型auc84%，验证集77%，在以上四种情况中均为最高的情况，性能最好，说明我们的特征工程是有意义的。<br>
接下来看看特征的重要程度情况：<br>
```python
fi_sorted4 = plot_feature_importances(fi4)
```
![特征重要性](https://img-blog.csdnimg.cn/20200519164016780.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Ffemh1b18=,size_16,color_FFFFFF,t_70)<br>
根据特征的重要性可以看出：Credit_Term每年还款金额/贷款金额与是否违约的关系较强，该特征是我们在特征工程阶段新增加的，这里验证了抽取的特征，得出Credit_Term与用户是否违约的关系缺失比较大。<br>
预测结果已经存入了submission4表中，我们来看看前5行：<br>
```python
print(submission4.head())
```
![预测结果](https://img-blog.csdnimg.cn/20200519170722482.png)<br>
'TARGET'结果表示预测为违约用户的概率。一般情况下，我们设置阈值为0.5，大于0.5被认为是违约用户；小于0.5被认为是非违约用户。<br>
至此，信用贷款违约预测项目就结尾了，现在总结一下：<br>
（1）初识数据<br>
（2）数据质量探索、清洗脏数据<br>
（3）特征工程：编码、特征选择、特征抽取<br>
（4）建模及预测<br>
Ending...<br>

