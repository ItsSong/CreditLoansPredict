# 项目：信用贷款违约风险预测

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



# 1.初识数据------------------------------------------------------------------------------
application_train = pd.read_csv('application_train.csv')
application_test = pd.read_csv('application_test.csv')
print("application_train.shape:",application_train.shape)
print("application_test.shape:",application_test.shape)
print("application_train demo:")
print(application_train.head())
print("application_test demo:")
print(application_test.head())


# 2.数据质量探索（异常值、缺失值）-----------------------------------------------------------
# 2.1 将训练集和测试集拼接在一块处理，减少工作量
n_train = application_train.shape[0]
n_test = application_test.shape[0]
y_train = application_train['TARGET']
all_data = pd.concat([application_train,application_test],axis=0)
print("all data shape:{}".format(all_data.shape))

# 2.2 缺失值检测及处理
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
# 2.3 缺失值处理（根据分析，本项目暂且不做缺失值处理）

# 2.4 异常值检测及处理
# 2.4.1 用户基本信息检测：年龄/工作年限/车龄/拥有孩子数量/收入
# 2.4.1.1 描述性统计：年龄/工作年限/车龄/拥有孩子数量
# （1）年龄DAYS_BIRTH
print(all_data['DAYS_BIRTH'].head())               # 查看一下数据格式
print((all_data['DAYS_BIRTH'] / -365).describe())  # 去掉负号，且将天数转化为年（无异常）
# （2）工作年限DAYS_EMPLOYED
print(all_data['DAYS_EMPLOYED'].head())
print((all_data['DAYS_EMPLOYED'] / -365).describe()) # 这里发现工作年限最小值为负数，说明原表存再正数，下来查看一番
sort_DayEmployed = all_data['DAYS_EMPLOYED'].sort_values(ascending=False)   # 降序工作年限
print(sort_DayEmployed.head(10))                     # 确实存在一种取值：365243（1000年），明显异常
# 处理工作年限DAYS_EMPLOYED异常值
all_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)           # 暂且使用nan替换掉异常值
(all_data['DAYS_EMPLOYED'] /(-365)).plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')
plt.ylabel('Ferquency')
plt.grid()
plt.show()
# （3）车龄OWN_CAR_AGE
print(all_data['OWN_CAR_AGE'].head(10))
print(all_data['OWN_CAR_AGE'].describe())
# 处理车龄OWN_CAR_AGE异常值
all_data.reset_index(inplace=True)  # 重置一下索引，便于查找缺失值所在的位置
sort_OWN_CAR_AGE = all_data['OWN_CAR_AGE'].sort_values(ascending=False)
print(sort_OWN_CAR_AGE.head())
for i in (321542,294131,271741): # 注意：顺序一定不能错，先删除索引大的行，对小索引没有影响
    all_data = all_data.drop(i,axis=0)
print(all_data.shape)
# （4）拥有孩子数量CNT_CHILDREN
print(all_data['CNT_CHILDREN'].head())
print(all_data['CNT_CHILDREN'].describe())

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
# （2）检测收入AMT_INCOME_TOTAL异常值，并进行处理
all_data = outliers(all_data,'AMT_INCOME_TOTAL')
# 2.4.2 贷款信息检测：贷款信用额AMT_CREDIT/贷款年金AMT_ANNUITY/获得贷款的商品价格AMT_GOODS_PRICE
for col in ('AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE'):
    all_data = outliers(all_data,col)
print(all_data.shape)


# 3.违约用户画像探索-----------------------------------------------------------------
# 3.1 违约用户的性别CODE_GENDER差异
# 3.1.1 对违约用户进行画像之前，将训练集和测试集分开来
train_new = all_data[all_data['TARGET'].notnull()]  # TARGET不为空的则是训练集
train_y = train_new['TARGET'] # 保存下清洗完数据的预测变量，拟合模型时要用
test_new = all_data[~all_data['TARGET'].notnull()]  # TARGET为空的则是测试集
test_new = test_new.drop(['TARGET'], axis=1)
print('After data cleaning, train_new.shape:',train_new.shape)
print('After data cleaning, test_new.shape:',test_new.shape)

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
    plt.grid()
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)  # tick_params()定义参数刻度线样式：axis='both'表示对所有坐标轴都设置，默认也是both；which='major'表示设置主刻度线，默认也是major
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
# 3.1.2 查看违约用户的性别CODE_GENDER差异
plot_stats('CODE_GENDER')

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

# 3.3 违约用户与贷款类型NAME_CONTRACT_TYPE的关系
plot_stats('NAME_CONTRACT_TYPE')
# 3.4 违约用户与是否有车FLAG_OWN_CAR/是否有房FLAG_OWN_REALTY的关系
plot_stats('FLAG_OWN_CAR')
plot_stats('FLAG_OWN_REALTY')
# 3.5 违约用户与家庭类型NAME_FAMILY_STATUS的关系
plot_stats('NAME_FAMILY_STATUS',True,True)
# 3.6 违约用户与家庭孩子数量CNT_CHILDREN的关系
plot_stats('CNT_CHILDREN')
# 3.7 违约用户与收入类型NAME_INCOME_TYPE的关系
plot_stats('NAME_INCOME_TYPE',False,False)
# 3.8 违约用户与职业类型OCCUPATION_TYPE的关系
plot_stats('OCCUPATION_TYPE',True,True)
# 3.9 违约用户与学历NAME_EDUCATION_TYPE的关系
plot_stats('NAME_EDUCATION_TYPE',True)


# 4. 特征工程-----------------------------------------------------------------------
# 4.1 特征取值编码
cols = ('CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_CONTRACT_TYPE','OCCUPATION_TYPE',
        'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE')
for feature in cols:
    LE1 = LabelEncoder()
    LE1.fit(list(all_data[feature].values)) # 将每一列的特征取值作为一个列表塞进编码字典中进行编码
    all_data[feature] = LE1.transform(list(all_data[feature].values)) # 将每一列特征值转化为编码字典中的索引，对应fit就能知道每一个值编码后的结果

# 4.2 特征选择
drop_feature = ('WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','NAME_TYPE_SUITE',
                'FONDKAPREMONT_MODE','EMERGENCYSTATE_MODE')
for f1 in drop_feature:
    all_data = all_data.drop([f1], axis=1)
print("all data shape:{}".format(all_data.shape))
# 计算变量间相关性，剔除相关系数大于0.9的特征
train_y_new = all_data['TARGET']  # 对all_data不断进行更新中，更新下目标变量的存储
all_data_new = all_data.drop(['TARGET'], axis=1)
threshold = 0.9
corr_matrix = all_data_new.corr().abs()  # 计算相关系数矩阵
print(corr_matrix.head())
# 选择上三角阵处理
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)) # np.triu()获取上三角矩阵，k表示对角线起始位置
print(upper.head())
# 删除系数大于0.9的特征
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %s columns to remove.'%(len(to_drop)))
all_data_new = all_data_new.drop(columns=to_drop)
print("all_data_New shape:{}".format(all_data_new.shape))

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


# 5.建模及预测---------------------------------------------------------------
# （1）all_data：异常值和缺失值、编码
# （2）all_data_new：异常值和缺失值、编码、特征选择
# （3）all_data_copy：异常值和缺失值、编码、特征抽取
# （4）all_data_new_copy：异常值和缺失值、编码、特征选择、特征抽取
#  先划分数据集
train_1 = all_data[all_data['TARGET'].notnull()] # train_1存储all_data中的训练集
test_1 = all_data[~all_data['TARGET'].notnull()]
train_y1 = train_1['TARGET']
all_data_new['TARGET'] = train_y_new  # 特征选择的时候去掉了，现在加上
train_2 = all_data_new[all_data_new['TARGET'].notnull()] # train_2存储all_data_new中的训练集
test_2 = all_data_new[~all_data_new['TARGET'].notnull()]
train_y2 = train_2['TARGET']
train_3 = all_data_copy[all_data_copy['TARGET'].notnull()] # train_3存储all_data_copy中的训练集
test_3 = all_data_copy[~all_data_copy['TARGET'].notnull()]
train_y3 = train_3['TARGET']
all_data_new_copy['TARGET'] = train_y_new
train_4 = all_data_new_copy[all_data_new_copy['TARGET'].notnull()] # train_new4存储all_data_new_copy中的训练集
test_4 = all_data_new_copy[~all_data_new_copy['TARGET'].notnull()]
train_y4 = train_4['TARGET']

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

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

# 拟合模型
submission1, fi1, metrics1 = model(train_1, test_1)
print('Baseline metrics1')
print(metrics1)
submission2, fi2, metrics2 = model(train_2, test_2)
print('Baseline metrics2')
print(metrics2)
submission3, fi3, metrics3 = model(train_3, test_3)
print('Baseline metrics3')
print(metrics2)
submission4, fi4, metrics4 = model(train_4, test_4)
print('Baseline metrics4')
print(metrics4)

# 通过lgb自带的函数查看特征重要性
def plot_feature_importances(df):
    df = df.sort_values('importance', ascending=False).reset_index()
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    plt.figure(figsize=(10,6))
    ax = plt.subplot()
    ax.barh(list(reversed(list(df.index[:15]))), df['importance_normalized'].head(15), align='center', edgecolor='k')
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.grid()
    plt.show()
    return df
fi_sorted4 = plot_feature_importances(fi4)
# 看一下预测的结果
print(submission4.head())










