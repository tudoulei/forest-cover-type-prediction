"""机器学习"""

# 导入类库
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score

from scipy.stats import randint, uniform
from sklearn.metrics import roc_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
import pandas as pd
import pandas
from sklearn.decomposition import PCA
from pandas import read_csv
from scipy.stats import uniform
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.stats import uniform
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# 导入数据


# 数据描述性
def data_descirble(dataset):
    # 数据维度
    print(dataset.shape)
    # 特征属性的字段类型
    print(dataset.dtypes)
    '''
    # 查看最开始的30条记录
    print('查看最开始的30条记录')
    pd.set_option('display.max_columns', None)
    # set_option('display.line_width', 120)
    print(data.head(2))
    '''

    # 描述性统计信息
    pd.set_option('display.max_columns', None)
    print(dataset.describe())

    # 显示各类数据的偏离
    # print(dataset.skew)

    # 显示分类的各类的数量
    # Number of instances belonging to each class

    dataset.groupby('Cover_Type').size()

    # create a dataframe with only 'size' features
    size = 10
    data = dataset.iloc[:, :size]

    # get the names of all the columns
    cols = data.columns
    print(cols)

    # Calculates pearson co-efficient for all combinations
    data_corr = data.corr()

    # Set the threshold to select only only highly correlated attributes
    threshold = 0.5

    # List of pairs along with correlation above threshold
    corr_list = []

    # Checking the value count for different soil_types
    for i in range(10, dataset.shape[1] - 1):
        j = dataset.columns[i]
        print(dataset[j].value_counts())

    # Search for the highly correlated pairs
    # 输出相关性大于0.5小于-0.5的特征
    for i in range(0, size):  # for 'size' features
        for j in range(i + 1, size):  # avoid repetition
            if (data_corr.iloc[i, j] >= threshold and data_corr.iloc[i, j] < 1) or (
                    data_corr.iloc[i, j] < 0 and data_corr.iloc[i, j] <= -threshold):
                corr_list.append([data_corr.iloc[i, j], i, j])  # store correlation and columns index

    # Sort to show higher ones first
    s_corr_list = sorted(corr_list, key=lambda x: -abs(x[0]))

    # Print correlations and column names
    for v, i, j in s_corr_list:
        print("%s and %s = %.2f" % (cols[i], cols[j], v))

    # Strong correlation is observed between the following pairs
    # This represents an opportunity to reduce the feature set through transformations such as PCA

    '''
    # import plotting libraries
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Scatter plot of only the highly correlated pairs
    for v, i, j in s_corr_list:
        sns.pairplot(dataset, hue="Cover_Type", size=6, x_vars=cols[i], y_vars=cols[j])
        plt.show()

    '''
    # Horizontal_Distance_To_Hydrology
    from scipy import stats
    plt.figure(figsize=(8, 6))
    sns.distplot(dataset['Horizontal_Distance_To_Hydrology'], fit=stats.norm)
    fig = plt.figure(figsize=(8, 6))
    res = stats.probplot(dataset['Horizontal_Distance_To_Hydrology'], plot=plt)
    plt.show()

    # data visualization
    fig = plt.figure()
    fig.set(alpha=0.2)

    plt.subplot2grid((2, 3), (0, 0))
    dataset.Cover_Type.value_counts().plot(kind='bar')
    plt.title('cover')
    plt.xlabel('cover')

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(dataset.Elevation, dataset.Cover_Type)
    plt.ylabel("cover")  # sets the y axis lable
    plt.grid(b=True, which='major', axis='y')  # formats the grid line style of our graphs
    plt.title("cover and Elevation")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    for i in range(1, 8):
        dataset.Elevation[dataset.Cover_Type == i].plot(kind='kde')
    plt.xlabel("elevation")  # plots an axis lable
    plt.ylabel("midu")
    plt.title("cover and Elevation")

    plt.show()


# data cleaning
'''
# Removal list initialize
# 移除方差为0的无用特征
rem = []

# Add constant columns as they don't help in prediction process
for c in dataset.columns:
    if dataset[c].std() == 0:  # standard deviation is zero
        rem.append(c)

# drop the columns
dataset.drop(rem, axis=1, inplace=True)

print(rem)
'''


# Following columns are dropped


"绘制直方图"
def zhifang(dataset):
    cols = dataset.columns
    r, c = dataset.shape

    # Create a new dataframe with r rows, one column for each encoded category, and target in the end
    new_data = pd.DataFrame(index=np.arange(0, r), columns=['Wilderness_Area', 'Soil_Type', 'Cover_Type'])

    # Make an entry in data for each r for category_id, target_value
    for i in range(0, r):
        p = 0;
        q = 0;
        # Category1_range
        for j in range(10, 14):
            if (dataset.iloc[i, j] == 1):
                p = j - 9  # category_class
                break
        # Category2_range
        for k in range(14, 54):
            if (dataset.iloc[i, k] == 1):
                q = k - 13  # category_class
                break
        # Make an entry in data for each r
        new_data.iloc[i] = [p, q, dataset.iloc[i, c - 1]]

    # plot for category1
    sns.countplot(x='Wilderness_Area', hue='Cover_Type', data=new_data)
    plt.show()

    # Plot for category2
    plt.rc("figure", figsize=(25, 10))
    sns.countplot(x='Soil_Type', hue='Cover_Type', data=new_data)
    plt.show()


'''
df_train = dataset
df_test = dataset_test
df_train = df_train.iloc[:,1:]
df_test = df_test.iloc[:,1:]
Size = 10
X_temp = df_train.iloc[:,:Size]
X_test_temp = df_test.iloc[:,:Size]

r,c = df_train.shape
X_train = np.concatenate((X_temp,df_train.iloc[:,Size:c-1]),axis=1)
y_train = df_train.Cover_Type.values

r,c = df_test.shape
X_test = np.concatenate((X_test_temp, df_test.iloc[:,Size:c]), axis = 1)


X_train, X_validation, Y_train, Y_validation  = train_test_split(X_train, y_train, test_size = 0.3)
'''

"数据特征处理"
def data_pre(data):
    print("去掉不正常值前：")
    print(data.shape)
    # 移除一些不好数据
    # 绝对不能讲线上test数据清洗了
    r, c = data.shape
    if r == 15120:
        data.drop(['Hillshade_3pm'] == 0, axis=0, inplace=True)

    print("去掉不正常值后：")
    print(data.shape)
    data['Ele_minus_VDtHyd'] = data.Elevation - data.Vertical_Distance_To_Hydrology

    data['Ele_plus_VDtHyd'] = data.Elevation + data.Vertical_Distance_To_Hydrology

    data['Distanse_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology'] ** 2 + data[
        'Vertical_Distance_To_Hydrology'] ** 2) ** 0.5

    data['Hydro_plus_Fire'] = data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Fire_Points']

    data['Hydro_minus_Fire'] = data['Horizontal_Distance_To_Hydrology'] - data['Horizontal_Distance_To_Fire_Points']

    data['Hydro_plus_Road'] = data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Roadways']

    data['Hydro_minus_Road'] = data['Horizontal_Distance_To_Hydrology'] - data['Horizontal_Distance_To_Roadways']

    data['Fire_plus_Road'] = data['Horizontal_Distance_To_Fire_Points'] + data['Horizontal_Distance_To_Roadways']

    data['Fire_minus_Road'] = data['Horizontal_Distance_To_Fire_Points'] - data['Horizontal_Distance_To_Roadways']

    data['Soil'] = 0
    for i in range(1, 41):
        data['Soil'] = data['Soil'] + i * data['Soil_Type' + str(i)]
    for i in range(1, 41):
        name = 'Soil_Type' + str(i)
        data.drop([name], axis=1, inplace=True)
    data['Wilderness_Area'] = 0
    for i in range(1, 5):
        data['Wilderness_Area'] = data['Wilderness_Area'] + i * data['Wilderness_Area' + str(i)]
    for i in range(1, 5):
        name = 'Wilderness_Area' + str(i)
        data.drop([name], axis=1, inplace=True)

    return data


"缺失数据填补算法"
def set_missing_data(dataset):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中,构造预测模型data_model
    col = ['a', 'b']
    data_model = dataset[col]

    # 缺失数据分成已知和未知两部分,修改age为缺失数据
    known = data_model[data_model.Age.notnull()].as_matrix()
    unknown = data_model[data_model.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known[:, 0]

    # X即特征属性值
    X = known[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    pred = rfr.predict(unknown[:, 1::])

    # 用得到的预测结果填补原缺失数据
    dataset.loc[(dataset.Age.isnull()), 'Age'] = pred

    return dataset, rfr


"LGB算法进行训练"
def lgb_model(X_train, X_validation, Y_train, Y_validation):
    print('Training LGBM model...')

    # 使用较大的 max_bin （学习速度可能变慢）使用较小的 learning_rate 和较大的 num_iterations使用较大的 num_leaves （可能导致过拟合）使用更大的训练数据尝试 dart
    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',

        num_leaves=35,
        depth=8,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=20000)
    LGB = lgb0.fit(X_train, Y_train, eval_set=[(X_validation, Y_validation)], early_stopping_rounds=200)
    y_pred = LGB.predict(X_validation)

    cm = confusion_matrix(Y_validation, y_pred)
    print('混淆矩阵为：')
    print(cm)
    # cmap参数为绘制矩阵的颜色集合，这里使用灰度
    plt.matshow(cm, cmap=plt.cm.gray)
    plt.show()
    # print('调和平均值F1_score：%.f' % (f1_score(Y_validation, y_pred)))
    model_score = LGB.score(X_validation, Y_validation)
    print('分类精度为：%.f' % (model_score))
    target = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7']
    print(classification_report(Y_validation, y_pred, target_names=target))
    accuracy = accuracy_score(Y_validation, y_pred);
    print('accuracy:%0.2f%%' % (accuracy * 100))
    return LGB


"xgboost算法"
def xgb_model(X_train, X_validation, Y_train, Y_validation):
    params_dist_grid = {
        'max_depth': [1, 5, 10],
        'gamma': [0, 0.5, 1],
        'n_estimators': randint(1, 1001),  # uniform discrete random distribution
        'learning_rate': uniform(),  # gaussian distribution
        'subsample': uniform(),  # gaussian distribution
        'colsample_bytree': uniform(),  # gaussian distribution
        'reg_lambda': uniform(),
        'reg_alpha': uniform()
    }
    XGBC = XGBClassifier(silent=1, n_estimators=641, learning_rate=0.2, max_depth=10, gamma=0.5, nthread=-1,
                         reg_alpha=0.05, reg_lambda=0.35, max_delta_step=1, subsample=0.83, colsample_bytree=0.6)
    eval_set = [(X_validation, Y_validation)]

    XGBC.fit(X_train, Y_train, early_stopping_rounds=100, eval_set=eval_set, eval_metric='merror', verbose=True)

    pred = XGBC.predict(X_validation)

    accuracy = accuracy_score(Y_validation, pred);
    print('accuracy:%0.2f%%' % (accuracy * 100))
    target = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7']
    print(classification_report(Y_validation, pred, target_names=target))
    return XGBC





"用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve"
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    print('starting draw learning curve ....')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("data amount")
        plt.ylabel("score")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="the score of training ")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="the score of spilt")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()
    print('over...')
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff



# 使用k近邻的效果并不好
'''
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
num_folds = 10
seed = 7
scoring = 'r2'
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X=X_train, y=Y_train)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
# 最优：0.657371268812 使用{'n_neighbors': 1}
'''

'''
num_leaves=55,  0.338577
num_leaves=85, 0.331606

# 线上测试
# 第一次0.7314

'''


# 获取测试集裁剪
def data_test():
    filename = 'test.csv'
    dataset = read_csv(filename)  # 若不加入dtype=float 会出现warning
    dataset = dataset.iloc[:, 1:]  # 导入数据集一般直接用这一步，就可以将headline直接导入

    data = dataset
    data['Ele_minus_VDtHyd'] = data.Elevation - data.Vertical_Distance_To_Hydrology

    data['Ele_plus_VDtHyd'] = data.Elevation + data.Vertical_Distance_To_Hydrology

    data['Distanse_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology'] ** 2 + data[
        'Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
    data['Hydro_plus_Fire'] = data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Fire_Points']

    data['Hydro_minus_Fire'] = data['Horizontal_Distance_To_Hydrology'] - data['Horizontal_Distance_To_Fire_Points']

    data['Hydro_plus_Road'] = data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Roadways']

    data['Hydro_minus_Road'] = data['Horizontal_Distance_To_Hydrology'] - data['Horizontal_Distance_To_Roadways']

    data['Fire_plus_Road'] = data['Horizontal_Distance_To_Fire_Points'] + data['Horizontal_Distance_To_Roadways']

    data['Fire_minus_Road'] = data['Horizontal_Distance_To_Fire_Points'] - data['Horizontal_Distance_To_Roadways']
    data['Soil'] = 0
    for i in range(1, 41):
        data['Soil'] = data['Soil'] + i * data['Soil_Type' + str(i)]
    data['Wilderness_Area'] = 0
    for i in range(1, 5):
        data['Wilderness_Area'] = data['Wilderness_Area'] + i * data['Wilderness_Area' + str(i)]
    for i in range(1, 5):
        name = 'Wilderness_Area' + str(i)
        dataset.drop([name], axis=1, inplace=True)
    data['Soil'] = 0
    for i in range(1, 41):
        data['Soil'] = data['Soil'] + i * data['Soil_Type' + str(i)]
    for i in range(1, 41):
        name = 'Soil_Type' + str(i)
        dataset.drop([name], axis=1, inplace=True)
    return dataset


def data_in_sub(model):
    # dataset_test = pandas.read_csv("test.csv")
    # 获取id值
    dataset_test = read_csv('test.csv')

    # Drop unnecessary columns
    ID = dataset_test['Id']
    # dataset_test.drop('Id',axis=1,inplace=True)
    dataset_test = data_test()
    # dataset_test.drop('Id', axis=1, inplace=True)
    X_test = dataset_test.values

    # Make predictions using the best model
    predictions = model.predict(X_test)
    print(predictions.shape)
    # Write submissions to output file in the correct format
    print("开始输出线上预测数据....")
    with open("submission.csv", "w") as subfile:
        subfile.write("Id,Cover_Type\n")
        for i, pred in enumerate(list(predictions)):
            subfile.write("%s,%s\n" % (ID[i], pred))


if __name__ == "__main__":
    filename = 'train.csv'
    dataset = read_csv(filename)  # 若不加入dtype=float 会出现warning
    dataset = dataset.iloc[:, 1:]  # 导入数据集一般直接用这一步，就可以将headline直接导入
    # data_descirble(dataset)
    # zhifang(dataset)
    dataset = data_pre(dataset)


    Y = dataset['Cover_Type'].values
    dataset.drop('Cover_Type', axis=1, inplace=True)
    array = dataset.values
    X = array[:, :]
    validation_size = 0.3
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # LGB = lgb_model(X_train, X_validation, Y_train, Y_validation)
    XGBC = xgb_model(X_train, X_validation, Y_train, Y_validation)
    # data_in_sub(LGB)
    # model = ExtraTreesClassifier(max_features=0.3, n_estimators=500)
    # plot_learning_curve(model, "STUDYING CURVE", X, Y, cv=5)
    data_in_sub(XGBC)

'''
print('多元线性回归,R方为0.36')

def model_LDA(X, Y):
    model = LinearDiscriminantAnalysis()
    validation_size = 0.2
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    model.fit(X_train, Y_train)
    result = model.score(X_validation, Y_validation)
    print(result)
model_LDA(X,Y)

'''
