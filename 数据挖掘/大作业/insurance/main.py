import pandas as pd
import datetime
import warnings
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

#读取数据集
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
submission = pd.read_csv("submission.csv")
dataset = pd.concat([data_train,data_test])

#若为日期的处理
dataset['incident_date'] = pd.to_datetime(dataset['incident_date'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2024-06-19', '%Y-%m-%d')#设置开始日期
dataset['time'] = dataset['incident_date'].apply(lambda x: startdate-x).dt.days

#分离出对象类型变量（非数值）
numerical_fea = list(dataset.select_dtypes(include=['object']).columns)

#对分离的数据进行处理，转换为编码
division = LabelEncoder()
for feature in numerical_fea:
    division.fit(dataset[feature].values)
    dataset[feature] = division.transform(dataset[feature].values)

#空值处理
testA = dataset[dataset['fraud'].isnull()].drop(['policy_id','incident_date','fraud'],axis=1)
trainA = dataset[dataset['fraud'].notnull()]
#训练集中与预测无关的值id,date等单独处理（噪声处理）
data_x = trainA.drop(['policy_id','incident_date','fraud'],axis=1)
data_y = data_train[['fraud']].copy()

col=['policy_state','insured_sex','insured_education_level','incident_type','collision_type','incident_severity','authorities_contacted','incident_state',
     'incident_city','police_report_available','auto_make','auto_model']
for i in data_x.columns:
    if i in col:
        data_x[i] = data_x[i].astype('str')
for i in testA.columns:
    if i in col:
        testA[i] = testA[i].astype('str')

#模型初始化
model=CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    task_type="CPU",
    learning_rate=0.2,
    iterations=5000,
    random_seed=1,
    od_type="Iter",
    depth=5,
    early_stopping_rounds=250)

answers = []
mean_score = 0
n_folds = 10#10折交叉
sk = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
for train, test in sk.split(data_x, data_y):

    x_train = data_x.iloc[train]
    y_train = data_y.iloc[train]
    x_test = data_x.iloc[test]
    y_test = data_y.iloc[test]

    clf = model.fit(x_train,y_train, eval_set=(x_test,y_test),verbose=500,cat_features=col)
    y_pred=clf.predict(x_test)
    print('auc:{}'.format(roc_auc_score(y_test, y_pred)))

    mean_score += roc_auc_score(y_test, y_pred) / n_folds
    answers.append(clf.predict(testA,prediction_type='Probability')[:,-1])

print('10折平均auc:{}'.format(mean_score))
lgb_pre=sum(answers)/n_folds
submission['fraud']=lgb_pre
submission.to_csv('predict.csv',index=False)