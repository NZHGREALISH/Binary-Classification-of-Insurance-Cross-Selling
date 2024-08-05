import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
##数据读取
train_data= pd.read_csv("../data/train.csv")
test_data= pd.read_csv('../data/test.csv')

#处理数据，one-hotEncoding 及 分割验证数据集并进行数据归一化
categorical_columns = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
train_data = pd.get_dummies(train_data.drop(columns=['id']), columns=categorical_columns, drop_first=True, dtype=int) 
X = train_data.loc[:, train_data.columns != "Response"]
y = train_data['Response']
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

X_train, X_validation, y_train, y_validation = train_test_split(X_scaled, y, test_size=0.3, random_state=100)

xgb_params = {
    'max_depth': 11,
    'min_child_weight': 0.8,
    'learning_rate': 0.02,
    'colsample_bytree': 0.6,
    'max_bin': 3000,
    'n_estimators': 1500,
    'tree_method': 'hist',     # 使用 GPU 进行训练
    'device': 'cuda'     # 使用 GPU 进行预测
}

model = XGBClassifier(**xgb_params)
model.fit(X_train, y_train)

# y_prob = model.predict_proba(X_validation)[:, 1]
# auc = roc_auc_score(y_validation, y_prob)
# print("Validation AUC: ", auc)
test_ids = test_data['id']
test_data.drop('id', axis=1, inplace=True)
test_encoded = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True, dtype=int) 
test_scaled = scaler.transform(test_encoded)
predictions_test = model.predict_proba(test_scaled)[:,1]

submission = pd.DataFrame({
    'id': test_ids,
    'Response': predictions_test.flatten()},columns=['id', 'Response'])
submission.to_csv('../submission/submission.csv', index=False)


