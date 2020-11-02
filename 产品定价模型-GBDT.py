# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:47:43 2020

@author: ASK
"""

import pandas as pd

df = pd.read_excel('产品定价模型.xlsx')
print(df['类别'].value_counts())
print(df['彩印'].value_counts())
print(df['纸张'].value_counts())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['类别'] = le.fit_transform(df['类别'])
df['纸张'] = le.fit_transform(df['纸张'])

X = df.drop(columns='价格')
y = df['价格']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=123)

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=123)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

a = pd.DataFrame()
a['预测值']=list(y_pred)
a['真实值']=list(y_test)
print(a)

score = model.score(X_test,y_test)
print("预测准确地评分：",score)

features = X.columns
importances = model.feature_importances_

importance_df = pd.DataFrame()
importance_df['特征名']=features
importance_df['特征重要性']= importances
importance_df.sort_values('特征重要性',ascending=True)
print(importance_df)
