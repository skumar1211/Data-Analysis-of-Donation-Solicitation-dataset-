#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 23:24:47 2018

@author: sharad
"""

import pandas as pd

#Data cleaning and preprocessing

df_zip = pd.read_csv('zipCodeMarketingCosts.csv')
df = pd.read_csv("train.csv", index_col='id')

df['zip'] = df['zip'].replace('\-','',regex=True).astype(float)

new_df = df.merge(df_zip, how = 'inner', on = 'zip')

training_df = new_df.loc[:,['zip','cluster','major','pepstrfl','rfa_2f','rfa_2a','marketingCost','responded']]
training_df['pepstrfl'].replace(('X',' '), (1,0), inplace=True)
training_df['major'].replace(('X',' '), (1,0), inplace=True)


training_df['cluster'] = pd.to_numeric(training_df['cluster'], errors='coerce')
training_df = pd.get_dummies(training_df, columns = ['rfa_2a'])
training_df['cluster'].fillna((training_df['cluster'].mean()), inplace=True)


X_df = training_df.drop('responded', axis=1)
Y_df = training_df['responded']



#Standardisation 

from sklearn.preprocessing import StandardScaler
n = StandardScaler()
X_df = pd.DataFrame(n.fit_transform(X_df), columns = [X_df.columns])


#Modeling

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_df,Y_df, test_size=0.2, random_state=0) 



#Logistic Regression

from sklearn.linear_model import LogisticRegressionCV
cls_lr_cv = LogisticRegressionCV(cv = 10)

cls_lr_cv.fit(X_train,Y_train)
pred_lr_cv = cls_lr_cv.predict(X_test)
cls_lr_cv.score(X_test, Y_test)



#Decision Tree

from sklearn.tree import DecisionTreeClassifier

cls_dt = DecisionTreeClassifier()
cls_dt.fit(X_train,Y_train)
pred_dt = cls_dt.predict(X_test)
cls_dt.score(X_test,Y_test)



#Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

cls_rf = RandomForestClassifier(n_estimators = 30)
cls_rf.fit(X_train,Y_train)
pred_rf = cls_rf.predict(X_test)
cls_rf.score(X_test,Y_test)



#Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

cls_gb = GradientBoostingClassifier()
cls_gb.fit(X_train,Y_train)
pred_gb = cls_gb.predict(X_test)
cls_gb.score(X_test,Y_test)



#Support Vector Machines

from sklearn.svm import SVC

cls_sv = SVC()
cls_sv.fit(X_train,Y_train)
pred_sv = cls.predict(X_test)
cls_sv.score(X_test,Y_test)




#Testing dataset preparation

test_df = pd.read_csv("test.csv", index_col='id')

test_df['zip'] = test_df['zip'].replace('\-','',regex=True).astype(float)

test_df = test_df.merge(df_zip, how = 'left', on = 'zip')

testing_df = test_df.loc[:,['zip','cluster','major','pepstrfl','rfa_2f','rfa_2a','marketingCost']]

testing_df['pepstrfl'].replace(('X',' '), (1,0), inplace=True)
testing_df['major'].replace(('X',' '), (1,0), inplace=True)

testing_df['cluster'] = pd.to_numeric(testing_df['cluster'], errors='coerce')
testing_df = pd.get_dummies(testing_df, columns = ['rfa_2a'])

testing_df['cluster'].fillna((testing_df['cluster'].mean()), inplace=True)


#prediction using best model

predicted_test = cls_rf.predict(testing_df)
print(predicted_test.shape)


final_test_df = pd.read_csv("test.csv", index_col='id')

final_test_df['market'] = predicted_test