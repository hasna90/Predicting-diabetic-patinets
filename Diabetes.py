#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:04:53 2023

@author: hasnaghamdi
"""


import pandas as pd

pd.read_csv("D:/diabetes.csv")
diabetes=pd.read_csv("D:/diabetes.csv")

#1 How to explore the given data?
#2 How to perform data pre-processing (missing values, outliers, transformations, etc.)
#Data Exploratory actions:
    
# Checking outcome variable and data balance:
diabetes['Outcome'].value_counts()
#0    500
#1    268
268/(500+268)
# 35 %
# Data are partially balanced 65% - 35% (value of interest)

# Checking Missing /Null Values:
diabetes.isnull().sum()
# There are no Missing /Null Values as per below result:
#Pregnancies                 0
#Glucose                     0
#BloodPressure               0
#SkinThickness               0
#Insulin                     0
#BMI                         0
#DiabetesPedigreeFunction    0
#Age                         0
#Outcome                     0

# Checking outliers

#Checking outliers
diabetes['Pregnancies'].plot(kind='box')  # as per boxplot, there appears to have outliers 
diabetes['Glucose'].plot(kind='box')  # as per boxplot, there appears to have outliers 
diabetes['BloodPressure'].plot(kind='box')  # as per boxplot, there appears to have outliers 
diabetes['SkinThickness'].plot(kind='box') # as per boxplot, there appears to have outliers 
diabetes['Insulin'].plot(kind='box') # as per boxplot, there appears to have outliers 
diabetes['BMI'].plot(kind='box') # as per boxplot, there appears to have outliers 
diabetes['DiabetesPedigreeFunction'].plot(kind='box') # as per boxplot, there appears to have outliers 
diabetes['Age'].plot(kind='box') # as per boxplot, there appears to have outliers 
diabetes['Outcome'].value_counts() # there is no outlier in the outcome varibale

#Dealing with outliers:
from scipy import stats

diabetes['Pregnancies_z']= stats.zscore(diabetes['Pregnancies'])
diabetes['Glucose_z']= stats.zscore(diabetes['Glucose'])
diabetes['BloodPressure_z']= stats.zscore(diabetes['BloodPressure'])
diabetes['SkinThickness_z']= stats.zscore(diabetes['SkinThickness'])
diabetes['Insulin_z']= stats.zscore(diabetes['Insulin'])
diabetes['BMI_z']= stats.zscore(diabetes['BMI'])
diabetes['DiabetesPedigreeFunction_z']= stats.zscore(diabetes['DiabetesPedigreeFunction'])
diabetes['Age_z']= stats.zscore(diabetes['Age'])


Pregnancies_oultiers= diabetes.query('Pregnancies_z > 3 ')
#it was decided to leave them as they are true data and only in 4 records, less than (1%) of the records and it is not recommended to delete them or replace them. Just keep them.

Glucose_oultiers= diabetes.query('Glucose_z < -3')
#it was decided to leave them as they are true data and only in 5 records, less than (1%) of the records and it is not recommended to delete them or replace them. Just keep them.

BloodPressure_oultiers= diabetes.query('BloodPressure_z > 3| BloodPressure_z < -3')
#it was decided to leave them as they are true data and only in 35 records, less than (5%) of the records and it is not recommended to delete them or replace them. Just keep them.

SkinThickness_oultiers= diabetes.query('SkinThickness_z > 3')
#it was decided to leave them as they are true data and only in 1 record, less than (1%) of the records and it is not recommended to delete them or replace them. Just keep them.

Insulin_oultiers= diabetes.query('Insulin_z > 3')
#it was decided to leave them as they are true data and only in 18 records, less than (2.5%) of the records and it is not recommended to delete them or replace them. Just keep them.

BMI_oultiers= diabetes.query('BMI_z > 3| BMI_z < -3')
#it was decided to leave them as they are true data and only in 14 records, less than (2%) of the records and it is not recommended to delete them or replace them. Just keep them.

DiabetesPedigreeFunction_oultiers= diabetes.query('DiabetesPedigreeFunction_z > 3')
#it was decided to leave them as they are true data and only in 11 records, less than (2%) of the records and it is not recommended to delete them or replace them. Just keep them.

Age_oultiers= diabetes.query('Age_z > 3')
#it was decided to leave them as they are true data and only in 5 records, less than (1%) of the records and it is not recommended to delete them or replace them. Just keep them.

#it was decided to leave them as they are true data and only in small number of the records, less than (5%) of the records and not far away values that would affect the analysis of the data so it is not recommended to delete them or replace them. 


# Visual explratory data analysis for data distribution 

diabetes['Pregnancies'].plot(kind='hist', title='Pregnancies')
diabetes['Glucose'].plot(kind='hist', title='Glucose')
diabetes['BloodPressure'].plot(kind='hist', title='BloodPressure')
diabetes['SkinThickness'].plot(kind='hist', title='SkinThickness')
diabetes['Insulin'].plot(kind='hist', title='Insulin')
diabetes['BMI'].plot(kind='hist', title='BMI')
diabetes['DiabetesPedigreeFunction'].plot(kind='hist', title='DiabetesPedigreeFunction')
diabetes['Age'].plot(kind='hist', title='Age')
#for outcome categorical variable:
diabetes['Outcome_str']= diabetes['Outcome']
diabetes['Outcome_str'].replace([0,1],['Non-Diabetic','Diabetic'], inplace = True)
diabetes['Outcome_str'].value_counts()
#Non-Diabetic    500
#Diabetic        268
diabetes['Outcome_str'].value_counts().plot(kind='bar', title='Outcome')

import numpy as np

#. variable Corrlations:

import seaborn as sns

diabetes_org = diabetes[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']]

sns.heatmap(diabetes_org.corr(), cmap="YlGnBu", annot=True)
Corr_diabetes=diabetes_org.corr()
Corr_diabetes

# How to find the best analysis algorithm for the given data?


#As our outcome variable is categorical of 2 classes, 
#based on our knowledge most likely classification algorithms would work to predict the class of diabetic (1) or not (0), logistic regression was chosen.

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
#A commonly used ratio is 80:20, which means 80% of the data is for training and 20% for testing.

diabetes_train, diabetes_test = train_test_split(diabetes, test_size = 0.20, random_state = 7)
diabetes_train.shape # 614 (80%)
diabetes_test.shape  # 154 (20%)

diabetes_train['Outcome'].value_counts()
#0    403
#1    211
# as per count, the Partitioning is proportionally done and accepted (34% value of interest)
diabetes_test['Outcome'].value_counts()
#0    97
#1    57
# as per count, the Partitioning is proportionally done and accepted (58% value of interest)


# MODEL#1
# defining the Y and X variables
Xtrain = diabetes_train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Ytrain = diabetes_train[['Outcome']]

# building the model and fitting the data
reg_model1 = sm.Logit(Ytrain, Xtrain).fit()
#Optimization terminated successfully.
         #Current function value: 0.606803
         #Iterations 5

print(reg_model1.summary2()) 

#The most important variables show the p values less than 0.05:
#Pregnancies
#Glucose
#BloodPressure
#DiabetesPedigreeFunction
#Age


Xtest = diabetes_test[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Ytest = diabetes_test[['Outcome']]

# finding accuracy:
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(Xtrain,Ytrain)
Y_pred= logreg.predict(Xtest)
print('Logistic regression model accuracy: {:.2f}'.format(logreg.score(Xtest,Ytest)))
#Logistic regression model accuracy: 0.79 (79%)

#Finding Root mean squared error
from sklearn.metrics import mean_squared_error

#  Values to compare
Y_true = Ytest
Y_pred
y_pred=Y_pred
# Root mean squared error (by using: squared=False)

rmse = mean_squared_error(Y_true, Y_pred, squared=False)

print(rmse)
#0.4558423058385518


#############################

#Trying another Logistic regression model with only 
# MODEL#2
#The most important variables show the p values less than 0.05 of reg_model1:
#Pregnancies, Glucose, BloodPressure, DiabetesPedigreeFunction, Age

#re defining the X variables for training data
Xtrain2 = diabetes_train[['Pregnancies','Glucose','BloodPressure','DiabetesPedigreeFunction','Age']]


# building the model and fitting the data
reg_model2 = sm.Logit(Ytrain, Xtrain2).fit()
#Optimization terminated successfully.
         #Current function value: 0.607112
         #Iterations 5

print(reg_model2.summary2())

#The most important features show the p values less than 0.05 are still the same:
#Pregnancies
#Glucose
#BloodPressure
#DiabetesPedigreeFunction
#Age

#re defining the X variables for testing data
Xtest2 = diabetes_test[['Pregnancies','Glucose','BloodPressure','DiabetesPedigreeFunction','Age']]


# finding accuracy:
logreg2 = LogisticRegression()
logreg2.fit(Xtrain2,Ytrain)
Y_pred2 = logreg2.predict(Xtest2)
print('Logistic regression model accuracy: {:.2f}'.format(logreg2.score(Xtest2,Ytest)))

#Logistic regression model accuracy: 0.77 (77%)

#Finding Root mean squared error
#  Values to compare
y_true = Ytest
Y_pred2

# Root mean squared error (by using: squared=False)
rmse2 = mean_squared_error(Y_true, Y_pred2, squared=False)

print(rmse2)
#0.48349377841522817

# Confusion Matrix
from sklearn import metrics

# Model 1:
cm1 = metrics.confusion_matrix(y_true, y_pred)
cm1

cm1_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm1, display_labels = [False, True])
cm1_display.plot()

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_true ,y_pred,labels=[0, 1]).ravel()
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)

#True Positive 34
#True Negative 88
#False Positive 9
#False Negative 23

#TPR(Sensitivity)=TP/TP+FN =
34/(34+23) #0.59

#(Specificity)=TN/TN+FP =
88/(88+9) #0.907

# Model 2
cm2 = metrics.confusion_matrix(y_true, Y_pred2)
cm2

cm2_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm2, display_labels = [False, True])
cm2_display.plot()


from sklearn.metrics import confusion_matrix

tn2, fp2, fn2, tp2 = confusion_matrix(y_true ,Y_pred2,labels=[0, 1]).ravel()
print('True Positive', tp2)
print('True Negative', tn2)
print('False Positive', fp2)
print('False Negative', fn2)

#True Positive 33
#True Negative 85
#False Positive 12
#False Negative 24

#TPR (Sensitivity)=TP/TP+FN =
33/(33+24) #0.57

#(Specificity)=TN/TN+FP =
85/(85+12) #0.876

#If the TPR is closer to 1 that shows that it is a very good model. The model is able to distinguish between positive and negative outcomes correctly.
#If the TPR is 0.5, it shows that the model is as good as a guess.
#If the TPR is below 0.5, it shows that the model is predicting positive values as negative and negative values as positive

#Comparing & Evaluating Models;
#Model 1 to Model 2 (Accuracy, RMSE, Sensitivity & Specificity )

#Model 2 Accuracy (77%) < Model 1 Accuracy (79%)
#==> Model 1 is better as predicting model with higher accuracy

#Model 2 Root mean squared error rmse2(0.483) > Model 1 Root mean squared error rmse (0.455)
#==> Model 1 is better as predicting model with less RMSE

#Model 2 Sensitivity (0.57) < Model 1 Sensitivity (0.59)
#==> Model 1 is better as predicting model with higher Sensitivity

#Model 2 Specificity (0.907) < Model 1 Specificity (0.876)
#==> Model 1 is better as predicting model with higher Specificity



