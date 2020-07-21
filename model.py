# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:35:20 2020

@author: This PC
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('car_data.csv')
print(df.shape)

    
print(df['Seller_Type'].unique())    
print(df['Transmission'].unique())  
print(df['Owner'].unique())  

#check missing values
print(df.isnull().sum())
print(df.describe())
print(df.columns)
final_df=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_df['current_year']=2020

final_df['no_of_year']=final_df['current_year']-final_df['Year']
final_df.drop(['Year'],axis=1,inplace=True)
final_df.drop(['current_year'],axis=1,inplace=True)
final_df=pd.get_dummies(final_df,drop_first=True)
print(final_df.corr)

#ns.pairplot(final_df)

#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


X=final_df.iloc[:,1:]
y=final_df.iloc[:,0]

##Feature Importance

from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)

print(model.feature_importances_)

feat_importance=pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()

#Hyper parameter tuning
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

print(rf_random.best_params_)


predictions=rf_random.predict(X_test)

sns.distplot(y_test-predictions)

plt.scatter(y_test,predictions)


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)
