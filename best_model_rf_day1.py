#!/usr/bin/env python
# coding: utf-8

# In[129]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_rows', 500)


# In[130]:


def obj_handle(df):
    for col in ['MSSubClass', 'YrSold', 'MoSold']:
        df[col] = df[col].astype(object)
    return df


# In[131]:


def most_common(lst):
    return max(set(lst), key=lst.count)
def na_handler(df):
    df['PoolQC']=df['PoolQC'].fillna('No Pool')
    df['MiscFeature']=df['MiscFeature'].fillna('noAdd')
    df['Alley']=df['Alley'].fillna('NoAccess')
    df['Fence']=df['Fence'].fillna('NoFence')
    df['FireplaceQu']=df['FireplaceQu'].fillna('NoFireplace')
    df['GarageCond']=df['GarageCond'].fillna('noG')
    df['BsmtExposure']=df['BsmtExposure'].fillna('NoB')
    df['MasVnrType']=df['MasVnrType'].fillna('noM')
    df['Electrical']=df['Electrical'].fillna(most_common(df['Electrical'].tolist()))
    return df


# In[132]:


def float_handle(dff):
    dff['LotAreaAdditional']=dff['LotAreaAdditional'].fillna(0)
    dff=dff.drop('YearReconstruction',axis=1)
    dff=dff.drop('Variable01',axis=1)
    dff=dff.drop('Variable02',axis=1)
    dff=dff.drop('Variable03',axis=1)
    dff=dff.drop('Variable04',axis=1)
    dff['3rdFlrSF']=dff['3rdFlrSF'].fillna(0)
    dff['LotFrontage']=dff['LotFrontage'].fillna(dff['LotFrontage'].mean())
    dff['GarageYrBlt']=dff['GarageYrBlt'].fillna(min(dff['GarageYrBlt']))
    dff['MasVnrArea']=dff['MasVnrArea'].fillna(0)
    return dff
def years_to_age(dff):
    dff['YearBuilt']=dff['YearBuilt'].apply(lambda x: 2010-x)
    dff['YearRemodAdd']=dff['YearRemodAdd'].apply(lambda x: 2010-x)
    dff['GarageYrBlt']=dff['GarageYrBlt'].apply(lambda x: 2010-x)
    return dff

    


# In[133]:


def make_model_data(df,trainn=True,regr_qual = RandomForestClassifier(),ohe = OneHotEncoder(sparse=False,handle_unknown='ignore'),
      ohe1 = OneHotEncoder(sparse=False,handle_unknown='ignore')):
    df=obj_handle(df)
    df=float_handle(df)
    df=na_handler(df)
    df['OverallQual']=df['OverallQual'].astype('str').apply(lambda x: ''.join([i for i in x if i not in '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~']) )
    df['OverallQual']=df['OverallQual'].astype('float')
    popped=df.pop('OverallQual')
    df_obj=df.select_dtypes(include=['object'])
    
    for col in ['GarageType', 'GarageQual', 'GarageFinish', 'BsmtFinType1',
       'BsmtFinType2', 'BsmtCond', 'BsmtQual']:
        df_obj[col] = df_obj[col].fillna(str(0))
    if trainn==True:
        array_hot_encoded = ohe.fit_transform(df_obj)
        column_name = ohe.get_feature_names()

        data_hot_encoded = pd.DataFrame(array_hot_encoded, index=df_obj.index,columns= column_name)

    else:
        array_hot_encoded = ohe.transform(df_obj)
        column_name = ohe.get_feature_names()
        data_hot_encoded = pd.DataFrame(array_hot_encoded, index=df_obj.index,columns= column_name)

    
    
    df_ok=df.select_dtypes(include=['int64'])
    df_1=pd.concat([df_ok,data_hot_encoded], axis=1)
    try:
        SalePrice=df_1.pop('SalePrice')
        Y=SalePrice.values.tolist()
    except:
        Y=[0]
    if trainn==True:
        df_1['OverallQual']=popped
        df_1['OverallQual']=df_1['OverallQual'].fillna(123)
        df2=df_1.loc[df_1['OverallQual']!=123]
        df3=df_1.loc[df_1['OverallQual']==123]

        y=df2.pop('OverallQual')

        regr_qual = RandomForestClassifier()
        regr_qual.fit(df2, y)
        df3.pop('OverallQual')
        predictions=regr_qual.predict(df3)
        df2['OverallQual']=y
        df3['OverallQual']=predictions.tolist()
        df_1=pd.concat([df2,df3],axis=0)
        df_1.sort_index(inplace=True)
        array_hot_encoded = ohe1.fit_transform(df_1['OverallQual'].values.reshape(-1, 1))
        column_name = ohe1.get_feature_names()

        data_hot_encoded = pd.DataFrame(array_hot_encoded, index=df_obj.index,columns= column_name)
        df_1=pd.concat([df_1,data_hot_encoded], axis=1)
        df_1.pop('OverallQual')
        return df_1,Y,regr_qual,ohe,ohe1
    else:
        df_1['OverallQual']=popped
        df_1['OverallQual']=df_1['OverallQual'].fillna(123)
        df2=df_1.loc[df_1['OverallQual']!=123]
        df3=df_1.loc[df_1['OverallQual']==123]

        y=df2.pop('OverallQual')

        regr_qual = regr_qual
        df3.pop('OverallQual')
        predictions=regr_qual.predict(df3)
        df2['OverallQual']=y
        df3['OverallQual']=predictions.tolist()
        df_1=pd.concat([df2,df3],axis=0)
        array_hot_encoded = ohe1.transform(df_1['OverallQual'].values.reshape(-1, 1))
        column_name = ohe1.get_feature_names()

        data_hot_encoded = pd.DataFrame(array_hot_encoded, index=df_obj.index,columns= column_name)
        df_1=pd.concat([df_1,data_hot_encoded], axis=1)
        df_1.pop('OverallQual')
        df_1.sort_index(inplace=True)
        return df_1


# In[134]:


df=pd.read_csv(r'hack\train.csv')
df=df.drop('Unnamed: 0', axis=1)
df,Y,regr_qual,ohe,ohe1=make_model_data(df)


# In[135]:


df_test=pd.read_csv(r'hack\test.csv')
df_test=df_test.drop('Unnamed: 0', axis=1)
df_test=make_model_data(df=df_test,trainn=False,regr_qual=regr_qual,ohe=ohe,ohe1=ohe1)


# In[136]:


# list_val=[1241 for x in range(88)]
# list_percentage=[]
# nans=df.isna().sum().tolist()
# for i in range(len(nans)):
#     list_percentage.append(nans[i]/list_val[i]*100)
# df_perc=pd.DataFrame([list_percentage,list(df.columns.values),df.dtypes.values.tolist()])    
# df_perc=df_perc.T.rename(columns={0: "percents", 1: "cols",2:'dtype'})
# df_perc.sort_values(by=['percents'],ascending=False)


# <h1>ОБУЧЕНИЕ</h1>

# In[137]:


# xgb.XGBRegressor(verbosity=0) 
# linear_model.Ridge()
# RandomForestRegressor
# parameters = { 
#                     'objective':['reg:linear'],
#                     'learning_rate': [0.3], 
#                     'max_depth': [4],
#                     'min_child_weight': [1],
#                     'silent': [1],
#                     'subsample': [0.5],
#                     'colsample_bytree': [0.7],
#                     'n_estimators': [100]}
# reg=xgb.XGBRegressor(objective:'reg:linear',learning_rate: 0.3, max_depth: 4,min_child_weight: 1,silent: 1,
#                     subsample: 0.5,colsample_bytree: 0.7, n_estimators: 100) 


# In[138]:


# reg=xgb.XGBRegressor(learning_rate= 0.03, max_depth= 4,min_child_weight= 1,silent= 1,
#                     subsample= 0.5,colsample_bytree= 0.7, n_estimators= 100) 
# regr = RandomForestClassifier()


# In[144]:


regr = RandomForestClassifier()

param_grid = {
    'bootstrap': [True],
    'max_depth': [100],
    'max_features': ['auto'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [6,8, 10],
    'n_estimators': [100, 120, 150]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
grid.fit(df, Y)
print(grid.best_params_)
best=grid.best_estimator_


# In[145]:


print(grid.best_params_)


# In[146]:


# СУПЕРПРЕДИКТ
def main_pred(regr,df,Y):
    df_test=pd.read_csv('testtarget.csv')
    df_test=df_test.drop('Unnamed: 0', axis=1)
    y1=df_test.pop('SalePrice')

    df_test=make_model_data(df=df_test,trainn=False,regr_qual=regr_qual,ohe=ohe,ohe1=ohe1)
    predictions=regr.predict(df_test)
#     print(predictions)
    return r2_score(y1,predictions)

main_pred(best,df,Y)


# In[147]:


from matplotlib import pyplot as plt

plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})
sorted_idx = best.feature_importances_.argsort()
plt.barh(df.columns.values[sorted_idx][-30:], best.feature_importances_[sorted_idx][-30:])
plt.xlabel("Random Forest Feature Importance")


# In[110]:





# In[ ]:





# In[196]:


df_1['Variable04']=df['Variable04']


# In[197]:


df_1.corr()['Variable04'].nlargest(10)


# In[208]:


dfvar3=df_1.loc[df_1.Variable04.notna()]


# In[209]:


dfvar3
y_var3=dfvar3.pop('Variable04')


# In[210]:


X_trainv, X_testv, y_trainv, y_testv = train_test_split(dfvar3, y_var3, test_size=0.25, random_state=42)


# In[212]:


regr = LinearRegression()
regr.fit(X_trainv, y_trainv)
from sklearn.metrics import r2_score
predictions=regr.predict(X_testv)
r2_score(y_testv,predictions)


# In[ ]:




