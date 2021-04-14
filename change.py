
import datetime
import time
# change
def obj_handle(df):
    for col in ['MSSubClass', 'MoSold']:
        df[col] = df[col].astype(object)
    return df
# add

def drop_useless(dff):
    dff=dff.drop('Utilities',axis=1)
    dff=dff.drop('Street',axis=1)
    return dff
# add in makedata
    df=drop_useless(df)

# add в обучение


def main_pred(regr,df):
    df_test=pd.read_csv(r'C:\Users\Администратор\Desktop\REAskills2021\train.csv')

    df_test=df_test.drop('Unnamed: 0', axis=1)

    df_test=make_model_data(df=df_test,trainn=False,regr_qual=regr_qual,ohe=ohe,ohe1=ohe1)
    predictions=regr.predict(df_test)
    return predictions
def gridsearch_and_pred():
    param_grid = {
    'bootstrap': [True],
    'max_features': ['auto'],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [6,7,8],
    'n_estimators': [80,90, 100,110]
    }
    rf = RandomForestRegressor(random_state=73)
    grid = GridSearchCV(estimator = rf, param_grid = param_grid, scoring='r2',
                          cv = 5, n_jobs = -1, verbose = 2)
    grid.fit(df, Y)
    print('score',grid.best_score_)
    best=grid.best_estimator_
    best = RandomForestRegressor(**grid.best_params_,random_state=73)
    best.fit(df, Y)
    predictions=main_pred(best,df)
    pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction_%s.csv'%round(time.time()))

    return predictions
# add to end
predictions=gridsearch_and_pred()