#!/usr/bin/env python
# Javid 


import pandas as pd
import numpy as np
import warnings
import os
from utils import *

import argparse
parser = argparse.ArgumentParser(description='Process gameweek')
parser.add_argument('--gameweek', type=int, help='Gameweek value', default=16)
args = parser.parse_args()
gameweek = args.gameweek

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold,KFold

def replace_spaces_with_underscore(column_list):
    return [col.replace(' ', '_') for col in column_list]

# Applying the transformation to the existing lists

from hyperopt import tpe,hp,fmin,STATUS_OK,Trials
from hyperopt.pyll.base import scope

def convert_minutes(val):
    """CONVERTS MINUTES TO A CATEGORICAL OUTPUT"""
    if val > 10:
        return 1
    else:
        return 0


forward_statistics =['value', 'was_home', 'last_season_position', 'percent_value',
       'position rank', 'goals_scored_ex', 'assists_ex', 'total_points_ex',
       'minutes_ex', 'goals_conceded_ex', 'creativity_ex', 'influence_ex',
       'threat_ex', 'bonus_ex', 'bps_ex', 'ict_index_ex', 'now_cost_ex', 'GW', 'opponent_last_season_position',
        'mean assists 3','mean bonus 3', 'mean bps 3','mean creativity 3', 'mean goals_scored 3',
       'mean ict_index 3', 'mean influence 3', 'mean minutes 3', 'mean penalties_missed 3',  'mean threat 3',
       'mean total_points 3','mean value 3', 'mean match_result 3', 'std bps 3', 'std creativity 3',
       'std ict_index 3', 'std influence 3', 'std minutes 3',
       'std threat 3', 'std total_points 3', 'std value 3']

leak_columns = [
    "name",
    "team",
]  # columns that shouldnt be used in training fir fear of data leakage


dropped_columns = [
    "season",
    "opponent",
    "match_result",
   # "position",
    "assists",
    "penalties_missed",
    "bonus",
    "bps",
    "clean_sheets",
    "creativity",
    "goals_conceded",
    "goals_scored",
    "ict_index",
    "influence",
    "own_goals",
    "penalties_saved",
    "red_cards",
    "saves",
    "selected",
    "threat",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
    "yellow_cards"
]  # "value",

midfielder_statistics =['value', 'was_home', 'last_season_position', 'percent_value',
       'position rank', 'goals_scored_ex', 'assists_ex', 'total_points_ex',
       'minutes_ex', 'goals_conceded_ex', 'creativity_ex', 'influence_ex',
       'threat_ex', 'bonus_ex', 'bps_ex', 'ict_index_ex', 'now_cost_ex', 'GW', 'opponent_last_season_position',
        'mean assists 3','mean bonus 3', 'mean bps 3','mean creativity 3', 'mean goals_scored 3',
       'mean ict_index 3', 'mean influence 3', 'mean minutes 3', 'mean penalties_missed 3',  'mean threat 3',
       'mean total_points 3','mean value 3', 'mean match_result 3', 'std bps 3', 'std creativity 3',
       'std ict_index 3', 'std influence 3', 'std minutes 3',
       'std threat 3', 'std total_points 3', 'std value 3']

goalkeeper_statistics = ['value', 'was_home', 'last_season_position', 'percent_value',
       'position rank', 'total_points_ex', 'minutes_ex', 'goals_conceded_ex',
       'bonus_ex', 'bps_ex', 'ict_index_ex', 'clean_sheets_ex',
       'red_cards_ex', 'now_cost_ex', 'GW', 'opponent_last_season_position',
       'mean bonus 3', 'mean bps 3', 'mean clean_sheets 3', 'mean goals_conceded 3',
       'mean ict_index 3',  'mean minutes 3',
       'mean own_goals 3',  'mean penalties_saved 3',
        'mean saves 3',  'mean threat 3',
       'mean total_points 3',
       'mean value 3', 'mean match_result 3', 'std bps 3',
       'std ict_index 3', 'std influence 3', 'std minutes 3',
       'std threat 3', 'std total_points 3', 'std value 3']

statistics =['value', 'position','was_home', 'last_season_position', 'percent_value',
       'position rank', 'goals_scored_ex', 'assists_ex', 'total_points_ex',
       'minutes_ex', 'goals_conceded_ex', 'creativity_ex', 'influence_ex',
       'threat_ex', 'bonus_ex', 'bps_ex', 'ict_index_ex', 'clean_sheets_ex',
       'yellow_cards_ex','now_cost_ex', 'GW', 'opponent_last_season_position', 'mean assists 3',
       'mean bonus 3', 'mean bps 3', 'mean clean_sheets 3',
       'mean creativity 3', 'mean goals_conceded 3', 'mean goals_scored 3',
       'mean ict_index 3', 'mean influence 3', 'mean minutes 3',
       'mean own_goals 3',
       'mean red_cards 3',  'mean threat 3','mean total_points 3',
       'mean value 3', 'mean match_result 3', 'std bps 3', 'std creativity 3',
       'std ict_index 3', 'std influence 3', 'std minutes 3',
       'std threat 3', 'std total_points 3', 'std value 3','mean saves 3',"mean assists all",
       'mean bonus all', 'mean bps all', 'mean clean_sheets all',
       'mean creativity all', 'mean goals_conceded all', 'mean goals_scored all',
       'mean ict_index all', 'mean influence all', 'mean minutes all',
       'mean own_goals all',
       'mean red_cards all',  'mean threat all','mean total_points all',
       'mean value all', 'mean match_result all',
       'mean team Goal scored 3','mean team Goal scored all','mean team Goal conceded 3','mean team Goal conceded all',"ratio_goal_scored all","ratio_goal_scored 3",
       'opp mean team Goal scored 3','opp mean team Goal conceded 3','opp mean team Goal scored all','opp mean team Goal conceded all',"opp mean match_result all"]


forward_statistics = replace_spaces_with_underscore(forward_statistics)
midfielder_statistics = replace_spaces_with_underscore(midfielder_statistics)
goalkeeper_statistics = replace_spaces_with_underscore(goalkeeper_statistics)
statistics = replace_spaces_with_underscore(statistics)

# statistics = replace_spaces_with_underscore(statistics)

date_cols=["day_of_week","month","hour","week"]

path = f"predicted_dataset/lgbm/GW{gameweek}"

# create new single directory
# check whether directory already exists
if not os.path.exists(path):
    os.mkdir(path)
    print("Folder %s created!" % path)
else:
    print("Folder %s already exists" % path)
    


# In[50]:


train = pd.read_csv("cleaned_dataset/cleaned_previous_seasons.csv",header=0,index_col=0)#/content/drive/MyDrive/Fplpredict /cleaned_previous_seasons.csv", index_col=0)
# train.columns = train.columns.str.strip()
train.columns = replace_spaces_with_underscore(train.columns.tolist())

old_gameweek_cleaned = []
for i in range(1, gameweek):
    # old_gameweek_cleaned.append(pd.read_csv(f"cleaned_dataset/2023-24/GW{i}.csv"))
    df = pd.read_csv(f"cleaned_dataset/2023-24/GW{i}.csv")
    # Replace spaces with underscores in column names
    df.columns = df.columns.str.replace(' ', '_')
    old_gameweek_cleaned.append(df)

old_gameweeks = pd.concat(old_gameweek_cleaned)[train.columns]
train = pd.concat([train, old_gameweeks])

# data for current gameweek we want to predict on
test = pd.read_csv(f"cleaned_dataset/2023-24/GW{gameweek}.csv", header=0,index_col=0)
# test.columns = test.columns.str.strip()
test.columns = replace_spaces_with_underscore(test.columns.tolist())


# In[51]:


def filter_existing_columns(df, columns):
    return [col for col in columns if col in df.columns]

train["position"]=train["position"].replace({"GKP":"GK" })

test = test[train.columns]
train_copy = train.copy()
test_copy = test.copy()

# predict points
train["index"] = train["name"] + train['kickoff_time'].astype("str")
train.drop_duplicates("index", keep="last", inplace=True)
train= train.set_index("index")
train["date"]=pd.to_datetime(train['kickoff_time'])
train["day_of_week"]=train["date"].dt.day_name
train["month"]=train["date"].dt.month
train["hour"]=train["date"].dt.hour
train["week"]=train["date"].dt.week
train.drop(['kickoff_time',"date"], axis=1, inplace=True)

test["index"] = test["name"] + test['kickoff_time'].astype("str")
test = test.set_index("index")
test["date"]=pd.to_datetime(test['kickoff_time'])
test["day_of_week"]=test["date"].dt.day_name
test["month"]=test["date"].dt.month
test["hour"]=test["date"].dt.hour
test["week"]=test["date"].dt.week
test.drop(['kickoff_time',"date"], axis=1, inplace=True)

largest_gw = find_largest_gw('datasets/2023-24/fixtures/')
if largest_gw > gameweek:
    test_target = test[["total_points", "GW","position" ]]
    
target = train[["total_points", "GW","position" ]]
train.drop(["total_points", "minutes"], axis=1, inplace=True)
test.drop(["total_points", "minutes"], axis=1, inplace=True)
train.drop(dropped_columns, axis=1, inplace=True)
test.drop(dropped_columns, axis=1, inplace=True)


for col in train.columns:
    if train[col].dtype == "object":
        if col not in ["team", "name","position"]:
            train[col] = pd.factorize(train[col])[0]
            test[col] = pd.factorize(test[col])[0]

train["position"]=train["position"].astype("category")
test["position"]=test["position"].astype("category")
train["was_home"] = train["was_home"].replace({True: 0, False: 1})
test["was_home"] = test["was_home"].replace({True: 0, False: 1})


selected_columns = statistics + leak_columns + date_cols
selected_columns_filtered = filter_existing_columns(train, selected_columns)
train = train[selected_columns_filtered]
test = test[selected_columns_filtered]

# train = train[statistics + leak_columns+date_cols]
# test= test[statistics + leak_columns+date_cols]

x, val, y, y_val = train_test_split(
    train.drop(leak_columns, axis=1),
    target["total_points"],
    test_size=0.1,
    random_state=0,
)


# In[52]:


from sklearn.model_selection import KFold

#cross_validator to splite the data into folds
folds=KFold(n_splits=8,shuffle=True,random_state=0)

#a dataframe to store the predictions made by each fold
predictions_df=pd.DataFrame()

#list to save the mean absolute errors from validatingon each folds
rmse_val=[]
rmse_X=[]

#a simple catboost regressor
model=LGBMRegressor(**{'colsample_bytree': 0.4199299182268318, 'learning_rate': 0.0032874466037521254, 'max_depth': 9, 'min_split_gain': 0.5685369160138952, 'num_leaves': 99, 'reg_alpha': 0.5621526419488447, 'reg_lambda': 0, 'subsample': 0.6534153111773866}, verbose=-50,random_state=0,early_stopping_rounds=200,n_estimators=10000)

#train model, make predictions and check the validation accuracy on  each fold
for i,(train_index,test_index) in enumerate(folds.split(train.drop(leak_columns, axis=1),target["total_points"])):
    train_fold=train.drop(leak_columns, axis=1).iloc[train_index]
    val_fold=train.drop(leak_columns, axis=1).iloc[test_index]
    y_fold=target["total_points"].iloc[train_index]
    y_val_fold=target["total_points"].iloc[test_index]


    model.fit(train_fold,y_fold,eval_set=[(val_fold,y_val_fold)])
    prediction=model.predict(test.drop(leak_columns, axis=1))
    predictions_df[i]=prediction
    rmse_val.append(mean_squared_error(model.predict(val_fold),y_val_fold,squared=False))
    rmse_X.append(mean_squared_error(model.predict(train_fold),y_fold,squared=False))
preds = np.mean(predictions_df, axis=1).values

import numpy as np

if largest_gw > gameweek:
    # test_inputs = np.array(test.values, dtype=np.float32)
    test_targets = np.array(test_target["total_points"].values, dtype=np.float32)
else:
    # test_inputs = np.array(test.values, dtype=np.float32)
    test_targets = np.zeros((test.shape[0],), dtype=np.float32)


performance = mean_squared_error(preds,test_targets)
# print(mean_squared_error(preds,test_target['total_points']))

# print(rmse_val)
# print(rmse_X)


# In[54]:


import os
import pandas as pd
from sklearn.metrics import mean_squared_error

file_path = 'performance.csv'
if not os.path.exists(file_path):
    initial_data = {'lgbm': [0], 'lasso': [0], 'cnn': [0], 'lstm': [0]}
    pd.DataFrame(initial_data).to_csv(file_path, index=False)

predictions_df2 = pd.read_csv(file_path)
for i in range(predictions_df2.index.max() + 1, gameweek+1):
    if i not in predictions_df2.index:
        predictions_df2.loc[i] = [0, 0, 0,0]
    else:
        missing_columns = set(['lgbm', 'lasso', 'cnn','lstm']) - set(predictions_df2.columns)
        for col in missing_columns:
            predictions_df2.loc[i, col] = 0

# performance = mean_squared_error(preds, test_target['total_points'])
if 'lasso' in predictions_df2.columns:
    if pd.isnull(predictions_df2.loc[gameweek, 'lgbm']) or predictions_df2.loc[gameweek, 'lgbm']==0:
        predictions_df2.loc[gameweek-1, 'lgbm'] = performance
    else:
        predictions_df2.loc[gameweek-1, 'lgbm'] += performance

# In[31]:

predictions_df2.to_csv(file_path, index=False)


# In[39]:


gameweek


# 2.7320675301713573
# 1.6917027774360753

# In[55]:


test["points"]=np.mean(predictions_df, axis=1).values

test[leak_columns + ["points", "value"]].sort_values(
    "points", ascending=False
).to_csv("points.csv")


# In[56]:


test[test["position"]=="MID"].sort_values(by="points",ascending=False).head(11)[["name","points","team"]]


# In[57]:


test[test["position"]=="DEF"].sort_values(by="points",ascending=False).head(10)[["name","points","team"]]


# In[58]:


test[test["position"]=="GKP"].sort_values(by="points",ascending=False).head(10)[["name","points","team"]]


# In[59]:


test[test["position"]=="FWD"].sort_values(by="points",ascending=False).head(10)[["name","points","team"]]


# In[60]:


test["points"].sort_values(ascending=False).head(50)


# In[61]:


# feature_importance.tail(30)


# #Save predictions

# In[62]:


test_copy[test_copy["position"]=="DEF"][["name","team","minutes"]].to_csv(f"predicted_dataset/lgbm/GW{gameweek}/defenders_minutes.csv")
test_copy[test_copy["position"]=="GKP"][["name","team","minutes"]].to_csv(f"predicted_dataset/lgbm/GW{gameweek}/goalkeepers_minutes.csv")
test_copy[test_copy["position"]=="MID"][["name","team","minutes"]].to_csv(f"predicted_dataset/lgbm/GW{gameweek}/midfielders_minutes.csv")
test_copy[test_copy["position"]=="FWD"][["name","team","minutes"]].to_csv(f"predicted_dataset/lgbm/GW{gameweek}/forwards_minutes.csv")


# In[63]:


test[test["position"]=="DEF"][["name","team","points","value"]].sort_values(by="points",ascending=False).to_csv(f"predicted_dataset/lgbm/GW{gameweek}/defenders_points.csv")
test[test["position"]=="GKP"][["name","team","points","value"]].sort_values(by="points",ascending=False).to_csv(f"predicted_dataset/lgbm/GW{gameweek}/goalkeepers_points.csv")
test[test["position"]=="MID"][["name","team","points","value"]].sort_values(by="points",ascending=False).to_csv(f"predicted_dataset/lgbm/GW{gameweek}/midfielders_points.csv")
test[test["position"]=="FWD"][["name","team","points","value"]].sort_values(by="points",ascending=False).to_csv(f"predicted_dataset/lgbm/GW{gameweek}/forwards_points.csv")


# In[64]:


test_copy[test_copy["position"]=="DEF"][["name","team","minutes"]]


# In[ ]:




