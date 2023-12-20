# Javid 

import pandas as pd
import numpy as np
import warnings
import os
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch
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

class LossCallback:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def __call__(self, epoch, iteration, loss):
        global_step = epoch * len(train_loader) + iteration
        self.writer.add_scalar("Training Loss", loss.item(), global_step=global_step)

    def close(self):
        self.writer.close()
        
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

path = f"predicted_dataset/lstm/GW{gameweek}"

# create new single directory
# check whether directory already exists
if not os.path.exists(path):
    os.mkdir(path)
    print("Folder %s created!" % path)
else:
    print("Folder %s already exists" % path)




# x, val, y, y_val = train_test_split(
#     train.drop(leak_columns, axis=1),
#     target["minutes"],
#     test_size=0.1,
#     random_state=0,
# )


# In[36]:


# print(train.keys())


# # Minutes

# In[37]:


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

from sklearn.preprocessing import LabelEncoder

def filter_existing_columns(df, columns):
    return [col for col in columns if col in df.columns]

train["position"]=train["position"].replace({"GKP":"GK" })
test["position"]=test["position"].replace({"GKP":"GK" })

test = test[train.columns]
train_copy = train.copy()
test_copy = test.copy()

train["index"] = train["name"] + train["kickoff_time"].astype("str")
train.drop_duplicates("index", keep="last", inplace=True)
train= train.set_index("index")
train["date"]=pd.to_datetime(train["kickoff_time"])
train["day_of week"]=train["date"].dt.day_name
train["month"]=train["date"].dt.month
train["hour"]=train["date"].dt.hour
train["week"]=train["date"].dt.week
train.drop(["kickoff_time","date"], axis=1, inplace=True)

test["index"] = test["name"] + test["kickoff_time"].astype("str")
test= test.set_index("index")
test["date"]=pd.to_datetime(test["kickoff_time"])
test["day_of week"]=test["date"].dt.day_name
test["month"]=test["date"].dt.month
test["hour"]=test["date"].dt.hour
test["week"]=test["date"].dt.week
test.drop(["kickoff_time","date"], axis=1, inplace=True)
train["minutes"] = train["minutes"].apply(convert_minutes)

target = train[["minutes", "GW","position" ]]
train.drop(["total_points", "minutes"], axis=1, inplace=True)
test.drop(["total_points", "minutes"], axis=1, inplace=True)

for col in train.columns:
    if train[col].dtype == "object":
        if col not in ["team", "name","position"]:
            train[col] = pd.factorize(train[col])[0]
            test[col] = pd.factorize(test[col])[0]

train["was_home"] = train["was_home"].replace({True: 0, False: 1})
test["was_home"] = test["was_home"].replace({True: 0, False: 1})

# train = train[statistics + leak_columns+date_cols]
# test = test[statistics + leak_columns+date_cols]

x, val, y, y_val = train_test_split(
    train.drop(leak_columns, axis=1),
    target["minutes"],
    test_size=0.1,
    random_state=0,
)

params={'colsample_bylevel': 0.8070621518153563, 'learning_rate': 0.04765984972709895, 'max_depth': 7, 'reg_lambda': 5, 'scale_pos_weight': 2.5,'subsample': 0.6794390204583894}
model=CatBoostClassifier(**params,cat_features=["position"],random_state=0,early_stopping_rounds=500,use_best_model=True,verbose=500,n_estimators=10000)

model.fit(x, y,eval_set=[(val,y_val)])

# test_copy["minutes"] = model.predict(test.drop(leak_columns, axis=1))
test_copy["minutes"] = model.predict(test.drop(leak_columns, axis=1))

test_copy[leak_columns + ["minutes"]].to_csv(
    f"minutes.csv"
)

predicted_minutes=model.predict(val)
val_=pd.DataFrame({"ind":val.index,"actul_minutes":y_val,"predicted_minutes":predicted_minutes,"position":val["position"]})

train_copy0 = train_copy
test_copy0 = test_copy


# In[38]:


# print(test_copy0)
# print(np.sum(test_copy['minutes']==1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.cuda.is_available()
# print(device)


# # Points

# In[45]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

from sklearn.preprocessing import LabelEncoder

def filter_existing_columns(df, columns):
    return [col for col in columns if col in df.columns]

train["position"]=train["position"].replace({"GKP":"GK" })
test["position"]=test["position"].replace({"GKP":"GK" })

# train = train_copy[train_copy0["minutes"] > 0]
# test = test_copy[test_copy0["minutes"] > 0]

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

target = train[["total_points", "GW","position" ]]

largest_gw = find_largest_gw('datasets/2023-24/fixtures/')
if largest_gw > gameweek:
    test_target = test[["total_points", "GW","position" ]]
    

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

train = train.fillna(0)
test = test.fillna(0)

label_encoder_name = LabelEncoder()
combined_data = pd.concat([train, test], axis=0)
combined_data['name'] = combined_data['name'].astype(str)
combined_data['name_encoded'] = label_encoder_name.fit_transform(combined_data['name'])

label_encoder_team = LabelEncoder()
combined_data['team_encoded'] = label_encoder_team.fit_transform(combined_data['team'])

label_encoder_position = LabelEncoder()
combined_data['position'] = label_encoder_position.fit_transform(combined_data['position'])


# List of non-numeric columns
non_numeric_cols = ['was_home', 'name', 'position', 'team']

for col in non_numeric_cols:
    if col == 'name':
        train[col] = label_encoder_name.transform(train[col])
        test[col] = label_encoder_name.transform(test[col])
    elif col == 'team':
        train[col] = label_encoder_team.transform(train[col])
        test[col] = label_encoder_team.transform(test[col])
    elif col == 'position':
        train[col] = label_encoder_position.transform(train[col])
        test[col] = label_encoder_position.transform(test[col])


name_mapping = dict(zip(label_encoder_name.classes_, label_encoder_name.transform(label_encoder_name.classes_)))
team_mapping = dict(zip(label_encoder_team.classes_, label_encoder_team.transform(label_encoder_team.classes_)))
position_mapping = dict(zip(label_encoder_position.classes_, label_encoder_position.transform(label_encoder_position.classes_)))

# Invert the dictionaries to map back from encoded values to original strings
name_id_to_string = {v: k for k, v in name_mapping.items()}
team_id_to_string = {v: k for k, v in team_mapping.items()}
position_id_to_string = {v: k for k, v in position_mapping.items()}

# train = train_copy[train_copy0["minutes"] > 0]
# test = test_copy[test_copy0["minutes"] > 0]

# Define a custom PyTorch Dataset
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create datasets and data loaders
train_dataset = CustomDataset(train, target["total_points"])
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)



import torch.nn.functional as F

import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F


class FlexibleRegressionModel2(nn.Module):
    def __init__(self, input_size):
        super(FlexibleRegressionModel2, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(832, 128)  # Adjusted the linear layer to 128
        self.fc2 = nn.Linear(128, 64)  # Updated subsequent layers' sizes
        self.fc3 = nn.Linear(64, 1)  # Output layer
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LSTMRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMRegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)  # Output layer
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # Output shape: (batch_size, seq_length, hidden_size)
        
        # Use the output of the last time step for regression
        out = out[:, -1, :]  # Get output of the last time step
        
        # Apply fully connected layers
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
        
class FlexibleRegressionModel1(nn.Module):
    def __init__(self, input_size):
        super(FlexibleRegressionModel1, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(832, 128)  # Adjusted the linear layer to 128
        self.fc2 = nn.Linear(128, 1)  # Output layer
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


from torch.utils.tensorboard import SummaryWriter

class LossCallback:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.update_frequency = 1000
        self.running_loss = 0.0

    def __call__(self, epoch, iteration, loss):
        global_step = epoch * len(train_loader) + iteration
        self.running_loss += loss.item()

        if iteration % self.update_frequency == (self.update_frequency - 1):  # Log every 100 iterations
            average_loss = self.running_loss / self.update_frequency
            self.writer.add_scalar("Training Loss", average_loss, global_step=global_step)
            self.running_loss = 0.0  # Reset running loss

    def close(self):
        self.writer.close()

# model = FlexibleRegressionModelWithGRU(input_size=train.shape[1],hidden_size=10)
model = FlexibleRegressionModel1(input_size=train.shape[1])
model = model.to(device)  # Move the model to CUDA

input_size = train.shape[1]  # Update this with your actual input size
hidden_size = 64  # Define the size of the LSTM hidden state

# Create an instance of LSTMRegressionModel
# lstm_model = LSTMRegressionModel(input_size=input_size, hidden_size=hidden_size)
# model = lstm_model.to(device) 

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

log_dir = "runs"  # Directory to save logs
loss_callback = LossCallback(log_dir)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the selected device
        # print(inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))  # Unsqueeze to match output shape
        loss.backward()
        optimizer.step()

        loss_callback(epoch, i, loss)

loss_callback.close()


# In[46]:


# print(train.shape,target["total_points"].shape)
# print(target["total_points"].shape)
print(largest_gw,gameweek)


# In[47]:


# Evaluation on the test set
if largest_gw > gameweek:
    test_dataset = CustomDataset(test, test_target["total_points"])
else:    
    test_dataset = CustomDataset(test, pd.Series([0] * test.shape[0]))

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
all_predictions = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the selected device
        outputs = model(inputs)
        all_predictions.extend(outputs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())  # Move targets to CPU before conversion

# Calculate and print the RMSE
performance = mean_squared_error(all_targets, all_predictions, squared=False)
# print(f"Root Mean Squared Error (RMSE): {rmse}")


# In[52]:

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
    if pd.isnull(predictions_df2.loc[gameweek, 'lstm']) or predictions_df2.loc[gameweek, 'lstm']==0:
        predictions_df2.loc[gameweek-1, 'lstm'] = performance
    else:
        predictions_df2.loc[gameweek-1, 'lstm'] += performance


# In[49]:




# In[24]:


# print(train.keys().tolist())
# print(test.keys().tolist())
# Assuming you have generated predictions in all_predictions variable

import numpy as np
import pandas as pd

all_predictions = np.concatenate(all_predictions, axis=0)

# Combine the predictions with the test data
test["points"] = all_predictions

# Assuming leak_columns is a list of columns to include in the CSV file
columns_to_include = leak_columns + ["points", "value"]


test['name'] = test['name'].map(name_id_to_string)
test['team'] = test['team'].map(team_id_to_string)
test['position'] = test['position'].map(position_id_to_string)


# 2.7320675301713573
# 1.6917027774360753

# In[ ]:





# In[25]:


test[leak_columns + ["points", "value"]].sort_values(
    "points", ascending=False
).to_csv("points.csv")


# In[26]:


test[test["position"]=="MID"].sort_values(by="points",ascending=False).head(11)[["name","points","team"]]


# In[27]:


test[test["position"]=="DEF"].sort_values(by="points",ascending=False).head(10)[["name","points","team"]]


# In[28]:


test[test["position"]=="GK"].sort_values(by="points",ascending=False).head(10)[["name","points","team"]]


# In[29]:


test[test["position"]=="FWD"].sort_values(by="points",ascending=False).head(10)[["name","points","team"]]


# In[30]:


test["points"].sort_values(ascending=False).head(50)


# #Save predictions

# In[31]:


test_copy0[test_copy0["position"]=="DEF"][["name","team","minutes"]].to_csv(f"predicted_dataset/lstm/GW{gameweek}/defenders_minutes.csv")
test_copy0[test_copy0["position"]=="GK"][["name","team","minutes"]].to_csv(f"predicted_dataset/lstm/GW{gameweek}/goalkeepers_minutes.csv")
test_copy0[test_copy0["position"]=="MID"][["name","team","minutes"]].to_csv(f"predicted_dataset/lstm/GW{gameweek}/midfielders_minutes.csv")
test_copy0[test_copy0["position"]=="FWD"][["name","team","minutes"]].to_csv(f"predicted_dataset/lstm/GW{gameweek}/forwards_minutes.csv")


# In[32]:


test[test["position"]=="DEF"][["name","team","points","value"]].sort_values(by="points",ascending=False).to_csv(f"predicted_dataset/lstm/GW{gameweek}/defenders_points.csv")
test[test["position"]=="GK"][["name","team","points","value"]].sort_values(by="points",ascending=False).to_csv(f"predicted_dataset/lstm/GW{gameweek}/goalkeepers_points.csv")
test[test["position"]=="MID"][["name","team","points","value"]].sort_values(by="points",ascending=False).to_csv(f"predicted_dataset/lstm/GW{gameweek}/midfielders_points.csv")
test[test["position"]=="FWD"][["name","team","points","value"]].sort_values(by="points",ascending=False).to_csv(f"predicted_dataset/lstm/GW{gameweek}/forwards_points.csv")


# In[18]:


test_copy[test_copy["position"]=="DEF"][["name","team","minutes"]]


# In[ ]:




