import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchsummary as summary

data_2014 = pd.read_csv('/data/tennis_atp/atp_matches_2014.csv')
data_2015 = pd.read_csv('/data/tennis_atp/atp_matches_2015.csv')
data_2016 = pd.read_csv('/data/tennis_atp/atp_matches_2016.csv')
data_2017 = pd.read_csv('/data/tennis_atp/atp_matches_2017.csv')
data = pd.concat([data_2014, data_2015, data_2016, data_2017])

data.fillna(
    {
        "w_ace": 0,
        "w_df": 0,
        "w_svpt": 0,
        "w_1stIn": 0,
        "w_1stWon": 0,
        "w_2ndWon": 0,
        "w_SvGms": 0,
        "w_bpSaved": 0,
        "w_bpFaced": 0,
        "winner_age": 25,
        "winner_ht": 175,
        "winner_rank": 1000,
        "winner_rank_points": 0,
        "l_ace": 0,
        "l_df": 0,
        "l_svpt": 0,
        "l_1stIn": 0,
        "l_1stWon": 0,
        "l_2ndWon": 0,
        "l_SvGms": 0,
        "l_bpSaved": 0,
        "l_bpFaced": 0,
        "loser_age": 25,
        "loser_ht": 175,
        "loser_rank": 1000,
        "loser_rank_points": 0,
        "winner_hand": 0,
        "loser_hand": 0,
    },
    inplace=True,
)

data["winner_hand"] = data["winner_hand"].map({"R": 0, "L": 1, "U": 2})
data["loser_hand"] = data["loser_hand"].map({"R": 0, "L": 1, "U": 2})

features_list = []
labels_list = []

node_id = 0
for index, row in data.iterrows():
    # Winner node
    winner_features = [
        row["w_ace"],
        row["w_df"],
        row["w_svpt"],
        row["w_1stIn"],
        row["w_1stWon"],
        row["w_2ndWon"],
        row["w_SvGms"],
        row["w_bpSaved"],
        row["w_bpFaced"],
        row["winner_age"],
        row["winner_ht"],
        row["winner_rank"],
        row["winner_rank_points"],
        row["winner_hand"],
    ]
    # print(winner_features)
    features_list.append(winner_features)
    labels_list.append(1)  # winner label

    # Loser node
    loser_features = [
        row["l_ace"],
        row["l_df"],
        row["l_svpt"],
        row["l_1stIn"],
        row["l_1stWon"],
        row["l_2ndWon"],
        row["l_SvGms"],
        row["l_bpSaved"],
        row["l_bpFaced"],
        row["loser_age"],
        row["loser_ht"],
        row["loser_rank"],
        row["loser_rank_points"],
        row["loser_hand"],
    ]
    features_list.append(loser_features)
    labels_list.append(0)  # loser label
features = torch.tensor(features_list, dtype=torch.float32)
labels = torch.tensor(labels_list, dtype=torch.long)

# Extract features and target
X = features
X[torch.isnan(X)] = 0
y = labels


# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Train logistic regression model
clf = LogisticRegressionCV(cv=3, max_iter=3, solver="lbfgs")
clf.fit(X_train, y_train)

# 5. Make predictions
y_pred = clf.predict(X_test)

mean_accuracies = np.array(clf.scores_[1]).mean(axis=0)
print(mean_accuracies)
plt.plot(clf.Cs_, mean_accuracies)
plt.xlabel("Inverse Regularization Strength (C)")
plt.ylabel("Mean Accuracy")
plt.title("Mean Accuracy for Each Fold")
plt.show()
