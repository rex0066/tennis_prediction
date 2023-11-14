import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchsummary as summary

from sklearn.preprocessing import StandardScaler
data_2014 = pd.read_csv('/data/tennis_atp/atp_matches_2014.csv')
data_2015 = pd.read_csv('/data/tennis_atp/atp_matches_2015.csv')
data_2016 = pd.read_csv('/data/tennis_atp/atp_matches_2016.csv')
data_2017 = pd.read_csv('/data/tennis_atp/atp_matches_2017.csv')
# data_2018 = pd.read_csv('data/tennis_atp-master/atp_matches_2018.csv')
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


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BinaryClassificationModel, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


input_dim = 14
hidden_dim = 256


model = BinaryClassificationModel(input_dim, hidden_dim)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

features = torch.tensor(features_list, dtype=torch.float32)
labels = torch.tensor(labels_list, dtype=torch.long)

features[torch.isnan(features)] = 0


assert not torch.isnan(features).any()
assert not torch.isinf(features).any()
input_data = features

labels = labels
scaler = StandardScaler()
ft = scaler.fit_transform(input_data)
# print(ft)
input_data = torch.tensor(ft, dtype=torch.float32)


train_data = input_data[:16000]
train_target = labels[:16000]
test_data = input_data[16000:]
test_target = labels[16000:]

epochs = 100
  
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(train_data)
    #print(outputs)
    #print(train_target)
    loss = criterion(outputs, train_target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(test_data)
    _, predicted = torch.max(outputs.data, 1)
    total += test_target.size(0)
    correct += (predicted == test_target).sum().item()
    print(f"Accuracy: {100 * correct / total}%")
