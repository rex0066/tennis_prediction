import pandas as pd
import networkx as nx
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# load data

data_2014 = pd.read_csv('/data/tennis_atp/atp_matches_2014.csv')
data_2015 = pd.read_csv('/data/tennis_atp/atp_matches_2015.csv')
data_2016 = pd.read_csv('/data/tennis_atp/atp_matches_2016.csv')
data_2017 = pd.read_csv('/data/tennis_atp/atp_matches_2017.csv')
data_2018 = pd.read_csv('/data/tennis_atp/atp_matches_2018.csv')
data_2019 = pd.read_csv('/data/tennis_atp/atp_matches_2019.csv')
#data_2020 = pd.read_csv('/data/tennis_atp/atp_matches_2020.csv')
#data_2021= pd.read_csv('/data/tennis_atp/atp_matches_2021.csv')


data = pd.concat([data_2014,data_2015,data_2016,data_2017,data_2018,data_2019])

# data preprocessing
data.fillna({
    'w_ace': 0, 'w_df': 0, 'w_svpt': 0, 'w_1stIn': 0,
    'w_1stWon': 0, 'w_2ndWon': 0, 'w_SvGms': 0, 'w_bpSaved': 0,
    'w_bpFaced': 0, 'winner_age': 25, 'winner_ht': 175, 'winner_rank': 1000,
    'winner_rank_points': 0, 'l_ace': 0, 'l_df': 0, 'l_svpt': 0,
    'l_1stIn': 0, 'l_1stWon': 0, 'l_2ndWon': 0, 'l_SvGms': 0,
    'l_bpSaved': 0, 'l_bpFaced': 0, 'loser_age': 25, 'loser_ht': 175,
    'loser_rank': 1000, 'loser_rank_points': 0,'winner_hand' :0, 'loser_hand': 0
}, inplace=True)

data['winner_hand'] = data['winner_hand'].map({'R': 0, 'L': 1, 'U': 2})
data['loser_hand'] = data['loser_hand'].map({'R': 0, 'L': 1, 'U': 2})

print(data)


from collections import defaultdict

G = nx.Graph()
last_match_node_ids = defaultdict(list)

# store labels and features
features_list = []
labels_list = []

node_id = 0
for index, row in data.iterrows():
    # Winner node
    winner_features = [
        row['w_ace'], row['w_df'], row['w_svpt'], row['w_1stIn'],
        row['w_1stWon'], row['w_2ndWon'], row['w_SvGms'], row['w_bpSaved'],
        row['w_bpFaced'], row['winner_age'], row['winner_ht'], row['winner_rank'],
        row['winner_rank_points'], row['winner_hand']
    ]
    #print(winner_features)
    features_list.append(winner_features)
    labels_list.append(1)  # winner label

    G.add_node(node_id)

    if row['winner_id'] in last_match_node_ids:
        last_match_node_ids[row['winner_id']].append(node_id)
    else:
        last_match_node_ids[row['winner_id']] = [node_id]
    node_id += 1

    # Loser node
    loser_features = [
        row['l_ace'], row['l_df'], row['l_svpt'], row['l_1stIn'],
        row['l_1stWon'], row['l_2ndWon'], row['l_SvGms'], row['l_bpSaved'],
        row['l_bpFaced'], row['loser_age'], row['loser_ht'], row['loser_rank'],
        row['loser_rank_points'], row['loser_hand']
    ]
    features_list.append(loser_features)
    labels_list.append(0)  # loser label

    G.add_node(node_id)

    if row['loser_id'] in last_match_node_ids:
        last_match_node_ids[row['loser_id']].append(node_id)
    else:
        last_match_node_ids[row['loser_id']] = [node_id]

    node_id += 1

    # Edge between winner and loser
    G.add_edge(node_id - 2, node_id - 1)

# Convert features and labels to tensors
#print(features_list)
# for idx, item in enumerate(features_list):
#     if not isinstance(item, (int, float)):
#         print(f"Non-numeric value at index {idx}: {item}")
features = torch.tensor(features_list, dtype=torch.float32)
labels = torch.tensor(labels_list, dtype=torch.long)
#print(labels)
# print(last_match_node_ids)

# iterate last_match_node_ids
for player_id, node_ids_list in last_match_node_ids.items():

    print(f"Player ID: {player_id}, Node IDs: {node_ids_list}")

add_extra_edges = True

for player_id, node_ids_list in last_match_node_ids.items():
    # only choose previous 3 nodes
    node_ids_to_connect = node_ids_list[:3]

    #  Creates edge between two nodes if true
    if add_extra_edges:
        for i in range(len(node_ids_to_connect)):
            for j in range(i+1, len(node_ids_to_connect)):
                src_node_id = node_ids_to_connect[i]
                dst_node_id = node_ids_to_connect[j]

                # 使用add_edge函数来添加边
                G.add_edge(src_node_id, dst_node_id)

print(G.adj)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size,num_classes):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size)
        self.conv2 = dgl.nn.GraphConv(hidden_size, num_classes)
        # self.conv3 = dgl.nn.GraphConv(hidden_size2, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

net = GCN(14, 28, 2)  # 14 features input, hidden size of 28, 2 classes (winner/loser)

H = G.subgraph(list(range(0, 50)))
nx.draw(H,with_labels=True,node_color='lightblue', edge_color='gray')

G = dgl.from_networkx(G)
print(G)

G.ndata['feat'] = features
G.ndata['label'] = labels
print(features.shape)
print(G.ndata['feat'])
print(G)

import random
train_mask = torch.zeros(len(features), dtype=torch.bool)
train_indices = random.sample(range(len(features)), int(0.8 * len(features)))
train_mask[train_indices] = True
test_mask = ~train_mask
G.ndata['train_mask'] = train_mask
G.ndata['test_mask'] = test_mask
print(G.ndata['train_mask'])

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

net.apply(weights_init)

#print(torch.isnan(G.ndata['feat']).any())
nan_mask = torch.isnan(G.ndata['feat'])
G.ndata['feat'][nan_mask] = 1
# Use torch.nonzero() to find the index position of the nan value
nan_indices = torch.nonzero(nan_mask)

#print(nan_indices)
#print(G.ndata['feat'][5381])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
ft = scaler.fit_transform(G.ndata['feat'])
print(ft)
G.ndata['feat'] = torch.tensor(ft, dtype=torch.float32)

torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)

optimizer = torch.optim.Adam(net.parameters(), lr=0.2)
criterion = nn.CrossEntropyLoss()

losses = []

for epoch in range(200):
    net.train()
    logits = net(G, G.ndata['feat'])
    #print(logits)
    loss = criterion(logits[G.ndata['train_mask']], labels[G.ndata['train_mask']])
    #print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    #print(f"Epoch {epoch} | Loss: {loss.item()}")

    if epoch % 20 == 0:
        net.eval()
        logits = net(G, G.ndata['feat'])
        train_acc = ((logits[G.ndata['train_mask']].argmax(1) == G.ndata['label'][G.ndata['train_mask']].long()).float().sum() / len(train_indices)).item()
        test_acc = ((logits[G.ndata['test_mask']].argmax(1) == G.ndata['label'][G.ndata['test_mask']].long()).float().sum() / (len(features)-len(train_indices))).item()
        print(f"Epoch {epoch}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()

net.eval()
logits = net(G, G.ndata['feat'])
predicted = logits.argmax(1)
accuracy = (predicted[test_mask] == labels[test_mask]).float().mean().item()
print(f"Accuracy: {accuracy:.4f}")