import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

# load the data
filelist = ['data/tennis_atp/atp_matches_2014.csv',
            'data/tennis_atp/atp_matches_2015.csv',
            'data/tennis_atp/atp_matches_2016.csv',
            'data/tennis_atp/atp_matches_2017.csv',
            'data/tennis_atp/atp_matches_2018.csv']

df_list = []
for filename in filelist:
    df = pd.read_csv(filename)
    df_list.append(df)
df = pd.concat(df_list,ignore_index=True)

print(df)
# preprocess the data
hand_dict = {'R': 0, 'L': 1, 'U': 2, np.nan: 3}
surface_dict = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3, np.nan: 4}
level_dict = {'G': 0, 'M': 1, 'A': 2, 'D': 3, 'C': 4, 'F': 5, 'B': 6, 'L': 7, 'O': 8, 'S': 9, np.nan: 10}

df['winner_hand'] = df['winner_hand'].map(hand_dict)
df['loser_hand'] = df['loser_hand'].map(hand_dict)
df['surface'] = df['surface'].map(surface_dict)
df['tourney_level'] = df['tourney_level'].map(level_dict)

df['winner_rank'] = df['winner_rank'].fillna(0)
df['loser_rank'] = df['loser_rank'].fillna(0)
df['winner_ht'] = df['winner_ht'].fillna(0)
df['loser_ht'] = df['loser_ht'].fillna(0)
df['winner_age'] = df['winner_age'].fillna(0)
df['loser_age'] = df['loser_age'].fillna(0)

# create player attributes dataframe
player_attributes = pd.concat([df[['winner_id', 'winner_rank', 'winner_hand', 'winner_ht', 'winner_age']].rename(columns={'winner_id': 'player_id', 'winner_rank': 'rank', 'winner_hand': 'hand', 'winner_ht': 'ht', 'winner_age': 'age'}),
                               df[['loser_id', 'loser_rank', 'loser_hand', 'loser_ht', 'loser_age']].rename(columns={'loser_id': 'player_id', 'loser_rank': 'rank', 'loser_hand': 'hand', 'loser_ht': 'ht', 'loser_age': 'age'})])

# average player attributes over all matches
player_attributes = player_attributes.groupby('player_id').mean().reset_index()
print(player_attributes.head(-5))
# create edge features dataframe
match_details = df[['winner_id', 'loser_id', 'surface', 'tourney_level', 'match_num']].copy()
match_details['label'] = 1 # label 1 for matches where player 1 (winner) won
match_details_rev = match_details.copy()
match_details_rev['winner_id'], match_details_rev['loser_id'] = match_details['loser_id'], match_details['winner_id']
match_details_rev['label'] = 0 # label 0 for matches where player 1 (loser) lost
edge_features = pd.concat([match_details, match_details_rev])
print(edge_features.head(-5))



# create a mapping from player_id to node_id
player_ids = player_attributes['player_id'].unique()
node_ids = range(len(player_ids))
player_to_node = dict(zip(player_ids, node_ids))

# map the player IDs in your edges DataFrame to the corresponding node IDs
edge_features['winner_node'] = edge_features['winner_id'].map(player_to_node)
edge_features['loser_node'] = edge_features['loser_id'].map(player_to_node)

print(type(edge_features['winner_node']))
print(edge_features['winner_node'].shape)

# create the graph
g = dgl.graph((edge_features['winner_node'].values, edge_features['loser_node'].values))

# re-order player_attributes according to node_ids and convert the DataFrame to tensor
player_attributes = player_attributes.set_index('player_id')
player_attributes = player_attributes.loc[player_ids]
node_features = torch.tensor(player_attributes.values, dtype=torch.float32)

# add node features to the graph
g.ndata['feat'] = node_features

# create edge features
edge_feats = edge_features[['surface', 'tourney_level', 'match_num']]
edge_feats_tensor = torch.tensor(edge_feats.values, dtype=torch.float32)

# add edge features to the graph
g.edata['feat'] = edge_feats_tensor

# create labels and split the dataset into a training set and a test set
labels = torch.tensor(edge_features['label'].values, dtype=torch.float32)
# map player IDs to node indices
player_to_node = {player_id: node_index for node_index, player_id in enumerate(player_attributes.index)}

train_mask, test_mask = train_test_split(range(len(edge_features)), test_size=0.2, random_state=42)

all_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
all_mask[train_mask] = True
g.edata['train_mask'] = all_mask

all_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
all_mask[test_mask] = True
g.edata['test_mask'] = all_mask

g.edata['label'] = labels



class GCNEdge(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super(GCNEdge, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, hidden_size)
        self.edge_func_fc = nn.Linear(2*hidden_size + 3, 1)

    # assume that edge features are static
    def forward(self, g, node_features,edge_features):
        h = F.relu(self.conv1(g, node_features))
        h = self.conv2(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['feat'] = edge_features
            g.apply_edges(self.edge_func)
            return torch.sigmoid(g.edata['h'])  # Apply sigmoid to output for binary classification

    def edge_func(self, edges):
        src = edges.src['h']
        dst = edges.dst['h']
        old_efeat = edges.data['feat']
        e = torch.cat([src, dst, old_efeat], dim=1)  # Concatenate src, dst and old_efeat
        e = self.edge_func_fc(e)  # Apply linear transformation
        return {'h': e.squeeze(-1)}

import matplotlib.pyplot as plt

# Initialize model and optimizer
model = GCNEdge(node_features.shape[1], 32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.BCELoss()

loss_value = []

# Training loop
for epoch in range(200):
    model.train()

    # Forward pass
    logits = model(g, g.ndata['feat'],g.edata['feat'])
    loss = loss_func(logits[g.edata['train_mask']], g.edata['label'][g.edata['train_mask']])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_value.append(loss.item())
    # Print results
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Plot the loss values
plt.figure(figsize=(10,5))
plt.plot(loss_value)
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

def evaluate(model, g, node_features, edge_features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, node_features, edge_features)
        logits = logits[mask]
        print(logits)
        labels = labels[mask]
        print(labels)
        predicted_labels = (torch.round(logits)).long()
        correct = torch.sum(predicted_labels == labels)
        return correct.item() * 1.0 / len(labels)

def predict(model, g, node_features, edge_features):
    model.eval()
    with torch.no_grad():
        logits = model(g, node_features, edge_features)
        probabilities = torch.round(logits)
        return probabilities

train_acc = evaluate(model, g, g.ndata['feat'], g.edata['feat'], g.edata['label'], g.edata['train_mask'])
test_acc = evaluate(model, g, g.ndata['feat'], g.edata['feat'], g.edata['label'], g.edata['test_mask'])
print('Train Accuracy:', train_acc)
print('Test Accuracy:', test_acc)


# For prediction
probabilities = predict(model, g, g.ndata['feat'], g.edata['feat'])
print('Predictions:', probabilities)