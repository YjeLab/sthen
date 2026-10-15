import os

import pickle

import numpy as np

import torch

from torch import nn

from torch.utils.data import DataLoader, Dataset

import pandas as pd



# use scikit-network instead of pymetis for graph partitioning

def graph_split(num_nodes, adj, patch_size):

    '''

    graph partitioning with scikit-network

    '''

    import numpy as np

    from sklearn.cluster import SpectralClustering

    

    # build the adjacency matrix

    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i, neighbors in enumerate(adj):

        for j in neighbors:

            adj_matrix[i, j] = 1

    

    # compute the number of groups

    num_clusters = num_nodes // patch_size

    if num_clusters < 2:

        num_clusters = 2  # at least 2 groups

    

    # graph partitioning via spectral clustering

    clustering = SpectralClustering(n_clusters=num_clusters, 

                                   affinity='precomputed',

                                   assign_labels='discretize',

                                   random_state=0)

    labels = clustering.fit_predict(adj_matrix)

    

    # group nodes by the partition result

    node_split = []

    for i in range(max(labels)+1):

        nodes = np.where(labels == i)[0]

        if len(nodes) > 0:

            node_split.append(nodes)

    

    # print the size of each subgraph

    lengths = [len(i) for i in node_split]

    # print(f"Subgraph size distribution: {set(lengths)}")

    

    return node_split



def get_subgraphs(Data_path, Data_name, patch_size,num_nodes):

    full_path = os.path.join(Data_path, Data_name)

    # pass header=0 when reading the CSV to obtain the column names

    data = pd.read_csv(full_path +'/{}.csv'.format(Data_name), header=0)  # the first row holds the column names

    # convert columns to numeric types (important fix)

    data['from'] = data['from'].astype(int)

    data['to'] = data['to'].astype(int)

    data['cost'] = data['cost'].astype(float)



    # rename columns for easier handling

    data.columns = ['from', 'to', 'cost']



    # get the node count (node ids are assumed to start at 0)

    num_nodes = max(data['from'].max(), data['to'].max()) + 1



    # initialize the adjacency matrix to zeros

    distance_matrix = np.zeros((num_nodes, num_nodes))



    # fill the adjacency matrix from distances; a larger distance gives a smaller weight

    # use the reciprocal as the weight: weight = 1 / distance

    for _, row in data.iterrows():

        from_node = int(row['from'])

        to_node = int(row['to'])

        distance = float(row['cost'])

        

        distance_matrix[from_node, to_node] = distance

        distance_matrix[to_node, from_node] = distance  # comment out this line for directed graphs



        # build the normalized adjacency matrix from the distance matrix. Entry (i, j) = distance(i, j) divided by the sum of distances from i to its other neighbors.

        adj_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):

            for j in range(num_nodes):

                if i != j and distance_matrix[i, j] > 0:

                    total_distance = np.sum(distance_matrix[i, :])

                    adj_matrix[i, j] = distance_matrix[i, j] / total_distance

                    

        # build a plain adjacency matrix from the adjacency list: 0 = no connection, 1 = connected

        adjacency = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):

            for j in range(num_nodes):

                if adj_matrix[i, j] > 0:

                    adjacency[i, j] = 1\

                        

    # convert to an adjacency list

    adjacency_list = [list(np.where(row == 1)[0]) for row in adjacency]  

    

    # perform graph partitioning

    subgraphs = graph_split(num_nodes=num_nodes, adj=adjacency_list, patch_size=patch_size) 

             

    return subgraphs





class TrafficDataset(Dataset):

    def __init__(self, data_path, data_name, mode='train'):

        self.data_path = data_path

        self.data_name = data_name

        self.mode = mode

        self.data = np.load(os.path.join(data_path, data_name, mode + '.npy'))

        print(os.path.join(data_path, data_name, mode + '.npy'))

        self.min = value_mins[data_name]

        self.max = value_maxs[data_name]

        # normalize to [0, 1]

        self.data = (self.data - self.min) / (self.max - self.min)

        self.len = self.data.shape[0]



    def __getitem__(self, index):

        item = torch.tensor(self.data[index])

        item = item.float()

        return item



    def __len__(self):

        return self.len





def get_dataloader(data_path, data_name, batch_size, mode):

    dataset = TrafficDataset(data_path, data_name, mode)

    if mode == 'test' or mode == 'val':

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    else:

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader





def get_data_loader_all(data_path, data_name, batch_size, mode):  # load all data

    return get_dataloader(data_path, data_name, batch_size, mode)



def MAE(y_pred, y_true):

    mae = torch.mean(torch.abs(y_pred - y_true))

    # mae = mae * T * C * H * W

    return mae





def RMSE(y_pred, y_true):

    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))

    # rmse = rmse * T * C * H * W

    return rmse

        

def reverse_norm(value, max_value, min_value, is_0_1=True):  # de-normalization

    if is_0_1:  # de-normalization from [0, 1]

        value = value * (max_value - min_value) + min_value

    else:  # de-normalization from [-1, 1]

        value = (value + 1) / 2 * (max_value - min_value) + min_value



    return value

def get_adjacency_matrix(path,data_name):

    return np.load(path+'/'+data_name+'/adj_matrix.npy')

# split input and output for each batch

def split_batch(batch, his_len, pred_len,data_name):

    # standard format is (B, T, C, H, W)

    if data_name  in ['TrafficSH', 'TrafficNJ','TrafficCD', 

                      'TrafficZZ', 'TrafficTJ', 'TrafficHZ',

                      'TDrive', 'TaxiNYC2', 'BikeNYC2','Crowd']:

        batch = batch.unsqueeze(2)

        # standard format is (B, T, C, H, W)

        x = batch[:, :his_len, :,:, :]

        y = batch[:, his_len:pred_len + his_len, :,:, :]

    elif data_name in ['TaxiBJ', 'TaxiCQ', 'BikeNYC', 'TaxiBJ_CS']:

        # standard format is (B, T, C, H, W)

        x = batch[:, :his_len, :,:, :]

        y = batch[:, his_len:pred_len + his_len, :,:, :]

    elif data_name in ['PEMS03','PEMS04', 'PEMS07', 'PEMS08']:

        # standard format is (B, T, N, D)

        x = batch[:, :his_len, :, :]

        y = batch[:, his_len:pred_len + his_len, :, :]

    return x, y



value_maxs = {'TrafficSH': 21.83, 'TaxiBJ': 1285.0, 'TaxiBJ_CS': 1285.0, 'TaxiCQ': 2000.0,

              'TrafficCD': 22.25, 'TrafficNJ': 24.82, 'TrafficZZ': 23.26,

              'TrafficTJ': 25.24, 'TrafficHZ': 29.50, 'TDrive': 2681,

              'BikeNYC': 267, 'BikeNYC2': 299, 'TaxiNYC2': 1283,

              'PEMS03':1178.0,'PEMS04':919.0,'PEMS07':1495.0,'PEMS08':1077.0}

value_mins = {'TrafficSH': 0.0, 'TaxiBJ': 0.0, 'TaxiBJ_CS': 0.0, 'TaxiCQ': 0.0,

              'TrafficCD': 0.0, 'TrafficNJ': 0.0, 'TrafficZZ': 0.0,

              'TrafficTJ': 0.0, 'TrafficHZ': 0.0, 'TDrive': 0.0,

              'BikeNYC': 0.0, 'BikeNYC2': 0.0, 'TaxiNYC2': 0.0,

              'PEMS03':0.0,'PEMS04':0.0,'PEMS07':0.0,'PEMS08':0.0}




