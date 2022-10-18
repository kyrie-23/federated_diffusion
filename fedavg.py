import numpy as np
import torch
from torch.utils.data import Subset,DataLoader

# class FedAvg():
#
#
#     def __init__(self):
def aggregate(global_model, models,data_size):
    # Objective: aggregates all local model to the global model
    # Inputs: global model, a list of secondary UEs, experiment parameters
    # Outputs: parameter dictionary of aggregated model

    with torch.no_grad():
        global_model_dict = dict(global_model.state_dict())
        aggregated_dict = dict(global_model.state_dict())
        parties_dict = {}
        for i in range(len(models)):
            parties_dict[i] = dict(models[i].state_dict())
        # Set weights by data size
        data_size=np.array(data_size)/sum(data_size)
        for name, param in global_model_dict.items():
            aggregated_dict[name].data.copy_(sum([data_size[i] * parties_dict[i][name].data for i in range(len(models))]))

    return aggregated_dict


def dirichlet_loaders(dataset,  n_clients, batch_size,beta=0.1):
    # beta = 0.1, n_clients = 10

    label_distributions = []
    for y in range(len(dataset.classes)):  # dataset.classes 可能会报错，可以人为输入数据集的总类别数目
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))

    labels = np.array(dataset.targets).astype(np.int)
    client_idx_map = {i: {} for i in range(n_clients)}
    client_size_map = {i: {} for i in range(n_clients)}

    for y in range(len(dataset.classes)):  # dataset.classes 可能会报错，可以人为输入数据集的总类别数目
        label_y_idx = np.where(labels == y)[0]
        label_y_size = len(label_y_idx)

        sample_size = (label_distributions[y] * label_y_size).astype(np.int)
        sample_size[n_clients - 1] += len(label_y_idx) - np.sum(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i - 1] if i > 0 else 0):sample_interval[i]]

    client_dataloaders = []
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        subset = Subset(dataset, client_i_idx)
        loader=DataLoader(subset,batch_size=batch_size,shuffle=True)
        client_dataloaders.append(loader)

    return client_dataloaders
