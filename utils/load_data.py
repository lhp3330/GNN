import copy
import torch
import numpy as np
from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid, Amazon
from torchmetrics import Accuracy, AUROC
import torch_geometric.transforms as T
import torch.nn.functional as F


def get_folds():
    return list(range(10))


def load_dataset(dataset_name):
    dataset_type = 'homo'
    env_act_type = 'relu'
    path = 'data'
    if dataset_name in ['roman-empire']:
        dataset_type = 'hetero'
        env_act_type = 'gelu'
        dataset = HeterophilousGraphDataset(root=path, name=dataset_name, transform=T.ToUndirected())[0]
    elif dataset_name in ['cora', 'CiteSeer',  'pubmed']:
        Planetoid.url = "https://gitee.com/jiajiewu/planetoid/raw/master/data/"
        dataset = Planetoid(root=path, name=dataset_name, transform=T.NormalizeFeatures())
        num_classes = dataset.num_classes
        dataset = dataset[0]
        # dataset = generate_single_random_split(data=dataset, num_classes=num_classes) # 622
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root=path, name=dataset_name, transform=T.NormalizeFeatures())
        num_classes = dataset.num_classes
        dataset = dataset[0]
        dataset = generate_split_masks(dataset, num_classes)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return dataset, dataset_type, env_act_type


def generate_split_mask_data(dataset, dataset_name, fold, num_classes):
    if dataset_name in ['cora', 'pubmed', 'CiteSeer']:
        return split_dataset(dataset, dataset_name, fold)
    if dataset_name in ['Computers', 'Photo']:
        return generate_split_masks(dataset, num_classes)
    else:
        return split_dataset2(dataset, fold)

def split_dataset(dataset, dataset_name, fold):
    if dataset_name == 'CiteSeer':
        return dataset

    device = dataset.x.device
    with np.load(f'utils/folds/{dataset_name}_split_0.6_0.2_{fold}.npz') as folds_file:
        train_mask = torch.tensor(folds_file['train_mask'], dtype=torch.bool, device=device)
        val_mask = torch.tensor(folds_file['val_mask'], dtype=torch.bool, device=device)
        test_mask = torch.tensor(folds_file['test_mask'], dtype=torch.bool, device=device)

    setattr(dataset, 'train_mask', train_mask)
    setattr(dataset, 'val_mask', val_mask)
    setattr(dataset, 'test_mask', test_mask)

    return dataset

def split_dataset2(dataset, fold):
    dataset_copy = copy.deepcopy(dataset)
    dataset_copy.train_mask = dataset_copy.train_mask[:, fold]
    dataset_copy.val_mask = dataset_copy.val_mask[:, fold]
    dataset_copy.test_mask = dataset_copy.test_mask[:, fold]
    return dataset_copy

def get_metric_type(num_classes):
    return Accuracy(task='multiclass', num_classes=num_classes)


def get_optimizer(model, lr, weight_decay):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def env_activation_type(act_type):
    if act_type == 'gelu':
        return F.gelu
    else:
        return F.relu


def generate_single_random_split(data, num_classes, train_ratio=0.6, val_ratio=0.2):
    num_nodes = data.num_nodes
    y = data.y

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        class_idx = (y == c).nonzero(as_tuple=False).view(-1)

        shuffled_idx = class_idx[torch.randperm(class_idx.size(0))]

        num_class_nodes = shuffled_idx.size(0)
        train_end = int(num_class_nodes * train_ratio)
        val_end = train_end + int(num_class_nodes * val_ratio)

        train_mask[shuffled_idx[:train_end]] = True
        val_mask[shuffled_idx[train_end:val_end]] = True
        test_mask[shuffled_idx[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def generate_split_masks(data, num_classes):
    num_nodes = data.y.size(0)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))]

        train_mask[idx[:20]] = True
        val_mask[idx[20:50]] = True
        test_mask[idx[50:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data
