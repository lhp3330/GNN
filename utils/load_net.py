import torch
from torch import nn
from model.layers import WeightedGNNConv, WeightedGCNConv


def get_env_net(model, in_dim, env_dim, out_dim, num_layers):
    layers = [nn.Linear(in_dim, env_dim)]  # encoder
    if model == 'GCN':
        for i in range(num_layers):
            layers.append(WeightedGCNConv(env_dim, env_dim, bias=True))
    if model == 'MEAN':
        for i in range(num_layers):
            layers.append(WeightedGNNConv(env_dim, env_dim, bias=True, aggr='mean'))
    if model == 'ADD':
        for i in range(num_layers):
            layers.append(WeightedGNNConv(env_dim, env_dim, bias=True, aggr='add'))
    layers.append(nn.Linear(env_dim, out_dim))  # decoder
    return nn.ModuleList(layers)


def get_act_net(model, env_dim, act_dim, out_dim, num_layers):
    layers = []
    pre_dim = env_dim
    if model == 'GCN':
        for i in range(num_layers - 1):
            layers.append(WeightedGCNConv(pre_dim, act_dim, bias=True))
            pre_dim = act_dim
        layers.append(WeightedGCNConv(act_dim, out_dim, bias=True))
    if model == 'MEAN':
        for i in range(num_layers - 1):
            layers.append(WeightedGNNConv(pre_dim, act_dim, bias=True, aggr='mean'))
            pre_dim = act_dim
        layers.append(WeightedGNNConv(act_dim, out_dim, bias=True))
    if model == 'ADD':
        for i in range(num_layers - 1):
            layers.append(WeightedGNNConv(pre_dim, act_dim, bias=True, aggr='add'))
            pre_dim = act_dim
        layers.append(WeightedGNNConv(act_dim, out_dim, bias=True, aggr='add'))
    return nn.ModuleList(layers)


# def get_CoGNN_Appnp(gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActArgs):
#     layers = [nn.Linear(env_args.in_dim, env_args.hidden_dim)]
#     for _ in range(2):
#         layers.append(CoGNNConv(gumbel_args, env_args, action_args))
#         layers.append(APPNPConv())
#     layers.append(nn.Linear(env_args.hidden_dim, env_args.out_dim))
#     return nn.ModuleList(layers)
