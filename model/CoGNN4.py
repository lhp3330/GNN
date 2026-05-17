import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module, Dropout, LayerNorm
from torch_geometric.nn import APPNP
from torch_geometric.typing import Adj

from arguments.act_args import ActArgs
from arguments.env_args import EnvArgs
from arguments.gumbel_args import GumbelArgs
from model.act_net import ActNet
from model.temp import TempSoftPlus
from utils.load_data import env_activation_type


class CoGNN(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActArgs):
        super(CoGNN, self).__init__()
        self.encoder = nn.Linear(env_args.in_dim, env_args.hidden_dim)
        self.dropout = nn.Dropout(env_args.dropout)
        self.act = env_activation_type(env_args.act_type)

        self.layer1 = CoGNNConv(gumbel_args, env_args, action_args)
        self.layer2 = APPNP(K=gumbel_args.k, alpha=gumbel_args.alpha)
        self.layer3 = CoGNNConv(gumbel_args, env_args, action_args)
        self.layer4 = APPNP(K=gumbel_args.k, alpha=gumbel_args.alpha)
        self.layer5 = CoGNNConv(gumbel_args, env_args, action_args)
        self.layer6 = APPNP(K=gumbel_args.k, alpha=gumbel_args.alpha)

        self.gates = nn.ModuleList([nn.Linear(env_args.hidden_dim, 1) for _ in range(2)])
        self.norms = nn.ModuleList([LayerNorm(env_args.hidden_dim) for _ in range(2)])

        self.decoder = nn.Linear(env_args.hidden_dim, env_args.out_dim)
        self.layer_norm = LayerNorm(env_args.hidden_dim)
        self.appnp = APPNP(K=gumbel_args.k, alpha=gumbel_args.alpha)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.act(x)

        x, edge_weight = self.layer1(x, edge_index)
        x_prob = self.layer2(x, edge_index)
        gate = torch.sigmoid(self.gates[0](x))
        x = self.norms[0](gate * x_prob + (1 - gate) * x)

        x, edge_weight = self.layer3(x, edge_index)
        x_prob = self.layer4(x, edge_index)
        gate = torch.sigmoid(self.gates[1](x))
        x = self.norms[1](gate * x_prob + (1 - gate) * x)

        # x, edge_weight = self.layer5(x, edge_index)
        # # x_prob = M_ppnp.mm(x)
        # x_prob = self.layer6(x, edge_index)
        # gate = torch.sigmoid(self.gates[2](x))
        # x = self.norms[2](gate * x_prob + (1 - gate) * x)

        x = self.layer_norm(x)
        embedding = x
        x = self.decoder(x)
        return x, edge_weight, embedding


class CoGNNConv(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActArgs):
        super().__init__()
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args.hidden_dim)
        self.temp = gumbel_args.temp

        self.num_layers = env_args.num_layers

        self.env_net = env_args.get_net()
        self.in_act_net = ActNet(action_args=action_args)
        self.out_act_net = ActNet(action_args=action_args)

        self.layer_norm = LayerNorm(env_args.hidden_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(env_args.hidden_dim) for _ in range(env_args.num_layers)])
        self.dropout = Dropout(p=env_args.dropout)
        self.act = env_activation_type(env_args.act_type)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.layer_norms[i](x)
            in_logits = self.in_act_net(x=x, edge_index=edge_index)
            out_logits = self.out_act_net(x=x, edge_index=edge_index)

            temp = self.temp_model(x=x) if self.learn_temp else self.temp
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)

            edge_weight = self.create_edge_weight(edge_index=edge_index, keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
            out = self.env_net[1 + i](x=x, edge_index=edge_index, edge_weight=edge_weight)
            out = self.dropout(out)
            out = self.act(out)
            x = x + out

        # x = self.layer_norm(x)
        return x, edge_weight

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor):
        u, v = edge_index
        return keep_in_prob[v] * keep_out_prob[u]
