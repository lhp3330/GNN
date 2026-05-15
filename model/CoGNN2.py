import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Dropout, LayerNorm
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
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args.hidden_dim)
        self.temp = gumbel_args.temp

        self.num_layers = env_args.num_layers
        self.env_net = env_args.get_net()

        self.layer_norm = LayerNorm(env_args.hidden_dim)
        self.dropout = Dropout(p=env_args.dropout)
        self.drop_ratio = env_args.dropout
        self.act = env_activation_type(env_args.act_type)
        self.in_act_net = ActNet(action_args=action_args)
        self.out_act_net = ActNet(action_args=action_args)

    def forward(self, x, edge_index):
        # node encode
        x = self.env_net[0](x)
        x = self.dropout(x)
        x = self.act(x)

        for i in range(self.num_layers):
            x = self.layer_norm(x)
            # action
            in_logits = self.in_act_net(x=x, edge_index=edge_index)
            out_logits = self.out_act_net(x=x, edge_index=edge_index)

            temp = self.temp_model(x=x) if self.learn_temp else self.temp
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)

            edge_weight = self.create_edge_weight(edge_index=edge_index, keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
            # edge_weight = None
            out = self.env_net[1 + i](x=x, edge_index=edge_index, edge_weight=edge_weight)
            out = self.dropout(out)
            out = self.act(out)
            x = out

        x = self.layer_norm(x)
        x = self.env_net[-1](x)  # decoder

        return x, edge_weight

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob
