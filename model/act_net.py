from torch import nn

from arguments.act_args import ActArgs
from utils.load_data import env_activation_type


class ActNet(nn.Module):
    def __init__(self, action_args: ActArgs):
        super(ActNet, self).__init__()
        self.num_layers = action_args.num_layers
        self.net = action_args.get_net()
        self.dropout = nn.Dropout(action_args.dropout)
        self.activation = env_activation_type(action_args.act_type)

    def forward(self, x, edge_index):
        for layer in self.net[:-1]:
            x = layer(x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = self.activation(x)
        x = self.net[-1](x=x, edge_index=edge_index)
        return x
