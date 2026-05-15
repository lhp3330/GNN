from torch import nn
from torch.nn import Module

from arguments.gumbel_args import GumbelArgs


class TempSoftPlus(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_dim: int):
        super(TempSoftPlus, self).__init__()

        self.net = nn.Linear(in_features=env_dim, out_features=1, bias=False)
        self.softplus = nn.Softplus(beta=1)
        self.tau0 = gumbel_args.tau0

    def forward(self, x):
        x = self.net(x)
        x = self.softplus(x) + self.tau0  # noise ?
        temp = x.pow_(-1)
        return temp.masked_fill_(temp == float('inf'), 0.)
