from typing import NamedTuple
from utils.load_net import get_act_net


class ActArgs(NamedTuple):
    model_type: str
    num_layers: int
    env_dim: int

    dropout: float
    act_type: str

    out_dim: int
    hidden_dim: int


    def get_net(self):
        return get_act_net(model=self.model_type, env_dim=self.env_dim, act_dim=self.hidden_dim,
                           out_dim=self.out_dim, num_layers=self.num_layers)
