from typing import NamedTuple

from utils.load_net import get_env_net


class EnvArgs(NamedTuple):
    model_type: str
    num_layers: int

    dropout: float
    act_type: str

    in_dim: int
    hidden_dim: int
    out_dim: int


    def get_net(self):
        return get_env_net(model=self.model_type, in_dim=self.in_dim, env_dim=self.hidden_dim,
                           out_dim=self.out_dim, num_layers=self.num_layers)
