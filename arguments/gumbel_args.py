from typing import NamedTuple


class GumbelArgs(NamedTuple):
    learn_temp: bool
    tau0: float
    temp: float


