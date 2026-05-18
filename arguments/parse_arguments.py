
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", dest="dataset_name", default='CiteSeer')

    # gumbel
    parser.add_argument("--learn_temp", dest="learn_temp", default=True)
    parser.add_argument("--tau0", dest="tau0", default=0.1, type=float)
    parser.add_argument("--temp", dest="temp", default=0.05, type=float)

    # optimization
    parser.add_argument("--max_epochs", dest="max_epochs", default=500, type=int)
    parser.add_argument("--lr", dest="lr", default=5e-3, type=float)
    parser.add_argument("--weight_decay", dest="weight_decay", default=5e-4, type=float)
    parser.add_argument("--dropout", dest="dropout", default=0.5, type=float)

    # env
    parser.add_argument("--env_model", dest="env_model", default="MEAN")
    parser.add_argument("--env_num_layers", dest="env_num_layers", default=2, type=int)
    parser.add_argument("--env_dim", dest="env_dim", default=128, type=int)

    # act
    parser.add_argument("--act_model", dest="act_model", default='MEAN')
    parser.add_argument("--act_num_layers", dest="act_num_layers", default=2, type=int)
    parser.add_argument("--act_dim", dest="act_dim", default=16, type=int)

    # reproduce
    parser.add_argument("--seed", dest="seed", type=int, default=42)  # 48, 42, 23, 170, 686(72.5), 779(73.0)
    parser.add_argument('--gpu', dest="gpu", type=int, default=0)

    # cluster loss
    parser.add_argument('--reg_weight', dest='reg_weight', type=float, default=0.01)

    # ppnp
    parser.add_argument('--k', dest='k', type=int, default=10)
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.1)

    parser.add_argument('--iter', dest='iter', type=int, default=0)

    return parser.parse_args()
