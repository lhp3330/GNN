from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", dest="dataset_name", default='roman-empire')

    # gumbel
    parser.add_argument("--learn_temp", dest="learn_temp", default=True, action='store_true', required=False)
    parser.add_argument("--temp_model", dest="temp_model", default='Linear')
    parser.add_argument("--tau0", dest="tau0", default=0.1, type=float, required=False)
    parser.add_argument("--temp", dest="temp", default=0.01, type=float, required=False)

    # optimization
    parser.add_argument("--max_epochs", dest="max_epochs", default=3000, type=int, required=False)
    parser.add_argument("--lr", dest="lr", default=3e-3, type=float, required=False)
    parser.add_argument("--weight_decay", dest="weight_decay", default=0, type=float, required=False)
    parser.add_argument("--dropout", dest="dropout", default=0.2, type=float, required=False)

    # env
    parser.add_argument("--env_model", dest="env_model", default="MEAN")
    parser.add_argument("--env_num_layers", dest="env_num_layers", default=5, type=int, required=False)
    parser.add_argument("--env_dim", dest="env_dim", default=128, type=int, required=False)
    parser.add_argument("--layer_norm", dest="layer_norm", default=True, action='store_true', required=False)
    parser.add_argument("--dec_num_layers", dest="dec_num_layers", default=1, type=int, required=False)

    # act
    parser.add_argument("--act_model", dest="act_model", default='MEAN')
    parser.add_argument("--act_num_layers", dest="act_num_layers", default=3, type=int, required=False)
    parser.add_argument("--act_dim", dest="act_dim", default=8, type=int, required=False)

    # reproduce
    parser.add_argument("--seed", dest="seed", type=int, default=42, required=False)
    parser.add_argument('--gpu', dest="gpu", type=int, default=0, required=False)

    # cluster loss
    parser.add_argument('--warmup_epochs', dest='warmup_epochs', type=int, default=50, required=False)
    parser.add_argument('--reg_weight', dest='reg_weight', type=float, default=0.005, required=False)
    parser.add_argument('--cluster_weight', dest='cluster_weight', type=float, default=0.001, required=False)
    #   cluster_weight = 0.1–0.5
    #   reg_weight = 0.01–0.1
    #   warmup_epochs = 10–20
    return parser.parse_args()
