from arguments.parse_arguments import parse_arguments
from experiment4 import Experiment4
from torch.cuda import set_device
import itertools

if __name__ == '__main__':
    args = parse_arguments()
    if args.gpu is not None:
        set_device(args.gpu)
    sota = Experiment4(args=args).run()



def grid_search():
    args = parse_arguments()
    if args.gpu is not None:
        set_device(args.gpu)

    search_space = {
        'lr': [1e-3, 1e-5, 3e-3, 3e-5, 5e-3, 5e-5],
        'weight_decay': [1e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'tau0': [0.01, 0.05, 0.1],
        'reg_weight': [0.0, 0.001, 0.005, 0.01, 0.05],
        'alpha': [0.1, 0.15, 0.2],
        # 'env_dim': [64, 128, 256],
        # 'act_dim': [8, 16, 32]
    }

    keys = list(search_space.keys())
    values = list(search_space.values())

    best_metric = 0.0
    best_config = None
    total = len(list(itertools.product(*values)))
    print(f"combination sum: {total}")

    for i, combo in enumerate(itertools.product(*values)):
        for key, val in zip(keys, combo):
            args.iter = i
            setattr(args, key, val)

        sota = Experiment4(args=args).run()

        if sota > best_metric:
            best_metric = sota
            best_config = {k: getattr(args, k) for k in keys}
            print('*' * 30, end=' ')
            print(f"* sota: {best_metric:.4f} | Config: {best_config}")

    print("=" * 50)
    print(f"best metric: {best_metric:.4f}")
    print(f"best hyper p: {best_config}")

# if __name__ == '__main__':
#     grid_search()
