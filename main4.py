from arguments.parse_arguments import parse_arguments
from experiment4 import Experiment4
from torch.cuda import set_device

tau0s = [0.01, 0.05, 0.1, 0.5, 1.0]
temps = [.01, 0.05, 0.1, 0.5]
reg_weights = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
lrs = [1e-3, 1e-5, 3e-3, 3e-5, 5e-3, 5e-5]
weight_decays = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
dropouts = [0.3, 0.4, 0.5, 0.6, 0.7]
env_dims = [64, 128, 256]
act_dims = [8, 16, 32]

ks = [5, 10, 15, 20]
alphas = [0.1, 0.15, 0.2, 0.3, 0.5]

if __name__ == '__main__':
    best_metric = 0
    args = parse_arguments()
    if args.gpu is not None:
        set_device(args.gpu)
    for lr in lrs:
        args.lr = lr
        for tau0 in tau0s:
            args.tau0 = tau0
            for temp in temps:
                args.temp = temp
                for reg_weight in reg_weights:
                    args.reg_weight = reg_weight
                    for weight_decay in weight_decays:
                        args.weight_decay = weight_decay
                        for env_dim in env_dims:
                            args.env_dim = env_dim
                            for act_dim in act_dims:
                                args.act_dim = act_dim
                                for alpha in alphas:
                                    args.alpha = alpha
                                    for dropout in dropouts:
                                        args.dropout = dropout
                                        for k in ks:
                                            args.k = k
                                            sota = Experiment4(args=args).run()
                                            if sota > best_metric:
                                                best_metric = sota
                                                print('*' * 30, end=' ')
                                                print(f'lr: {lr}, tau0: {tau0}, temp: {temp}, reg_weight: '
                                                      f'{reg_weight}, weight_decay: {weight_decay}, env_dim: '
                                                      f'{env_dim}, act_dim: {act_dim}, k: {k}, alpha: {alpha}, dropout: {dropout}, sota: {best_metric:.4f}')
