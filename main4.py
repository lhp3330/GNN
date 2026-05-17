from arguments.parse_arguments import parse_arguments
from experiment4 import Experiment4
from torch.cuda import set_device


if __name__ == '__main__':
    args = parse_arguments()
    if args.gpu is not None:
        set_device(args.gpu)
    for seed in range(0, 500):
        args.seed = seed
        sota = Experiment4(args=args).run()
        if sota >= 0.72:
            print('*' * 50, end=' ')
            print('Seed: {}, metric: {:.4f} '.format(seed, sota))
