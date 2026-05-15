from arguments.parse_arguments import parse_arguments
from experiment2 import Experiment2
from torch.cuda import set_device


if __name__ == '__main__':
    args = parse_arguments()
    if args.gpu is not None:
        set_device(args.gpu)
    Experiment2(args=args).run()
