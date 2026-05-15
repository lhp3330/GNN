from arguments.parse_arguments import parse_arguments
from experiment4 import Experiment4
from torch.cuda import set_device


if __name__ == '__main__':
    args = parse_arguments()
    if args.gpu is not None:
        set_device(args.gpu)
    Experiment4(args=args).run()
