import argparse
import importlib
import sys


class ArgProcessor:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train a GRU network.')
        self.dict_loss_func = {}
        self.loss_func_name = None
        self.loss_func = None

    def get_args(self):
        # Initial Arguments
        self.parser.add_argument('--dataset_name', default=None, help='Dataset names')
        temp_args = ['--dataset_name', sys.argv[sys.argv.index('--dataset_name') + 1]]
        args = self.parser.parse_args(temp_args)

        # Load modules according to dataset_name
        try:
            print("Using Dataset: ", args.dataset_name)
            module_args = importlib.import_module('modules.' + args.dataset_name + '.arguments')
            add_args = module_args.add_args
        except:
            raise RuntimeError('Please select a supported dataset.')

        # Add Dataset-specific Arguments
        add_args(self.parser)

        # Parse Arguments
        args = self.parser.parse_args()

        # Special process of arguments
        if args.dataset_name == 'sensorsgas':
            args.eval_val = 0
            args.score_val = 0
            args.save_every_epoch = 1
            args.lr_schedule = 0

        print(args)
        return args
