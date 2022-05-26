import tokenize
from model import load_model
import numpy as np
import argparse
import yaml
import torch
import random


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def argument_parsing():
    print('[Parsing] - Start Argument Parsing')

    parser = argparse.ArgumentParser(description='Argument Parser.')
    parser.add_argument('--config', type=str,
                        help='Path to experiment configuration.')
    parser.add_argument('--method', type=str, choices=['NOIS', 'INIT', 'CLSO', 'SRPR', 'GCLB', 'D4AM', 'GRID'],
                        default='D4AM')
    parser.add_argument('--model', type=str, choices=['CONF', 'W2V2', 'RNN', 'TRAN'],
                        default='TRAN')
    parser.add_argument('--task', type=str, default='train',
                        choices=['train', 'write', 'test'], help='Task to do.')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'],
                        help='optimizer type.')
    parser.add_argument('--logdir', default='log', type=str,
                        help='Directory for logging.', required=False)
    parser.add_argument('--loss', default='mrstft', type=str, choices=['sisdr', 'pmsqe', 'stoi', 'estoi', 'mse', 'l1', 'mrstft'],
                        help='The objective of denoising tasks.')
    parser.add_argument('--ckptdir', default='ckpt', type=str,
                        help='Path to store checkpoint result, if empty then default is used.')
    parser.add_argument(
        '--ckpt', type=str, help="Path to load target model")
    parser.add_argument('--init_ckpt', type=str, default='ckpt/INIT.pth',
                        help="Path to load source pretrain model")
    parser.add_argument('--test_set', type=str,
                        default='chime', choices=['chime', 'aurora'])
    parser.add_argument('--ds_train_conf', type=str, default='ds/hparams/ds_train.yaml',
                        help="The path to get asr configuration.")
    parser.add_argument('--out', type=str,
                        help="Path to output testing results")

    # Options
    parser.add_argument('--seed', default=1337, type=int,
                        help='Random seed for reproducable results.', required=False)
    parser.add_argument('--label_percentage', type=float,
                        help='Choosing toppest K similar noise for noise adaptation.')
    parser.add_argument('--n_jobs', default=4, type=int,
                        help='The number of process for loading data.')
    parser.add_argument('--bsz', type=int)
    parser.add_argument('--parallel', action='store_true',
                        help='Taking multiple devices for training.')
    parser.add_argument('--eval_init', action='store_true',
                        help='Computing initial scores before noise adaptaion.')
    parser.add_argument('--empty_cache', action='store_true',
                        help='Cleaning up the memory of GPU cache in each step.')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--continue_log', action='store_true')
    parser.add_argument('--test_root', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    args.config = f'config/config.yaml'
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if args.bsz is not None:
        args.config['data']['batch_size'] = args.bsz
    if args.method == 'GRID':
        args.method = f'{args.method}_alpha{args.alpha}'
        assert args.alpha is not None
    return args


def main():
    # parsing arguments
    args = argument_parsing()

    # set random seed
    set_random_seed(args.seed)

    if args.task == 'train':
        from data import get_trainloader, Corruptor
        from ds.agent import DownstreamAgent

        # construct corruptor
        corruptor = Corruptor(
            **args.config['data']['corruptor'], seed=args.seed)

        # build downtream agent to get classification objective
        ds_agent = DownstreamAgent(args, corruptor)

        # set dataloader
        print("[DataLoder] - Preparing dataset for the denoising task")
        se_loader = get_trainloader(args, corruptor)

        # set model and optimizer
        init_step = 0
        model, optimizer, init_step = load_model(args)

        # set loss function for denoising
        from loss import get_loss_func
        loss_func = get_loss_func(args)

        # set weight scheduler
        weighter = None
        if args.method != 'CLSO' and not 'GRID' in args.method:
            if args.alpha is not None:
                args.config['weighter']['alpha'] = args.alpha
            from weighter import Weighter
            weighter = Weighter(**args.config['weighter'], method=args.method)

        # process training
        from train import train
        train(args, model, optimizer, se_loader,
              ds_agent, loss_func, weighter, init_step)

    elif args.task == 'write':
        from data import get_simpleloader
        from utils import write_wav, write_csv

        if args.method != 'NOIS':
            # set dataloader
            print("[DataLoder] - Preparing dataset to write enhance result")
            simple_loader = get_simpleloader(args)

            # set model and optimizer
            assert args.ckpt is not None
            model, _, _ = load_model(args)
            write_wav(args, model, simple_loader)
        write_csv(args)

    elif args.task == 'test':
        from ds.eval import ASREvaluator
        evaluator = ASREvaluator(args)
        if args.test_set == 'chime':
            evaluator.eval_chime()
        elif args.test_set == 'aurora':
            evaluator.eval_aurora()


if __name__ == '__main__':
    main()
