from torch.utils.tensorboard import SummaryWriter
from model import save_model
from tqdm import tqdm

import numpy as np
import torch
import math
import time


class Recorder:
    def __init__(self, args, init_step, **kwargs):
        self.train_config = args.config['train']
        assert self.train_config['eval_step'] >= self.train_config['log_step']
        assert args.config['data']['acml_batch_size'] % args.config['data']['batch_size'] == 0

        self.step_count = init_step
        self.batch_count = 0
        self.batch_factor = args.config['data']['acml_batch_size'] // \
            args.config['data']['batch_size']
        self.max_grad = self.train_config['gradient_clipping']
        self.logger = load_logger(args)
        self.scores = ''
        self.pbar = tqdm(
            initial=init_step, total=self.train_config['total_steps'], dynamic_ncols=True)
        self.loss_record = []
        self.grad_record = []
        self.alpha_record = []

    def accumulate(self, grad_norm, loss, alpha=None):
        self.loss_record.append(loss)
        self.grad_record.append(grad_norm)
        if alpha is not None:
            self.alpha_record.append(alpha)

    def log(self):
        self.loss_avg = np.mean(self.loss_record)
        self.grad_avg = np.mean(self.grad_record)
        self.logger.add_scalar('train_loss', self.loss_avg, self.step_count)
        self.logger.add_scalar('grad_norm', self.grad_avg, self.step_count)
        if len(self.alpha_record) != 0:
            self.alpha_avg = np.mean(self.alpha_record)
            self.logger.add_scalar('alpha', self.alpha_avg, self.step_count)
            self.pbar.set_description(
                'train_loss {:.5f}{:} | alpha {:.4f}'.format(self.loss_avg, self.scores, self.alpha_avg))
        else:
            self.pbar.set_description(
                'train_loss {:.5f}{:}'.format(self.loss_avg, self.scores))

        self.loss_record = []
        self.grad_record = []
        self.alpha_record = []

    def eval(self, metrics):
        self.scores = ''.join(
            [' | dev_{:} {:.5f}'.format(m, s) for (m, s) in metrics])
        if hasattr(self, 'alpha_avg'):
            self.pbar.set_description(
                'train_loss {:.5f}{:} | alpha {:.4f}'.format(self.loss_avg, self.scores, self.alpha_avg))
        else:
            self.pbar.set_description(
                'train_loss {:.5f}{:}'.format(self.loss_avg, self.scores))
        for (m, s) in metrics:
            self.logger.add_scalar(f'dev_{m}', s, self.step_count)

    def update(self):
        self.batch_count = 0
        self.step_count += 1
        self.pbar.update(1)

    def close(self):
        self.logger.close()
        self.pbar.close()

    def is_update(self):
        return self.batch_count == self.batch_factor

    def is_log(self):
        return self.step_count % self.train_config['log_step'] == 0

    def is_eval(self):
        return self.step_count % self.train_config['eval_step'] == 0

    def is_stop(self):
        return self.step_count >= self.train_config['total_steps']


def load_logger(args):
    import os
    import shutil

    def process_filepath(path):
        if not args.method in path:
            path += '/{:}'.format(args.method)          
        return path

    # build logger directory
    args.logdir = process_filepath(args.logdir)
    if not args.continue_log:
        if os.path.isdir(args.logdir):
            shutil.rmtree(args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    logger = SummaryWriter(args.logdir)

    # build ckpt directory
    os.makedirs(args.ckptdir, exist_ok=True)
    return logger


def set_GPU_device(optim):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.cuda()
            if param._grad is not None:
                param._grad.data = param._grad.data.cuda()
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.cuda()
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.cuda()
    return optim

def inject_grad_noise(model, noise_var, lr, momentum):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.data = param.grad.data + \
                torch.randn_like(param.grad.data) * noise_var**0.5 * \
                (1 - momentum)**0.5 / lr * (param.grad.data**2)**0.25

def train(args, model, optimizer, se_loader, ds_agent, loss_func, weighter=None, init_step=0):
    # set iterator of denoising dataloader
    se_iterator = iter(se_loader)

    # set module devices to cuda
    loss_func = loss_func.cuda()
    ds_agent.to_cuda()
    model = model.cuda()
    if args.parallel:
        model.ae_model = torch.nn.DataParallel(model.ae_model)
    optimizer = set_GPU_device(optimizer)

    if args.eval_init:
        metrics = ds_agent.valid(model)
        scores = ''.join([' | dev_{:} {:.5f}'.format(m, s)
                          for (m, s) in metrics[1:]])
        print('[Initial] dev_loss {:.5f}{:}'.format(metrics[0][1], scores))

    # build recorder
    recorder = Recorder(args, init_step)
    print('[Training] - Start training {:} model'.format(args.method))

    loss_record = 0
    import time
    while not recorder.is_stop():
        try:
            # load downstream data
            noisy_ds_batch, clean_ds_batch, ds_lengths, labels = ds_agent.get_batch()

            # process forward and backward of the downstream task
            pred_ds_batch = model(noisy_ds_batch, ds_lengths)
            loss = ds_agent.get_main_loss(
                pred_ds_batch, ds_lengths, labels)
            loss = loss / recorder.batch_factor
            loss_record += loss.item()

            if args.method != 'CLSO':
                # load denoising data
                try:
                    noisy_se_batch, clean_se_batch, se_lengths = next(
                        se_iterator)
                except StopIteration:
                    se_iterator = iter(se_loader)
                    noisy_se_batch, clean_se_batch, se_lengths = next(
                        se_iterator)

                # process forward and backward of the denoising task
                pred_se_batch = model(noisy_se_batch, se_lengths)
                aux_loss = 0.5 * loss_func(pred_se_batch, clean_se_batch) + \
                    0.5 * loss_func(pred_ds_batch, clean_ds_batch)
                aux_loss = aux_loss / recorder.batch_factor
                if 'GRID' in args.method:
                    loss = loss + args.alpha * aux_loss
                else: 
                    weighter.accumulate(aux_loss, model)

            loss.backward()
            recorder.batch_count += 1

            if recorder.is_update():
                
                # update the parameters of weighter
                if weighter is not None:
                    weighter.update(model)
                    if args.method in ['D4AM', 'SRPR']:
                        inject_grad_noise(model, 2 * optimizer.param_groups[0]['lr'] * (1 - 0.9) / len(se_loader.dataset),
                                        optimizer.param_groups[0]['lr'], 0.9)

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()), recorder.max_grad)
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    print(
                        f'[TRAINING] - Error : grad norm is {grad_norm.item()} at step {recorder.step_count}')
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                
                alpha = weighter.aux_alpha if weighter is not None else None
                recorder.accumulate(grad_norm.item(), loss_record, alpha)
                loss_record = 0

                # update parameters
                optimizer.step()
                model.zero_grad()
                if args.empty_cache:
                    torch.cuda.empty_cache()

                # update loss and gradient norm recording
                recorder.update()

                # logging loss and gradient norm recording
                if recorder.is_log():
                    recorder.log()

                # evaluate performance on devlopment set and save model
                if recorder.is_eval():
                    print('[Training] - Evaluating on development set')
                    torch.cuda.empty_cache()
                    results = ds_agent.valid(model)
                    recorder.eval(results)
                    save_model(model, optimizer, args, recorder.step_count)

                if recorder.is_stop():
                    break

        except RuntimeError as e:
            print(e)
            if not 'CUDA out of memory' in str(e):
                raise
            print('[Training] - CUDA out of memory at step: ',
                  recorder.step_count)
            optimizer.zero_grad()
            torch.cuda.empty_cache()
    recorder.close()
