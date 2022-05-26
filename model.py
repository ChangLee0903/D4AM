from torch.autograd import Function
import torch.nn.functional as F
import torch


def save_model(model, optimizer, args, current_step):
    all_states = {
        'Model': model.state_dict(),
        'Optimizer': optimizer.state_dict(),
        'Current_step': current_step,
        'Args': args
    }
    torch.save(all_states, f'{args.ckptdir}/{args.model}.pth')


def load_model(args):
    if args.ckpt is not None:
        args.model = torch.load(args.ckpt, map_location='cpu')['Args'].model
    elif args.init_ckpt is not None:
        args.model = torch.load(args.init_ckpt, map_location='cpu')[
            'Args'].model

    print(f'[MODEL] - Building {args.model} model')
    model = WaveformDenoiseModel(args)
    optimizer = eval(f'torch.optim.{args.opt}')(model.parameters(),
                                                **args.config['train']['optimizer'][args.opt])

    init_step = 0
    if args.ckpt is not None:
        print(f'[MODEL] - Loading parameters from {args.ckpt}')
        ckpt = torch.load(args.ckpt, map_location='cpu')
        init_step = ckpt['Current_step']
        model.load_state_dict(
            {k.replace('module.', ''): ckpt['Model'][k] for k in ckpt['Model']})
        optimizer.load_state_dict(ckpt['Optimizer'])

    return model, optimizer, init_step


class WaveformDenoiseModel(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super(WaveformDenoiseModel, self).__init__()
        if args.model == 'DEMUCS':
            from denoiser.demucs import Demucs as DEMUCS
            self.ae_model = DEMUCS(**args.config['model']['DEMUCS'])

        assert args.init_ckpt is not None
        self.load_state_dict(torch.load(args.init_ckpt)['Model'])

    @ torch.no_grad()
    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        max_len = lengths.max().item()
        ascending = torch.arange(max_len).unsqueeze(
            0).expand(len(lengths), -1).to(lengths.device)
        length_masks = (ascending < lengths.unsqueeze(-1))
        return length_masks

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight' in name:
                if param.data.ndim == 1:
                    param.data = param.data.unsqueeze(-1)
                    torch.nn.init.xavier_uniform_(param.data)
                    param.data = param.data.squeeze(-1)
                else:
                    torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                torch.nn.init.constant_(param.data, 0)

    def forward(self, wav, lengths=None):
        if wav.ndim > 2:
            wav_len = wav.shape[1]
            predicted = []
            for i in range(wav.shape[-1]):
                wav_single = wav[:, :, i].cuda()
                predicted.append(self.ae_model(wav_single.unsqueeze(1)))
            predicted = torch.cat(predicted, dim=1).mean(dim=1, keepdim=True)
        else:
            wav_len = wav.shape[-1]
            wav = wav.cuda()
            predicted = self.ae_model(wav.unsqueeze(1))
        predicted = predicted.squeeze(1)
        if not lengths is None:
            lengths = lengths.cuda()
            lengths = (lengths.clone() * wav_len).long()
            length_masks = self._get_length_masks(lengths)
            predicted = predicted * length_masks
        return predicted
