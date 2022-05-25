from denoiser.stft_loss import MultiResolutionSTFTLoss
from asteroid.losses import SingleSrcNegSTOI
from asteroid.losses.sdr import SingleSrcNegSDR
import torch


class L1_MRSTFTLoss(torch.nn.Module):
    def __init__(self, factor_sc, factor_mag):
        super(L1_MRSTFTLoss, self).__init__()
        self.obj = torch.nn.L1Loss()
        self.mrstftloss = MultiResolutionSTFTLoss(
            factor_sc=factor_sc, factor_mag=factor_mag)

    def forward(self, predict, target):
        target = target.cuda()
        loss = self.obj(predict, target)
        sc_loss, mag_loss = self.mrstftloss(predict, target)
        loss = loss + sc_loss + mag_loss
        return loss


class DenoiseLossWrapper(torch.nn.Module):
    def __init__(self, loss_type: str, **kwargs):
        super(DenoiseLossWrapper, self).__init__()
        self.loss_type = loss_type
        if self.loss_type == 'sisdr':
            self.loss_func = SingleSrcNegSDR("sisdr")
        elif self.loss_type == 'stoi':
            self.loss_func = SingleSrcNegSTOI(sample_rate=16000)
        elif self.loss_type == 'estoi':
            self.loss_func = SingleSrcNegSTOI(sample_rate=16000, extended=True)

    def forward(self, est_targets, targets):
        loss = self.loss_func(est_targets, targets).mean()
        return loss


def get_loss_func(args):
    if args.loss == 'mrstft':
        loss = L1_MRSTFTLoss(**args.config['stftloss'])
    elif args.loss in ['sisdr', 'stoi', 'estoi']:
        loss = DenoiseLossWrapper(args.loss)
    else:
        loss = eval(f"torch.nn.{args.loss.upper()}Loss()")
    return loss
