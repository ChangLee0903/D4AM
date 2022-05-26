from asyncore import read
from librosa.util import find_files
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import os
import glob
import random
import speechbrain as sb
import torch.nn.functional as F

EPS = np.finfo(float).eps


def filestrs2list(filestrs, fileroot=None, **kwargs):
    path = filestrs
    if type(filestrs) is not list:
        filestrs = [filestrs]

    all_files = []
    for filestr in filestrs:
        if os.path.isdir(filestr):
            all_files += sorted(find_files(filestr))
        elif os.path.isfile(filestr):
            with open(filestr, 'r') as handle:
                all_files += sorted(
                    [f'{fileroot}/{line[:-1]}' for line in handle.readlines()])
        else:
            all_files += sorted(glob.glob(filestr))

    all_files = sorted(all_files)
    print(
        f'[Filestrs2List] - Parsing filestrs: {path}. Complete parsing: {len(all_files)} files found.')
    return all_files


def normalize(audio, target_level=-25):
    rms = (audio ** 2).mean(dim=0, keepdims=True) ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio


def readfile(name, target_level=-25, norm=True):
    if '.npy' in name:
        return torch.FloatTensor(np.load(name))

    elif '.wav' in name or '.flac' in name:
        '''Normalize the signal to the target level'''
        audio = sb.dataio.dataio.read_audio(name)
        if norm:
            audio = normalize(audio, target_level)
        return audio


def duplicate(sig, length):
    if length >= sig.size(-1):
        times = length // sig.size(-1) + 1
        sig = sig.repeat(times)
    return sig


def truncate(sig, seg_length):
    pos = random.randrange(max(1, len(sig) - seg_length))
    sig = sig[pos: pos + seg_length]
    return sig


def noise_scaling(speech, noise, snr, eps=1e-10):
    snr_exp = 10.0 ** (snr / 10.0)
    speech_power = speech.pow(2).sum(dim=-1, keepdim=True)
    noise_power = noise.pow(2).sum(dim=-1, keepdim=True)
    scalar = (speech_power / (snr_exp * noise_power + eps)).pow(0.5)
    scaled_noise = scalar * noise
    return scaled_noise


class Corruptor:
    def __init__(self, noise_path, snr_low, snr_high, duplicate, seed, **kwargs):
        self.noise_list = filestrs2list(noise_path)
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.isduplicate = duplicate
        random.seed(seed)

    @ torch.no_grad()
    def corrupt(self, speech):

        length = len(speech)
        noise = readfile(random.choice(self.noise_list))
        if noise.ndim > 1:
            noise = noise.mean(dim=-1)

        if self.isduplicate:
            noise = duplicate(noise, length)
        noise = truncate(noise, length)

        snr = random.uniform(self.snr_low, self.snr_high)
        scaled_noise = noise_scaling(speech, noise, snr)

        noisy = speech + scaled_noise
        return noisy


class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, speech_path, min_length, max_length, corrupt_agent, seed=5566, **kwargs):
        random.seed(seed)
        self.signal_list = []
        for p in speech_path:
            self.signal_list += filestrs2list(p)
        self.min_length = min_length
        self.max_length = max_length
        self.corruptor = corrupt_agent

    def __len__(self):
        return len(self.signal_list)

    def __getitem__(self, idx):
        length = random.randint(self.min_length, self.max_length)
        clean = readfile(self.signal_list[idx])
        clean = truncate(clean, length)
        noisy = self.corruptor.corrupt(clean)
        return noisy, clean

    def collate_fn(self, data):
        noisy = pad_sequence(
            [wav[0] for wav in data], batch_first=True).contiguous()
        clean = pad_sequence(
            [wav[1] for wav in data], batch_first=True).contiguous()
        lengths = torch.LongTensor([len(wav[0]) for wav in data])
        lengths = lengths / lengths.max().item()
        return noisy, clean, lengths


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, speech_path, **kwargs):
        self.signal_list = filestrs2list(speech_path)

    def __len__(self):
        return len(self.signal_list)

    def __getitem__(self, idx):
        path = self.signal_list[idx]
        sig = readfile(path, norm=True)
        return sig, path

    def collate_fn(self, data):
        wavs = pad_sequence(
            [d[0] for d in data], batch_first=True).contiguous()
        paths = [d[1] for d in data]
        lengths = torch.LongTensor([len(d[0]) for d in data])
        lengths = lengths / lengths.max().item()
        return wavs, lengths, paths


def get_simpleloader(args):
    dataset = SimpleDataset(args.test_root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.config['data']['batch_size'],
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=args.n_jobs)
    return dataloader


def get_trainloader(args, corruptor):
    dataset = DenoisingDataset(
        **args.config['data'], corrupt_agent=corruptor, seed=args.seed)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.config['data']['batch_size'],
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.n_jobs)

    return dataloader
