import torch
import os
import speechbrain as sb
import pandas as pd

from tqdm import tqdm
from data import normalize


@ torch.no_grad()
def write_wav(args, model, dataloader):
    model = model.cuda()
    if args.parallel:
        model.ae_model = torch.nn.DataParallel(model.ae_model)

    for (wavs, lens, paths) in tqdm(dataloader, desc='Computing enhancement prediction'):
        wavs, lens = wavs.cuda(), lens.cuda()
        pred_wavs = model(wavs, lens).cpu()
        wav_len = wavs.shape[1] if wavs.ndim > 2 else wavs.shape[-1]
        lens = (lens * wav_len).long()
        for pred_wav, l, path in zip(pred_wavs, lens, paths):
            file_name = path.split('/')[-1]
            dir_path = path.replace(file_name, '')
            dir_path = dir_path.replace(args.test_root, args.output_dir)
            os.makedirs(dir_path, exist_ok=True)
            sb.dataio.dataio.write_audio(
                f'{dir_path}/{file_name}', pred_wav[:l.item()], 16000)


def write_csv(args):
    for f in os.listdir(f'ds/manifests/{args.test_set}/template'):
        temp_pd = pd.read_csv(f'ds/manifests/{args.test_set}/template/{f}')
        temp_pd['wav'] = [args.output_dir + w for w in temp_pd['wav'].tolist()]
        temp_pd.to_csv(
            f'ds/manifests/{args.test_set}/{args.method}_{f}', index=False)
