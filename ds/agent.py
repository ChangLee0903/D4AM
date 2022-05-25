from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
import torch
import random
import speechbrain as sb
import numpy as np
EPS = np.finfo(float).eps

def dataio_prepare(hparams, corruptor):
    def normalize(audio, target_level=-25):
        rms = (audio ** 2).mean(dim=0, keepdims=True) ** 0.5
        scalar = 10 ** (target_level / 20) / (rms+EPS)
        audio = audio * scalar
        return audio


    def readfile(name, target_level=-25):
        if '.npy' in name:
            return torch.FloatTensor(np.load(name))

        elif '.wav' in name or '.flac' in name:
            '''Normalize the signal to the target level'''
            audio = sb.dataio.dataio.read_audio(name)
            audio = normalize(audio, target_level)
            return audio

    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")


    datasets = [train_data, valid_data]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("niy_sig", "cln_sig")
    def audio_pipeline(wav):
        if 'dev' in wav or 'test' in wav:
            niy_sig = cln_sig = readfile(wav)
        else:
            cln_sig = readfile(wav)
            niy_sig = corruptor.corrupt(cln_sig)
        yield niy_sig
        yield cln_sig
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "niy_sig", "cln_sig", "wrd",
                   "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, tokenizer


class DownstreamAgent:
    def __init__(self, args, corruptor):
        assert args.ds_train_conf is not None
        random.seed(args.seed)

        # Load hyperparameters file with command-line overrides
        with open(args.ds_train_conf) as fin:
            hparams = load_hyperpyyaml(fin)

        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=args.ds_train_conf,
        )

        # here we create the datasets objects as well as tokenization and encoding
        train_data, valid_data, tokenizer = dataio_prepare(
            hparams, corruptor)

        # Trainer initialization
        asr_brain = sb.core.Brain(
            modules=hparams["modules"],
            opt_class=hparams["Adam"],
            hparams=hparams,
            checkpointer=hparams["checkpointer"],
        )

        # adding objects to trainer:
        asr_brain.checkpointer.recover_if_possible(
            device=torch.device('cpu')
        )

        self.tokenizer = hparams["tokenizer"]
        hparams["pretrainer"].load_collected(device=torch.device('cuda'))

        self.modules = asr_brain.modules
        self.hparams = asr_brain.hparams
        for param in self.modules.parameters():
            param.requires_grad = False

        self.hparams.train_dataloader_opts['batch_size'] = args.config['data']['batch_size']
        self.hparams.valid_dataloader_opts['batch_size'] = args.config['data']['batch_size']
      
        if args.label_percentage is not None:
            assert args.label_percentage <= 1.0
            train_data.data_ids = random.sample(
                train_data.data_ids, int(args.label_percentage * len(train_data.data_ids)))

        self.train_loader = asr_brain.make_dataloader(
            train_data, stage=sb.Stage.TRAIN, **self.hparams.train_dataloader_opts
        )
        self.train_loader = asr_brain.make_dataloader(
            train_data, stage=sb.Stage.TRAIN, **hparams["train_dataloader_opts"])

        self.train_iterator = iter(self.train_loader)

        self.valid_loader = asr_brain.make_dataloader(
            valid_data, stage=sb.Stage.VALID, **hparams["train_dataloader_opts"]
        )
        self.parallel = args.parallel
        self.modules.eval()

    @ torch.no_grad()
    def to_cuda(self):
        self.modules = self.modules.cuda()
        if self.parallel:
            self.modules.Transformer = torch.nn.DataParallel(
                self.modules.Transformer)

    @ torch.no_grad()
    def clear_grad(self):
        self.modules.zero_grad()

    @ torch.no_grad()
    def load_data(self, batch):
        niy_wavs, lens = batch.niy_sig
        cln_wavs, lens = batch.cln_sig
        tokens, tokens_lens = batch.tokens
        tokens_bos, _ = batch.tokens_bos
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        return niy_wavs, cln_wavs, lens, (tokens, tokens_lens, tokens_bos, tokens_eos, tokens_eos_lens)
    
    @ torch.no_grad()
    def get_batch(self):
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            batch = next(self.train_iterator)

        niy_wavs, cln_wavs, lens, tokens = self.load_data(batch)
        return niy_wavs, cln_wavs, lens, tokens

    def predict(self, wavs, wav_lens, tokens_bos):
        # compute features
        feats = self.hparams.compute_features(wavs)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)
        return p_ctc, p_seq, enc_out

    def compute_loss(self, p_ctc, p_seq, wav_lens, tokens, tokens_lens, tokens_eos, tokens_eos_lens):
        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )
        return loss

    def get_main_loss(self, wavs, wav_lens, labels):
        wavs, wav_lens = wavs.cuda(), wav_lens.cuda()
        tokens, tokens_lens, tokens_bos, tokens_eos, tokens_eos_lens = labels
        tokens, tokens_lens, tokens_bos, tokens_eos, tokens_eos_lens = tokens.cuda(
        ), tokens_lens.cuda(), tokens_bos.cuda(), tokens_eos.cuda(), tokens_eos_lens.cuda()
        p_ctc, p_seq, enc_out = self.predict(wavs, wav_lens, tokens_bos)
        loss = self.compute_loss(
            p_ctc, p_seq, wav_lens, tokens, tokens_lens, tokens_eos, tokens_eos_lens)
        return loss

    def decode(self, p_ctc, enc_out, wav_lens, is_greedy=True):
        # Compute outputs
        if is_greedy:
            hyps = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        else:
            hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)

        # Decode token terms to words
        predicted_words = [
            self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
        ]
        return predicted_words

    @ torch.no_grad()
    def evaluate(self, dataloader, model=None):
        device = next(model.parameters()).device if model is not None else next(
            self.modules.parameters()).device
        self.cer_metric = self.hparams.cer_computer()
        self.wer_metric = self.hparams.error_rate_computer()
        loss_record = []

        for batch in tqdm(dataloader, desc="Evaluation"):
            wavs, _, wav_lens, labels = self.load_data(batch)
            wavs, wav_lens = wavs.to(device), wav_lens.to(device)

            tokens, tokens_lens, tokens_bos, tokens_eos, tokens_eos_lens = labels
            tokens, tokens_lens, tokens_bos, tokens_eos, tokens_eos_lens = tokens.to(device), tokens_lens.to(
                device), tokens_bos.to(device), tokens_eos.to(device), tokens_eos_lens.to(device)
            if model is not None:
                wavs = model(wavs, wav_lens)

            p_ctc, p_seq, enc_out = self.predict(wavs, wav_lens, tokens_bos)
            loss = self.compute_loss(
                p_ctc, p_seq, wav_lens, tokens, tokens_lens, tokens_eos, tokens_eos_lens)

            loss_record.append(loss.item())

            predicted_words = self.decode(p_ctc, enc_out, wav_lens)
            target_words = [wrd.split(" ") for wrd in batch.wrd]

            self.wer_metric.append(batch.id, predicted_words, target_words)
            self.cer_metric.append(batch.id, predicted_words, target_words)

        torch.cuda.empty_cache()
        loss_avg = sum(loss_record) / len(loss_record)
        wer_avg = self.wer_metric.summarize("error_rate")
        cer_avg = self.cer_metric.summarize("error_rate")
        return [('loss', loss_avg), ('wer', wer_avg), ('cer', cer_avg)]

    def valid(self, model=None):
        results = self.evaluate(self.valid_loader, model)
        return results