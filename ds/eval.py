from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from pathlib import Path

import torch
import random
import os
import speechbrain as sb
import numpy as np
EPS = np.finfo(float).eps

def dataio_prepare(hparams, method='D4AM', dataset='chime', model='TRANS'):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]
    
    # test is separate
    test_datasets = {}
    root = f'ds/manifests/{dataset}'
    
    for csv_file in os.listdir(root):
        if method in csv_file:
            csv_file = f'{root}/{csv_file}'
            name = Path(csv_file).stem
            test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=csv_file, replacements={"data_root": data_folder}
            )
            test_datasets[name] = test_datasets[name].filtered_sorted(
                sort_key="duration"
            )

    datasets = [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    if model != 'W2V2':
        # We get the tokenizer as we need it to encode the labels when creating mini-batches.
        tokenizer = hparams["tokenizer"]

        @sb.utils.data_pipeline.takes("wrd")
        @sb.utils.data_pipeline.provides(
            "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
        )
        def text_pipeline(wrd):
            wrd = wrd.replace('.', '')
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

    else:
        tokenizer = sb.dataio.encoder.CTCTextEncoder()

        @sb.utils.data_pipeline.takes("wrd")
        @sb.utils.data_pipeline.provides(
            "wrd", "char_list", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
        )
        def text_pipeline(wrd):
            wrd = wrd.replace('.', '')
            yield wrd
            char_list = list(wrd)
            yield char_list
            tokens_list = tokenizer.encode_sequence(char_list)
            yield tokens_list
            tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
            yield tokens_bos
            tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
            yield tokens_eos
            tokens = torch.LongTensor(tokens_list)
            yield tokens

        sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

        lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
        special_labels = {
            "bos_label": hparams["bos_index"],
            "eos_label": hparams["eos_index"],
            "blank_label": hparams["blank_index"],
        }
        train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
        )
        tokenizer.load_or_create(
            path=lab_enc_file,
            from_didatasets=[train_data],
            output_key="char_list",
            special_labels=special_labels,
            sequence_input=True,
        )
        
    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    return test_datasets, tokenizer

class ASREvaluator:
    def __init__(self, args):
        assert args.model is not None
        random.seed(args.seed)
        conf = f'ds/hparams/{args.model}.yaml'

        # Load hyperparameters file with command-line overrides
        with open(conf) as fin:
            hparams = load_hyperpyyaml(fin)

        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=conf,
        )

        # here we create the datasets objects as well as tokenization and encoding
        self.test_datasets, self.tokenizer = dataio_prepare(hparams, args.method, args.test_set, args.model)

        # Trainer initialization
        asr_brain = sb.core.Brain(
            modules=hparams["modules"],
            hparams=hparams,
            checkpointer=hparams["checkpointer"],
        )

        # adding objects to trainer:
        asr_brain.checkpointer.recover_if_possible(
            device=torch.device('cpu')
        )
        if "pretrainer" in hparams:
            hparams["pretrainer"].load_collected(device=torch.device('cuda'))

        self.modules = asr_brain.modules
        self.hparams = asr_brain.hparams
        for param in self.modules.parameters():
            param.requires_grad = False

        self.make_dataloader = asr_brain.make_dataloader

        self.parallel = args.parallel
        self.modules.eval()
        self.model = args.model

    @ torch.no_grad()
    def to_cuda(self):
        self.modules = self.modules.cuda()
        if self.parallel:
            self.modules.Transformer = torch.nn.DataParallel(
                self.modules.Transformer)

    @ torch.no_grad()
    def load_data(self, batch):
        wavs, lens = batch.sig
        tokens, tokens_lens = batch.tokens
        tokens_bos, _ = batch.tokens_bos
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        return wavs, lens, (tokens, tokens_lens, tokens_bos, tokens_eos, tokens_eos_lens)
    
    def W2V2_predict(self, wavs, wav_lens, tokens_bos):
        # Forward pass
        feats = self.modules.wav2vec2(wavs)
        x = self.modules.enc(feats)

        # Compute outputs
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)
        p_tokens = sb.decoders.ctc_greedy_decode(
            p_ctc, wav_lens, blank_id=self.hparams.blank_index
        )
        # Decode token terms to words
        predicted_words = [
            "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
            for utt_seq in p_tokens
        ]
        return predicted_words

    def RNN_predict(self, wavs, wav_lens, tokens_bos):
        self.modules.normalize.to(wavs.device)
        
        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_bos)  # y_in bos + tokens
        h, _ = self.modules.dec(e_in, x, wav_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        p_tokens = sb.decoders.ctc_greedy_decode(
        p_seq, wav_lens, blank_id=self.hparams.blank_index
        )

        # Decode token terms to words
        predicted_words = [
            self.tokenizer.decode_ids(utt_seq).split(" ")
            for utt_seq in p_tokens
        ]
        return predicted_words
    
    def TRAN_predict(self, wavs, wav_lens, tokens_bos):
        self.modules.normalize.to(wavs.device)
        
        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        src = self.hparams.CNN(feats)
        enc_out, pred = self.hparams.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for seq2seq log-probabilities
        pred = self.hparams.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        p_tokens = sb.decoders.ctc_greedy_decode(
            p_seq, wav_lens, blank_id=self.hparams.blank_index
        )
        # Decode token terms to words
        predicted_words = [
            self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in p_tokens
        ]
        return predicted_words

    def CONF_predict(self, wavs, wav_lens, tokens_bos):
        # forward modules
        feats = self.hparams.compute_features(wavs)
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        p_tokens = sb.decoders.ctc_greedy_decode(
            p_seq, wav_lens, blank_id=self.hparams.blank_index
        )
        # Decode token terms to words
        predicted_words = [
            self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in p_tokens
        ]
        return predicted_words

    @ torch.no_grad()
    def evaluate(self, dataloader):
        device = next(self.modules.parameters()).device
        self.wer_metric = self.hparams.error_rate_computer()
        
        for batch in tqdm(dataloader, desc="Evaluation"):
            wavs, wav_lens, labels = self.load_data(batch)
            wavs, wav_lens = wavs.to(device), wav_lens.to(device)

            tokens, tokens_lens, tokens_bos, tokens_eos, tokens_eos_lens = labels
            tokens, tokens_lens, tokens_bos, tokens_eos, tokens_eos_lens = tokens.to(device), tokens_lens.to(
                device), tokens_bos.to(device), tokens_eos.to(device), tokens_eos_lens.to(device)
            
            predicted_words = eval(f'self.{self.model}_predict')(wavs, wav_lens, tokens_bos)
            target_words = [wrd.split(" ") for wrd in batch.wrd]

            self.wer_metric.append(batch.id, predicted_words, target_words)

        torch.cuda.empty_cache()
        wer_avg = self.wer_metric.summarize("error_rate")
        return wer_avg

    @ torch.no_grad()
    def eval_chime(self):
        dataset_dict = {'dt05-real': [], 'dt05-simu': [], 'et05-real': [], 'et05-simu': []}
        # Testing
        for k in self.test_datasets.keys():  # keys are test_clean, test_other etc
            prefix = k.split('_')
            test_loader = self.make_dataloader(
                self.test_datasets[k], stage=sb.Stage.TEST, **self.hparams.test_dataloader_opts
            )
            dataset_dict[prefix[1] + '-' + prefix[-1]].append(self.evaluate(test_loader))
        print(' | '.join(['{:}: {:.2f}'.format(d, np.mean(dataset_dict[d])) for d in ['dt05-real', 'dt05-simu', 'et05-real', 'et05-simu']]))

    @ torch.no_grad()
    def eval_aurora(self):
        dataset_dict = {'dev-wv1': [], 'dev-wv2': [], 'test-wv1': [], 'test-wv2': []}
        # Testing
        for k in self.test_datasets.keys():  # keys are test_clean, test_other etc
            prefix = k.split('_')
            test_loader = self.make_dataloader(
                self.test_datasets[k], stage=sb.Stage.TEST, **self.hparams.test_dataloader_opts
            )
            dataset_dict[prefix[1] + '-' + prefix[-1]].append(self.evaluate(test_loader))
        print(' | '.join(['{:}: {:.2f}'.format(d, np.mean(dataset_dict[d])) for d in ['dev-wv1', 'dev-wv2', 'test-wv1', 'test-wv2']]))