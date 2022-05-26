# D4AM: A General Denoising Framework for Downstream Acoustic Models
<p align="center">
<img src="http://i.imgur.com/ysi9lu2.jpg" height="250"><img src="http://i.imgur.com/b8CFWsn.jpg" height="270">
</p>

This is the official implementation of D4AM. 
We will set our repository to the public if our paper gets accepted.
The demo page is locally provided in our supplementary materials, and our source code is provided in this anonymous Github repository.
All the model checkpoints can be found in our drive link as follows so that users can choose whether to train the fine-tuning models from scratch and reproduce the table results more precisely.
Next, we will describe how to run our experiments in the following description.

## Contents
- [Installation](#installation)
- [Steps and Usages](#steps-and-usages)

## Installation
Note that our environment is in ```Python 3.8.12```. To run the experiment of D4AM, you can copy our repository and install it by using the pip tool:
```bash
pip install -r requirements.txt
```

## Steps and Usages
### 1. Specify paths of speech and noise datasets:
Before starting the training process, there are two parts that we need to set the data paths. 
The first part is the speech and noise path in <code>config/config.yaml</code>. 
This part is responsible for preparing noisy-clean paired data only.
Note that all the noisy utterances for training are generated online. 
The noise dataset is provided by [DNS-Challenge](https://github.com/microsoft/DNS-Challenge), and the clean utterances come from the training subsets of LibriSpeech (Libri-360/Libri-500). 
Check <code>dataset</code> in <code>config/config.yaml</code>:
   <pre><code>data:
    ...
    speech_path: [../speech_data/LibriSpeech/train-clean-360, ../speech_data/LibriSpeech/train-other-500]
    corruptor:
      noise_path: ../noise_data/DNS_noise_pros
    ...
    </code></pre>
<code>speech_path</code> indicates which subsets that are only used as the clean speech for the regression objective, and <code>noise_path</code> is the noise dataset used to corrupt clean utterances.
In the second part, we need to set the wav path in the manifest under <code>ds/manifests/libri</code>.
This part is responsible for preparing both (noisy) speech-text and noisy-clean paired data for the classification and regression objectives, respectively. 
e.g. <code>../speech_data/LibriSpeech/train-clean-100</code> should be modified to your own root path of LibriSpeech.
Note that <code>dev-clean_wham.csv</code> and <code>test-clean_wham.csv</code> are our own mixed validation sets to observe learning curves, which follow the same format of original manifests.
Users can manually generate their own validation sets by preparing the corresponding manifests with the same format.
<br><br>

### 2. Download the checkpoints of SE models and downstream recognizers:
We keep our initial model and the checkpoints of other fine-tuning results in the [drive link]().
Users can decide to train the fine-tuning models individually or directly use our provided checkpoints for inference. 
All the downstream recognizers described in Section 4.2 can be found [here]().
Both of them should be put under the D4AM directory and execute <code>tar zxvf Filename.tar.gz</code>.
After the file extraction, make sure the pth files of SE models have been put under the <code>ckpt</code> folder (e.g. <code>ckpt/INIT.pth</code>) and the downstream recognizers have been put under the <code>ds</code> folder (e.g. <code>ds/models/conformer</code>).
<br><br>

### 3. Train your own fine-tuning models locally:
Most checkpoints have been provided in the link mentioned in the previous step.
This step can be skipped if the corresponding SE models have been prepared in <code>ckpt</code>.
To derive our own model, please execute this command: <code>python main.py --task train --method [assigned method]</code>.
e.g. <code>python main.py --task train --method D4AM</code>.
Note that you need to specify an alpha value as you want to choose GRID. e.g. <code>python main.py --task train --method GRID --alpha 0.1</code>
<br><br>

### 4. Writing enhanced results for evaluation:
As you have prepared the checkpoints of all methods, run <code>bash generate_enhanced_results.sh</code> to generate the corresponding enhanced results <code>results/</code> in and their manifests in <code>ds/manifests/chime</code> and <code>ds/manifests/aurora</code>.
Note that you need to first specify your own roots of CHIME-4 and Aurora-4 in the script (<code>chime_root</code> and <code>aurora_root</code>).
<br><br>

### 5. Evaluation with various downstream recognizers:
While you settle down all the enhanced results and downstream recognizers, you can run the following command to test the performance of enhancement methods:
<pre><code>python main.py --task test --method [NOIS/INIT/CLSO/SRPR/GCLB/D4AM] --model[CONF/TRAN/RNN/W2V2] --test_set [chime/aurora]</code></pre>
By specifying the methods, the downstream recognizers, and the testing corpus, users will get the table results like our paper.
