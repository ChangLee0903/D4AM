# D4AM: A General Denoising Framework for Downstream Acoustic Models
This is the official implementation of D4AM. 
We will set our repository to public if our paper gets accepted.
The demo page is locally provided in our supplementary materials, and our experiment needs several steps to reproduce the table results, which will be described as follows.

## Contents
- [Installation](#installation)
- [Steps and Usages](#steps-and-usages)

## Installation
Note that our environment is in ```Python 3.8.12```. To run the experiment of D4AM, you can copy the repository and install it by using the pip tool:
```bash
# Install all the necessary packages
pip install -r requirements.txt
```

## Steps and Usages
1. Specify pathes of speech and noise datasets:

Before conducting the training process, there are two parts to be set. 
The first part is the speech and noise path in <code>config/config.yaml</code>. 
Note that all the training noisy utterances are generated online. 
Our source noise dataset is provided by [DNS-Challenge](https://github.com/microsoft/DNS-Challenge) and the utterances come from the LibriSpeech corpus. 
Check <code>dataset</code> in <code>config/config.yaml</code>:
   <pre><code>data:
    ...
    speech_path: [../speech_data/LibriSpeech/train-clean-360, ../speech_data/LibriSpeech/train-other-500]
    corruptor:
      noise_path: ../noise_data/DNS_noise_pros
    ...
    </code></pre>
<code>speech_path</code> indicates which subsets that are only used as the training data of the regression objective, and <code>noise_path</code> is the noise dataset used to corrupt clean utterances.
The second part is the wav path in the manifest under <code>ds/manifests/libri</code>.
e.g. <code>../speech_data/LibriSpeech/train-clean-100</code> should be modified to your own root path of LibriSpeech.
Note that <code>dev-clean_wham.csv</code> and <code>test-clean_wham.csv</code> are our own mixed validation sets to observe learning curves, which follow the same format of original manifest.

2. Download the pre-training checkpoint (INIT) and downstream recognizers for testing:
We keep our initial model and the checkpoints of other fine-tuning results in the [link](https://drive.google.com/file/d/1vHJkzB0GSj7YqoD8-Qkqrej9bjQO_fRY/view?usp=sharing).
All the downstream recognizers described in Section 4.2 can be found [here](https://drive.google.com/file/d/1Alp2e5EZQdNUyDVs_xRFjj_wQfpiM5fK/view?usp=sharing).
Both of them shoul be put under the D4AM directory and execute <code>tar zxvf Filename.tar.gz</code>.
Check whether the <code>models</code> folder is put under <code>ds</code> (<code>ds/models</code>).

3. Train your own fine-tuning models locally:
Most checkpoints have been provided in the link mentioned in the previous step. 
You can derive our own model by taking the command: <code>python main.py --task train --method [assigned method]</code>.
e.g. <code>python main.py --task train --method D4AM</code>.
Note that you need to specify a alpha value as you want to choose GRID. e.g. <code>python main.py --task train --method GRID --alpha 0.1</code>

4. Writing enhanced output:
As you have prepared the checkpoints of all methods. 
Run <code>bash generate_enhanced_results.sh</code> to generate the corresponding enhanced results.
Note that you need to specify you own roots of CHIME-4 and Aurora-4 in the script.

5. Evaluation with various downstream recognizers:
While you settle down all the enhanced results and downstream recognizers, you can run the following command to testing the performance of enhancement methods:
<pre><code>python main.py --task test --method [NOIS/INIT/CLSO/SRPR/GCLB/D4AM] --model[CONF/TRAN/RNN/W2V2] --test_set [chime/aurora]</code></pre>