<!--- Authors: Zbyněk Lička -->

### Installation
Normal:
```bash
# create python environment and activate it
get_dataset.sh
get_models.sh
get_vocoder.sh
pip3 install -r requirements.txt
```

### Training setup
Now you need to create a datalist. Choose 2 speakers and update the `speakers` in `prepare_train_list.ipynb`. You shouldn't need to change the arguments in `Configs/config.yml` but here are some important options you can change there:
```yml
device: "cuda" # change this to debug on CPU, don't actually train this model on CPU please...
batch_size: 2 # change this to utilize your GPU memory, batch of 5 takes about 10 GB

pretrained_model: "Models/epoch_00102.pth" # start at a checkpoint

train_data: "Data/train_list.txt" # prepare_train_list.ipynb places the datalists into these files
val_data: "Data/val_list.txt"     # but if you have your own datalists you may change this
```
Then you just run.
```bash
python3 train.py
```

### Inference
We prepared a small notebook for debugging inference. Just change the speakers in `speakers` of `inference.ipynb` notebook.

Inference of the whole testing set is done with `Data/test_list.txt`. Just place the path to the latest model into `Configs/config.yml` `pretrained_model:` option and run:
```bash
python3 inference.py
```


### Files
Added:
+ run.sh PBS script for training
+ get_dataset.sh script for dataset download

Altered:
+ prepare_train_list.ipynb creates tuples of paths, also moved to main directory
+ trainer.py removed mapping loss learning rate
+ train.py altered the batch as it now contains different data, altered the passing of arguments, removed mapping and style embedding loss calculation, replaced them by a single calculation
+ models.py removed style encoder, removed mapping network, replaced AdaINResBlk by ResBlkDecoder, altered the generator to not use style embedding, altered
+ readme.md Basically fully replaced by our own readme.md
+ losses.py most losses are now calculated for two real samples, added second step adversarial loss (including consitency regularization), added identity loss
+ meldataset.py altered the handling to produce two normalized mel-spectrograms, instead of a normalized mel-spectrogram, labels and reference mel-spectrograms
+ Data/VCTK.ipynb changed name to prepare_train_list.ipynb and completely revamped it, leaving only fractions of the original code
+ Demo/inference.ipynb moved to main directory, inference altered to output recordings for 2 speakers
