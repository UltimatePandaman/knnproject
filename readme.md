### Installation
Normal:
```bash
# create python environment and activate it
get_dataset.sh
get_models.sh
get_vocoder.sh
pip install -r requirements.txt
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
python train.py
```

### Inference
We prepared a small notebook for debugging inference. Just change the speakers in `speakers` of `inference.ipynb` notebook.

Inference of the whole testing set is done with `Data/test_list.txt`. Just place the path to the latest model into `Configs/config.yml` `pretrained_model:` option and run:
```bash
python inference.py
```
