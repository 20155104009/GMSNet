# GMSNet

Pytorch implementation for the paper entitled **"Grouped Multi-Scale Network for Real-world Image Denoising"**

We simplified the code since the original code applied some methods from our unpublished work.

The modified code may lower the performance slightly, but it is easy to follow.

## Data

Download the dataset and pretrained model from [GoogleDrive](https://drive.google.com/drive/folders/1n2NKB7z2r13HAqFUNe4UDjq7d1JoGhU0?usp=sharing).

Extract the files to `data` folder and `save_model` folder as follow:

```
~/
  data/
    SIDD_train/
      ... (scene id)
    SIDD_valid/
      ... (id)
    Syn_train/
      ... (id)
    DND/
      images_srgb/
        ... (mat files)
      ... (mat files)
  save_model/
    gmsnet/ (model name)
      best_model.pth.tar
```

#### Synthesize

We provide the code to generate a synthetic dataset using clean images.

The code you can find in `utils/syn`.

## Train

Train the GMSNet model:

```
python train.py
```

Or you can copy the template code to build your own model:

```
~/
  model/
    gmsnet.py
    template.py
    ... (your model)
```

Then you can train your own model:

```
python train.py --model ... (model name)
```

## Submit

Evaluate the trained model:

```
python submit_dnd.py --model ... (model name)
```

Further boost score using two useful tricks (self-ensemble and larger patch):

```
python submit_dnd.py --model ... (model name) --ensemble
```

The results are in `result/submit_dnd/bundled`

*Note that you should upload your results to the DND benchmark website by yourself.*
