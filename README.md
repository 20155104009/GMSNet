# GMSNet

Pytorch implementation for the paper entitled **"Grouped Multi-Scale Network for Real-world Image Denoising"**

This repo contains our image synthesis, model training and results submission code.

We simplified the training code since the original training code applied some methods from our unpublished work.

The modified code may lower the performance slightly, but it is easy to follow.

The datasets and pre-trained model will be uploaded soon.

## Data

SIDD dataset: [train]() and [valid]().

Generated synthetic dataset: [syn]().

Pretrained model: [GMSNet-A]().

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

First, adjust the training hyperparameters based on the hardware performance (44 GB GPU memory is necessary for current settings).

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

Further boost score using two useful tricks (self-ensemble and input padding):

```
python submit_dnd.py --model ... (model name) --ensemble
```

The results are in `result/submit_dnd/bundled`

*Note that you should upload your results to the DND benchmark website by yourself.*
