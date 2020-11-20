# GMSNet

A simplified PyTorch implementation of our [GMSNet](https://doi.org/10.1109/LSP.2020.3039726).

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

Copy the template code to build your own model:

```
~/
  model/
    gmsnet.py
    template.py
    ... (your model)
```

Train your own model:

```
python train.py --model ... (model name)
```

## Submit

Evaluate the trained model (`--ensemble` for higher score):

```
python submit_dnd.py --model ... (model name) --ensemble
```

The results are in `result/submit_dnd/bundled`

*Note that you should upload your results to the DND benchmark website by yourself.*
