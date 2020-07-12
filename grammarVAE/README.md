# Grammar Variational Autoencoder

This repository contains training and sampling code for the paper: <a href="https://arxiv.org/abs/1703.01925">Grammar Variational Autoencoder</a>.

## Creating datasets

To create the molecule datasets, call in this directory:

```
unzip ../rockyou-processed.zip -d data/
python make_zinc_dataset_grammar.py
```

## Training

To train the models, call:

```
python train_zinc.py % the grammar model
python train_zinc.py --latent_dim=2 --epochs=50` % train a model with a 2D latent space and 50 epochs
```

## Sampling

The file molecule_vae.py can be used to encode and decode SMILES strings. For a demo run:

```
python encode_decode_zinc.py
python encode_decode_zinc.py --latent_dim=2 --epochs=50 --num_samples=200 % Sample 200 random strings from the model trained with a 2D latent space and 50 epochs
```

## Other details

- Model is defined in `models/model_zinc.py`. Modify `_buildEncoder` and `_encoderMeanVar` in a similar manner to change the model i.e. add same additonal layers to both functions.

- The folder `results/` contains some trained models which can be samples using the above commands. Note that the arguments need to be fed appropriately to the files.

- Some samples already generated from the modles are in `samples/`.