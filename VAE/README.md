# Different VAE models

This repository contains the Keras implementation of the VAE model and some slight variations.
*VAE-basic* contains the standard VAE code.
*VAE-dropout* adds a dropout layer to the encoder in the VAE.
*VAE-GRU* uses masking in the encoder and both the encoder and decoder use GRU.
*VAE-LSTM* uses masking in the encoder and both the encoder and decoder use LSTM. 

## Creating datasets

First unzip the file in the upper level directory

```
unzip ../rockyou-processed.zip #Execute this command in the base directory of the project
```

## Training

To train the models, modify the hyperparameters in start of train.py if needed, then run:

```
python train.py
```
The program saves the model weights after each epoch in the file e\<EPOCH\>\_weights.h5 , where \<EPOCH\> is the epoch number.

## Sampling

Run the following command:

```
python -i test.py -w <WEIGHT_FILE_TO_LOAD>
```
This opens up an interactive shell. Run the following python function in this shell to generate as many samples as you want.
```
decode()
```
