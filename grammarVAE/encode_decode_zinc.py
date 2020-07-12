import sys
import argparse
import molecule_vae
import numpy as np

print(molecule_vae.models.model_zinc.tf.__version__)

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular decoder network')
    parser.add_argument('--epochs', type=int, metavar='N', default=None,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=None,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--num_samples', type=int, metavar='N', default=100,
                        help='Number of samples to generate.')
    return parser.parse_args()

args = get_arguments()

# 1. load grammar VAE
L=args.latent_dim
E=args.epochs
grammar_weights = "results/zinc_vae_grammar_L" + str(L) + "_E" + str(E) + "_val.hdf5"
grammar_model = molecule_vae.ZincGrammarModel(grammar_weights,L)



# To encode and decode some example SMILES strings

# smiles = ["123456", "222222222","password","pass", "hello", "hi", "MyNameIs"]
# z1 = to_one_hot(smiles)
# for mol,real in zip(grammar_model.decode(z1),smiles):
# 	print(mol + '  ' + real)


# Decde random vectors and produce strings
z = np.random.rand(args.num_samples,L)

for i in grammar_model.decode(z):
	print(i)



# Sample datas from different models:
# 								Memorability			Guessability		stdevMem			stdevGuess
# gruConvDense_L50_E9.txt		0.1969241160730522  	0.6344780808225047	0.2319266875364846	0.2893683745536421
# gru_L50_E31.txt				0.02034126984126984		0.4437919463087248
# gru_L60_E31.txt				0.15262120468306034		0.5690514080121775
# orig_L52_E24.txt				0.034999999999999996	0.22536912751677857
# orig_L52_E34.txt				0.14983333333333332		0.2890268456375839
# orig_L206_E24.txt				0.0047619047619047615	0.33265100671140935
# orig_L206_E34.txt				0.03321428571428571		0.32953020134228184
# gruConvDenseMask_L110_E31.txt	0						0.2780536912751678	0					0.01870519017152946
# gruConvDenseMask_L60_E31.txt	0.028666666666666667	0.527986577181208	0.10025444620707512	0.2021396431977778
# gruMask_L150_E16.txt			0.16203174603174603		0.5289932885906041	0.19130647729091468	0.08815040299744324