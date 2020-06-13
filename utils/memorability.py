import sys
import editdistance as ed
import numpy as np
import argparse
from tqdm import tqdm

def score(pwd, words):
	score = min([ed.eval(pwd, w) for w in words])
	word = words[np.argmin([ed.eval(pwd, w) for w in words])]
	clip = min(len(pwd), len(word))
	score = max(0, clip - score)
	score /= (clip * 1.0)
	return score, word

if __name__ == '__main__':
	with open('100k-lower.txt', 'r') as f:
	    words = f.readlines()
	    words = [x.rstrip('\n') for x in words]

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', help='File from which to read passwords')
	parser.add_argument('-p', '--pwd', help='Single password if want to calculate its score')
	args = parser.parse_args()
	if args.file is not None:
		with open(args.file, 'r') as f:
			pwds = [x.strip() for x in f.readlines()]
	else:
		pwds = [args.pwd]
	scores = []
	for p in tqdm(pwds):
		s, w = score(p, words)
		scores.append(s)
	print(sum(scores) / len(scores))
	
