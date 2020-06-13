import sys
import requests_html as rh
import argparse
from tqdm import tqdm

def score(pwd):
	session = rh.HTMLSession()
	response = session.get('http://127.0.0.1/index.html?pass=' + pwd)
	response.html.render()
	span = response.html.find('#cups-passwordmeter-span', first=True)
	width = int(span.attrs['style'].split(';')[0][7:-2])
	score = width / 298.0
	response.close()
	session.close()
	return score
if __name__ == '__main__':
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
		scores.append(score(p))
	print(sum(scores) / len(scores))
	

