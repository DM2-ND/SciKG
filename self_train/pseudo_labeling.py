import sys, os, io
import random
import json
import torch
import pickle
import logging
import argparse
import gensim
import struct
import math
import itertools
import copy
import nltk

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from sklearn.metrics import precision_recall_fscore_support
from torch.autograd import Variable
from sklearn.utils import shuffle
from docopt import docopt

from config import *
from utils import *
from Stmt_Extraction_Net_try import Stmt_Extraction_Net

workdir = '/afs/crc.nd.edu/user/t/tjiang2/Private/www/workspace'
parser = argparse.ArgumentParser(description='PyTorch multi_input multi_output model')

# Model parameters.
parser.add_argument('--train', type=str, default=workdir+'/stmts-demo-train.tsv',
					help='location of the labeled training data')
parser.add_argument('--unlabeled_data', type=str, default=workdir+'/stmts-demo-test-unlabeled.tsv',
					help='location of the unlabeled data')
parser.add_argument('--prior_tag_fact', type=str, default=workdir+'/code_graph-based-SSL/pMP/data/testPrior',
					help='location of the unlabeled prior tag distribution for unlabeled data')
parser.add_argument('--prior_tag_condition', type=str, default=workdir+'/code_graph-based-SSL/pMP/data/testPrior',
					help='location of the unlabeled prior tag distribution for unlabeled data')
parser.add_argument('--valid', type=str, default=workdir+'/stmts-demo-valid.tsv',
					help='location of the labeled valid data')
parser.add_argument('--load_model', type=str, default='./models/model_word',
					help='location of the loading model')
parser.add_argument('--iter', type=str, default='1',
					help='the iteration')
parser.add_argument('--outfile', type=str, default='output_label.txt',
					help='the iteration')
parser.add_argument('--threshold', type=float, default=0.5,
					help='threshold for a postive pseudo tag')
parser.add_argument('--language_model', type=str, default=workdir+'/code-preprocessing/word_language_model/model.pt',
					help='language model checkpoint to use')
parser.add_argument('--wordembed', type=str, default=workdir+'/preprocessing/pubmed-vectors=50.bin',
					help='wordembedding file for words')
parser.add_argument('--tag_train', action='store_true',
					help='use tag to train')
parser.add_argument('--seed', type=int, default=824,
					help='random seed')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--all', action='store_true',
					help='label all')
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

cuda_flag = False
if torch.cuda.is_available():
	cuda_flag = True
	print 'CUDA is available.'
	# if "CUDA_VISIBLE_DEVICES" in os.environ:
		# dvice_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
	device_id = torch.cuda.current_device()
	print 'using device', device_id, torch.cuda.get_device_name(device_id)

if __name__ == '__main__':
	logging.debug(args)

	dim = 50
	
	dataCenter = DataCenter(args.wordembed, args.language_model, dim, device)
	dataCenter.loading_dataset(None, None, args.unlabeled_data, None)
	dataCenter.load_prior_tag('TEST', args.prior_tag_fact, args.prior_tag_condition)

	if args.outfile != 'output_label.txt':
		fname = args.outfile
	else:
		fname = workdir+'/prior-labeling/stmts-demo-prior-labeling_' + args.prior_tag_fact.split('unlabeledPrior/')[-1].replace('_fact.bin', '.tsv')
	prior_tag_outFile = open(fname, 'w')

	word2num = dict()
	stmts = []
	for index in range(len(dataCenter.TEST_SENTENCEs)):
		sentence = dataCenter.TEST_SENTENCEs[index]
		postags = dataCenter.TEST_POSTAGs[index]
		caps = dataCenter.TEST_CAPs[index]

		if not args.all:
			if len(sentence) > 20:
				continue

		prior_fact_label_list, prior_condition_label_list = dataCenter.TEST_OUTs[index]
		
		prior_fact_probs = np.max(prior_fact_label_list, 1)
		prior_fact_tagIDs = np.argmax(prior_fact_label_list, 1)

		prior_condition_probs = np.max(prior_condition_label_list, 1)
		prior_condition_tagIDs = np.argmax(prior_condition_label_list, 1)

		prior_tag_outFile.write('===== 00000000 stmt'+str(index)+' =====\n')
		prior_tag_outFile.write('WORD\t%s\n' % '\t'.join(sentence))
		prior_tag_outFile.write('POSTAG\t%s\n' % '\t'.join(postags))
		prior_tag_outFile.write('CAP\t%s\n' % '\t'.join(caps))

		stmt = ['stmt'+str(index),]

		prior_tags = []
		for j in range(len(prior_fact_probs)):
			prob = prior_fact_probs[j]
			tag = dataCenter.ID2Tag_fact[prior_fact_tagIDs[j]]
			prior_tags.append(tag)
		# facts = post_decoder(sentence, prior_tags, dataCenter.ID2Tag_fact)
		prior_tag_outFile.write('f\t%s\n' % '\t'.join(prior_tags))
		# for fact in facts:
		# 	for element in fact:
		# 		if element == 'NIL':
		# 			continue
		# 		if element[0] in word2num:
		# 			word2num[element[0]] += 1
		# 		else:
		# 			word2num[element[0]] = 1
		# stmt.append(facts)

		prior_tags = []
		for j in range(len(prior_condition_probs)):
			prob = prior_condition_probs[j]
			tag = dataCenter.ID2Tag_condition[prior_condition_tagIDs[j]]
			prior_tags.append(tag)
		# conditions = post_decoder(sentence, prior_tags, dataCenter.ID2Tag_fact)
		prior_tag_outFile.write('c\t%s\n' % '\t'.join(prior_tags))
		# for condition in conditions:
		# 	for element in condition:
		# 		if element == 'NIL':
		# 			continue
		# 		if element[0] in word2num:
		# 			word2num[element[0]] += 1
		# 		else:
		# 			word2num[element[0]] = 1
		# stmt.append(conditions)

		# stmts.append(stmt)

		if index % 1000 == 0:
			print index, 'done.'

	# for stmt in stmts:
	# 	index, facts, conditions = stmt
	# 	for fact in facts:
	# 		flag = True
	# 		for i in range(len(fact)):
	# 			if i in [1, 4] and e[0] == 'NIL':
	# 				continue
	# 			e = fact[i]
	# 			if e == 'NIL' or word2num[e[0]] < 100:
	# 				flag = False
	# 		if flag:
	# 			prior_tag_outFile.write(index+'\tfact\t')
	# 			prior_tag_outFile.write(str(fact))
	# 			prior_tag_outFile.write('\n')

	# 	for condition in conditions:
	# 		flag = True
	# 		for i in range(len(fact)):
	# 			if i in [1, 4] and e[0] == 'NIL':
	# 				continue
	# 			e = fact[i]
	# 			if e == 'NIL' or word2num[e[0]] < 100:
	# 				flag = False
	# 		if flag:
	# 			prior_tag_outFile.write(index+'\tcondition\t')
	# 			prior_tag_outFile.write(str(condition))
	# 			prior_tag_outFile.write('\n')
	prior_tag_outFile.write('#%d\n', index)
	prior_tag_outFile.close()
