import sys, os, io
sys.path.append('../')

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
from Stmt_Extraction_Net import Stmt_Extraction_Net

parser = argparse.ArgumentParser(description='PyTorch multi_input multi_output model')

parser.add_argument('--train', type=str, default=WORKDIR+'/stmts-demo-train.tsv',
					help='location of the labeled training data')
parser.add_argument('--auto_labeled_data', type=str, default=WORKDIR+'/prior-labeling/stmts-demo-prior-labeling_supervised_model_seperate_010100100000.tsv',
					help='location of the auto labeled data')
parser.add_argument('--language_model', type=str, default=WORKDIR+'/code-preprocessing/word_language_model/model.pt',
					help='language model checkpoint to use')
parser.add_argument('--wordembed', type=str, default=WORKDIR+'/preprocessing/pubmed-vectors=50.bin',
					help='wordembedding file for words')
parser.add_argument('--seed', type=int, default=824,
					help='random seed')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--rule', action='store_true',
					help='generate rules')
parser.add_argument('--AR', action='store_true')
parser.add_argument('--ST', action='store_true')
parser.add_argument('--DEL', action='store_true')
parser.add_argument('--STDEL', action='store_true')
parser.add_argument('--all', action='store_true')

args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

def get_position(VB_index, index):
	min_dis = 999
	position = -1
	if index in VB_index:
		return 0
	for vi in VB_index:
		if math.fabs(index-vi) <= min_dis:
			min_dis = math.fabs(index-vi)
			position = -1 if (index-vi<0) else 1
	return position

def generate_associate_rules(dataCenter, window_size=20):
	A2num_fact = dict()
	A2num_cond = dict()
	AB_fact2num = dict()
	AB_cond2num = dict()
	for index in range(len(dataCenter.TRAIN_SENTENCEs)):
		sentence = dataCenter.TRAIN_SENTENCEs[index]
		postag = dataCenter.TRAIN_POSTAGs[index]
		cap = dataCenter.TRAIN_CAPs[index]
		out = dataCenter.TRAIN_OUTs[index]
		assert len(sentence) == len(postag) == len(cap) == len(out[0]) == len(out[1])
		out_fact = []
		out_cond = []
		for tag_id in out[0]:
			tag = dataCenter.ID2Tag_fact[tag_id]
			out_fact.append(tag)
		for tag_id in out[1]:
			tag = dataCenter.ID2Tag_condition[tag_id]
			out_cond.append(tag)

		VB_index = []
		for i in range(len(postag)):
			if postag[i].startswith('VB'):
				VB_index.append(i)

		IN_index = []
		for i in range(len(postag)):
			if postag[i] == 'IN':
				IN_index.append(i)

		for i in range(len(sentence)):
			for j in range(i+1, i+window_size+1):
				if j > len(sentence):
					break

				segment = postag[i:j]
				for k in range(len(segment)):
					if segment[k] == 'IN':
						segment[k] += (':'+sentence[i+k])
					# if segment[k].startswith('NN'):
					# 	segment[k] = 'NN'
					# if segment[k].startswith('JJ'):
					# 	segment[k] = 'JJ'
					segment[k] += (':'+str(get_position(VB_index, i+k)))
					#if cap[i+k] != 'O':
						#segment[k] += ':Ph'
				A = '\t'.join(segment)
				B_fact = '\t'.join(out_fact[i:j])
				if A in A2num_fact:
					A2num_fact[A] += 1
				else:
					A2num_fact[A] = 1

				if (A, B_fact) in AB_fact2num:
					AB_fact2num[(A, B_fact)] += 1
				else:
					AB_fact2num[(A, B_fact)] = 1



				segment = postag[i:j]
				for k in range(len(segment)):
					if segment[k] == 'IN':
						segment[k] += (':'+sentence[i+k])
					# if segment[k].startswith('NN'):
					# 	segment[k] = 'NN'
					# if segment[k].startswith('JJ'):
					# 	segment[k] = 'JJ'
					segment[k] += (':'+str(get_position(IN_index, i+k)))
					#if cap[i+k] != 'O':
						#segment[k] += ':Ph'
				A = '\t'.join(segment)
				B_cond = '\t'.join(out_cond[i:j])
				if A in A2num_cond:
					A2num_cond[A] += 1
				else:
					A2num_cond[A] = 1
				if (A, B_cond) in AB_cond2num:
					AB_cond2num[(A, B_cond)] += 1
				else:
					AB_cond2num[(A, B_cond)] = 1

	fo = open('./association_rules_fact.txt', 'w')
	co = open('./association_rules_condition.txt', 'w')
	for A, B_fact in AB_fact2num:
		if 'B' not in B_fact:
			continue

		i = 0
		tmp = B_fact.split('\t')
		while tmp[i] == 'O' and i < len(tmp):
			i += 1
		if i == len(tmp) or (not tmp[i].startswith('B')):
			continue

		support = AB_fact2num[(A, B_fact)]
		confidence = support / float(A2num_fact[A])
		lenth = len(A.split('\t'))
		if support < 2:
			continue
		#print '%s-->%s#%d#%.2f' % (A, B_fact, support, confidence)
		fo.write('%s-->%s#%d#%.2f\n' % (A, B_fact, support, confidence))

	for A, B_cond in AB_cond2num:
		if 'B' not in B_cond:
			continue
			
		i = 0
		tmp = B_cond.split('\t')
		while tmp[i] == 'O' and i < len(tmp):
			i += 1
		if i == len(tmp) or (not tmp[i].startswith('B')):
			continue

		support = AB_cond2num[(A, B_cond)]
		confidence = support / float(A2num_cond[A])
		lenth = len(A.split('\t'))
		if support < 2:
			continue
		#print '%s-->%s#%d#%.2f' % (A, B_cond, support, confidence)
		co.write('%s-->%s#%d#%.2f\n' % (A, B_cond, support, confidence))

	fo.close()
	co.close()

if __name__ == '__main__':
	dataCenter = DataCenter(args.wordembed, args.language_model, 50, device)
	dataCenter.loading_dataset(args.train, None, args.auto_labeled_data, None, True)

	if args.rule:
		generate_associate_rules(dataCenter, 20)

	sys.exit(1)

	AR_fact_file_name = './association_rules_fact.txt'
	AR_condition_file_name = './association_rules_condition.txt'
	support_threshold = 3
	confidence_threshold = 0.7

	ar_correcter = AR_Correcter(AR_fact_file_name, AR_condition_file_name, support_threshold, confidence_threshold)
	if not args.all:
		filename = args.auto_labeled_data.split('.tsv')[0]+'_short'
	else:
		filename = args.auto_labeled_data.split('.tsv')[0]
	if args.AR:
		filename += '_AR'
	if args.ST:
		filename += '_ST'
	if args.DEL:
		filename += '_DEL'
	if args.STDEL:
		filename += '_STDEL'
	filename += '.tsv'
	fo = open(filename, 'w')
	count = 0
	for index in range(len(dataCenter.TEST_SENTENCEs)):
		sentence = dataCenter.TEST_SENTENCEs[index]
		postag = dataCenter.TEST_POSTAGs[index]
		cap = dataCenter.TEST_CAPs[index]
		out = dataCenter.TEST_OUTs[index]

		out_fact = []
		for tag_id in out[0]:
			out_fact.append(dataCenter.ID2Tag_fact[tag_id])
		out_cond = []
		for tag_id in out[1]:
			out_cond.append(dataCenter.ID2Tag_condition[tag_id])

		if not args.all:
			if len(sentence) > 15:
				continue

		if args.AR:
			VB_index = []
			for i in range(len(postag)):
				if postag[i].startswith('VB'):
					VB_index.append(i)
			i = 0
			while i < len(sentence):
				flag = False
				for j in range(len(sentence), i, -1):
					_A = postag[i:j]
					for k in range(len(_A)):
						if _A[k] == 'IN':
							_A[k] += (':'+sentence[i+k])
						# if _A[k].startswith('NN'):
						# 	_A[k] = 'NN'
						# if _A[k].startswith('JJ'):
						# 	_A[k] = 'JJ'

						_A[k] += (':'+str(get_position(VB_index, i+k)))
					_A = '\t'.join(_A)
					if _A in ar_correcter.A2B_fact:
						tags = ar_correcter.A2B_fact[_A].split('\t')
						flag = True
						out_fact[i:j] = tags
						i = j
						break
				if not flag:
					i += 1

			IN_index = []
			for i in range(len(postag)):
				if postag[i] == 'IN':
					IN_index.append(i)

			i = 0
			while i < len(sentence):
				flag = False
				for j in range(len(sentence), i, -1):
					_A = postag[i:j]
					for k in range(len(_A)):
						if _A[k] == 'IN':
							_A[k] += (':'+sentence[i+k])
						# if _A[k].startswith('NN'):
						# 	_A[k] = 'NN'
						# if _A[k].startswith('JJ'):
						# 	_A[k] = 'JJ'
						_A[k] += (':'+str(get_position(IN_index, i+k)))
					_A = '\t'.join(_A)
					if _A in ar_correcter.A2B_cond:
						tags = ar_correcter.A2B_cond[_A].split('\t')
						flag = True
						out_cond[i:j] = tags
						i = j
						break
				if not flag:
					i += 1

		if args.ST:
			out_fact, corrected_fact = smooth_tag_sequence(out_fact)
			out_cond, corrected_cond = smooth_tag_sequence(out_cond)

		if args.DEL:
			is_discarded_fact, fact_predicate_set = is_discarded(out_fact)
			is_discarded_cond, cond_predicate_set = is_discarded(out_cond)
			if is_discarded_fact or is_discarded_cond:
				continue
			if fact_predicate_set & cond_predicate_set != set():
				print out_fact
				print out_cond
				continue

		if args.STDEL:
			out_fact, corrected_fact = smooth_tag_sequence(out_fact)
			out_cond, corrected_cond = smooth_tag_sequence(out_cond)
			if corrected_fact or corrected_cond:
				continue

		#print len(sentence), len(out_fact)
		fo.write('===== 00000000 stmt'+str(index)+' =====\n')
		fo.write('WORD\t%s\n' % '\t'.join(sentence))
		fo.write('POSTAG\t%s\n' % '\t'.join(postag))
		fo.write('CAP\t%s\n' % '\t'.join(cap))
		fo.write('f\t%s\n' % '\t'.join(out_fact))
		fo.write('c\t%s\n' % '\t'.join(out_cond))
		count += 1
	fo.write('#%d\n' % count)
	fo.close()
