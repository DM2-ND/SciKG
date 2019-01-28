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

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from torch.autograd import Variable
from sklearn.utils import shuffle

from utils import *
from Stmt_Extraction_Net import Stmt_Extraction_Net

parser = argparse.ArgumentParser(description='Supervised MIMO (single featrue with multi-input gates)')

# Model parameters.
parser.add_argument('--train', type=str, default='./data/stmts-train.tsv',
					help='location of the labeled training set')
parser.add_argument('--eval', type=str, default='./data/stmts-eval.tsv',
					help='location of the evaluation set')
parser.add_argument('--language_model', type=str, default='./models/LM/model.pt',
					help='language model checkpoint to use')
parser.add_argument('--bert_model', type=str, default='./models/LM/',
					help='language model checkpoint to use')
parser.add_argument('--wordembed', type=str, default='./models/WE/pubmed-vectors=50.bin',
					help='wordembedding file for words')
parser.add_argument('--out_model', type=str, default='./models/supervised_model',
					help='location of the saved model')
parser.add_argument('--out_file', type=str, default='./results/evaluation_supervised_model',
					help='location of the saved results')
parser.add_argument('--check_point', type=str, default='./models/supervised_model_000111000.torch',
					help='continue to train the model from a checkpoint')
parser.add_argument('--use_gate', action='store_true',
					help='was used, not multi-input gates, ignore it')
parser.add_argument('--enhance', action='store_true',
					help='was used, ignore it')
parser.add_argument('--config', type=str, default='000000000',
					help='gates for three input sequence, i.e. LM(gate1, gate2, gate3), POS(gate1, gate2, gate3), CAP(gate1, gate2, gate3)')
parser.add_argument('--seed', type=int, default=824,
					help='random seed')
parser.add_argument('--num_pass', type=int, default=5,
					help='num of pass for evaluation')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--retrain', action='store_true',
					help='if continue to train the model from a checkpoint')
parser.add_argument('--bert_base', action='store_true',
					help='use bert_base as LM')
parser.add_argument('--bert_large', action='store_true',
					help='use bert_large as LM')
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)

if __name__ == '__main__':
	logging.debug(args)

	max_f1 = 0 # max macro-f1 of validation
	max_std = 0 # max std of macro-f1 of validation
	batch_size = 35
	dim = 50 # dimension of WE
	input_size = dim # input size of encoder
	hidden_dim = 300 # the number of LSTM units in encoder layer
	if args.bert_base:
		dataCenter = DataCenter(args.wordembed, args.bert_model+'bert-base-uncased.tar.gz', dim, device, 'bert-base')
	elif args.bert_large:
		dataCenter = DataCenter(args.wordembed, args.bert_model+'bert-large-uncased.tar.gz', dim, device, 'bert-large')
	else:
		dataCenter = DataCenter(args.wordembed, args.language_model, dim, device)
	dataCenter.loading_dataset(args.train, None, None, args.eval)

	config = [bool(int(i)) for i in args.config]
	assert len(config) == 9
	lm_config = config[:3]
	postag_config = config[3:6]
	cap_config = config[6:9]
	poscap_config = [False, False, False] # ingore POSCAP, which was used but not now

	print('lm config', lm_config)
	print('postag config', postag_config)
	print('cap config', cap_config)

	if args.use_gate: # it is not multi-input gates, ignore this is OK
		print('the gate is used.')
		out_model_name = args.out_model+'_gate_'
		out_file = args.out_file+'_gate_'
	else:
		print('the gate is not used.')
		out_model_name = args.out_model+'_'
		out_file = args.out_file+'_'

	if args.bert_base:
		out_model_name += 'bert_base_'
		out_file += 'bert_base_'
	elif args.bert_large:
		out_model_name += 'bert_large_'
		out_file += 'bert_large_'
	else:
		pass

	if args.enhance: # in our work, it is set False; poor performance when it is True
		out_model_name += 'enhance_'+args.config+'.torch'
		out_file += 'enhance_'+args.config+'.txt'
	else:
		out_model_name += (args.config+'.torch')
		out_file += (args.config+'.txt')
		
	print('out_model_name =', out_model_name)
	print('out_file =', out_file)
	
	if args.bert_base:
		stmt_extraction_net = Stmt_Extraction_Net(dataCenter.WordEmbedding, dataCenter.Word2ID, dataCenter.POS2ID, dataCenter.CAP2ID, dataCenter.POSCAP2ID, dim, input_size, hidden_dim, len(dataCenter.Tag2ID_fact), len(dataCenter.Tag2ID_condition), lm_config, postag_config, cap_config, poscap_config, device, args.seed, args.use_gate, args.enhance, 'bert-base')
	elif args.bert_large:
		stmt_extraction_net = Stmt_Extraction_Net(dataCenter.WordEmbedding, dataCenter.Word2ID, dataCenter.POS2ID, dataCenter.CAP2ID, dataCenter.POSCAP2ID, dim, input_size, hidden_dim, len(dataCenter.Tag2ID_fact), len(dataCenter.Tag2ID_condition), lm_config, postag_config, cap_config, poscap_config, device, args.seed, args.use_gate, args.enhance, 'bert-large')
	else:
		stmt_extraction_net = Stmt_Extraction_Net(dataCenter.WordEmbedding, dataCenter.Word2ID, dataCenter.POS2ID, dataCenter.CAP2ID, dataCenter.POSCAP2ID, dim, input_size, hidden_dim, len(dataCenter.Tag2ID_fact), len(dataCenter.Tag2ID_condition), lm_config, postag_config, cap_config, poscap_config, device, args.seed, args.use_gate, args.enhance)
	stmt_extraction_net.to(device)

	if args.retrain:
		print "loading model:"+args.check_point
		stmt_extraction_net.load_state_dict(torch.load(args.check_point))

	_weight_classes_fact = []
	for _id in range(len(dataCenter.ID2Tag_fact)):
		_weight_classes_fact.append((1.0/dataCenter.Tag2Num[dataCenter.ID2Tag_fact[_id]])*1000)
		# _weight_classes_fact.append(1.0)
	weight_classes_fact = torch.FloatTensor(_weight_classes_fact)
	print(weight_classes_fact)
	weight_classes_fact = weight_classes_fact.to(device)

	_weight_classes_condition = []
	for _id in range(len(dataCenter.ID2Tag_condition)):
		_weight_classes_condition.append((1.0/dataCenter.Tag2Num[dataCenter.ID2Tag_condition[_id]])*1000)
		# _weight_classes_condition.append(1.0)
	weight_classes_condition = torch.FloatTensor(_weight_classes_condition)
	print(weight_classes_condition)
	weight_classes_condition = weight_classes_condition.to(device)

	for epoch in range(1000):
		print('[epoch-%d] training ..' % epoch)
		apply_model(stmt_extraction_net, batch_size, 'TRAIN', dataCenter, device, weight_classes_fact, weight_classes_condition)
		print('validation ...')
		max_f1, max_std = evaluation(stmt_extraction_net, out_file, dataCenter, 0, 0, max_f1, max_std, out_model_name, args.num_pass, False, weight_classes_fact, weight_classes_condition)
