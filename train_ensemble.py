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
from Stmt_Extraction_Net import *

parser = argparse.ArgumentParser(description='Supervised MIMO (multi-input gates, multi-input ensembles)')

# Model parameters.
parser.add_argument('--train', type=str, default='./data/stmts-train.tsv',
					help='location of the labeled training set')
parser.add_argument('--eval', type=str, default='./data/stmts-eval.tsv',
					help='location of the evaluation set')
parser.add_argument('--language_model', type=str, default='./models/LM/model.pt',
					help='language model checkpoint to use')
parser.add_argument('--bert_model', type=str, default='./models/LM/',
					help='language model checkpoint to use')
parser.add_argument('--wordembed', type=str, default='./models/WE/',
					help='wordembedding file for words')
parser.add_argument('--out_model', type=str, default='./models/ensemble_supervised_model',
					help='location of the saved model')
parser.add_argument('--out_file', type=str, default='./results/SeT_evaluation_ensemble_supervised_model',
					help='location of the saved results')
parser.add_argument('--use_gate', action='store_true',
					help='was used, not multi-input gates, ignore it')
parser.add_argument('--enhance', action='store_true',
					help='was used, ignore it')
parser.add_argument('--config', type=str, default='110',
					help='ensemble of three input sequence, i.e. LM, POS, CAP')
parser.add_argument('--seed', type=int, default=824,
					help='random seed')
parser.add_argument('--num_pass', type=int, default=5,
					help='num of pass for evaluation')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print 'using device', device_id, torch.cuda.get_device_name(device_id)

device = torch.device("cuda" if args.cuda else "cpu")
print 'DEVICE:', device

if __name__ == '__main__':
	logging.debug(args)

	max_f1 = 0 # max macro-f1 of validation
	max_std = 0 # max std of macro-f1 of validation
	batch_size = 35
	dim = 50 # dimension of WE

	dataCenter = DataCenter(args.wordembed, args.language_model, dim, device)
	dataCenter.loading_dataset(args.train, None, None, args.eval)

	config = [bool(int(i)) for i in args.config]
	assert len(config) == 3
	use_lm = config[0]
	use_postag = config[1]
	use_cap = config[2]

	print 'lm config', use_lm
	print 'postag config', use_postag
	print 'cap config', use_cap

	if args.use_gate:
		print 'the gate is used.'
		out_model_name = args.out_model+'_gate_'
		out_file = args.out_file+'_gate_'
	else:
		print 'the gate is not used.'
		out_model_name = args.out_model+'_'
		out_file = args.out_file+'_'

	if args.enhance:
		out_model_name += 'enhance_'+args.config+'.torch'
		out_file += 'enhance_'+args.config+'.txt'
	else:
		out_model_name += (args.config+'.torch')
		out_file += (args.config+'.txt')
		
	print 'out_model_name =', out_model_name
	print 'out_file =', out_file

	ensemble_model = Ensemble_Net(use_lm, use_postag, use_cap, len(dataCenter.Tag2ID_fact), device, args.seed)
	# ensemble_model = Ensemble_Net_new(use_lm, use_postag, use_cap, 600, len(dataCenter.Tag2ID_fact), device, args.seed)
	ensemble_model.to(device)

	_weight_classes_fact = []
	for _id in range(len(dataCenter.ID2Tag_fact)):
		_weight_classes_fact.append((1.0/dataCenter.Tag2Num[dataCenter.ID2Tag_fact[_id]])*1000)
		# _weight_classes_fact.append(1.0)
	weight_classes_fact = torch.FloatTensor(_weight_classes_fact)
	print weight_classes_fact
	weight_classes_fact = weight_classes_fact.to(device)

	_weight_classes_condition = []
	for _id in range(len(dataCenter.ID2Tag_condition)):
		_weight_classes_condition.append((1.0/dataCenter.Tag2Num[dataCenter.ID2Tag_condition[_id]])*1000)
		# _weight_classes_condition.append(1.0)
	weight_classes_condition = torch.FloatTensor(_weight_classes_condition)
	print weight_classes_condition
	weight_classes_condition = weight_classes_condition.to(device)

	# you can change it to your trained best models (LM, POS, CAP) for ensembles
	model_files = ['./models/best_model/supervised_model_SeT_AR_SH_011000000.torch', './models/best_model/supervised_model_SeT_AR_TC_SH_000111000.torch', './models/best_model/supervised_model_SeT_AR_TCDEL_SH_000000100.torch']
	print model_files

	lm_model = single_model_load(model_files[0], device, dataCenter, args.seed, args.use_gate, args.enhance)
	pos_model = single_model_load(model_files[1], device, dataCenter, args.seed, args.use_gate, args.enhance)
	cap_model = single_model_load(model_files[2], device, dataCenter, args.seed, args.use_gate, args.enhance)

	models = [lm_model, pos_model, cap_model]

	for epoch in range(1000):
		print '[epoch-%d] training ..' % epoch
		apply_model_ensemble(models, batch_size, 'TRAIN', dataCenter, device, weight_classes_fact, weight_classes_condition, ensemble_model)
		print 'validation ...'
		max_f1, max_std = evaluation_ensemble(models, out_file, dataCenter, 0, 0, max_f1, max_std, out_model_name, args.num_pass, False, weight_classes_fact, weight_classes_condition, ensemble_model, device)
