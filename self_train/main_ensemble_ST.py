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

parser = argparse.ArgumentParser(description='PyTorch multi_input multi_output model')

# Model parameters.
parser.add_argument('--train', type=str, default=WORKDIR+'/stmts-demo-train.tsv',
					help='location of the labeled training data')
parser.add_argument('--udata', type=str, default='./udata/stmts-demo-unlabeled-pubmed',
					help='location of the unlabeled data')
parser.add_argument('--eval', type=str, default=WORKDIR+'/stmts-demo-eval.tsv',
					help='location of the unlabeled test data')
parser.add_argument('--check_point', type=str, default='./models/supervised_model_seperate_010100100000.torch',
					help='location of the saved model')
parser.add_argument('--out_file', type=str, default='./results/evaluation_supervised_model',
					help='location of the saved results')
parser.add_argument('--language_model', type=str, default=WORKDIR+'/code-preprocessing/word_language_model/model.pt',
					help='language model checkpoint to use')
parser.add_argument('--wordembed', type=str, default=WORKDIR+'/preprocessing/pubmed-vectors=50.bin',
					help='wordembedding file for words')
parser.add_argument('--use_gate', action='store_true')
parser.add_argument('--enhance', action='store_true')
parser.add_argument('--seed', type=int, default=824,
					help='random seed')
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')
parser.add_argument('--SH', action='store_true')
parser.add_argument('--AR', action='store_true')
parser.add_argument('--ST', action='store_true')
parser.add_argument('--DEL', action='store_true')
parser.add_argument('--STDEL', action='store_true')
parser.add_argument('--max_f1', type=list, default=50,
					help='random seed')
parser.add_argument('--max_std', type=list, default=1)

args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print 'using device', device_id, torch.cuda.get_device_name(device_id)

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

def auto_labeling(models, ensemble_model, dataCenter, data_file, AR, ST, DEL, STDEL):
	AR_fact_file_name = './association_rules_fact.txt'
	AR_condition_file_name = './association_rules_condition.txt'
	support_threshold = 3
	confidence_threshold = 0.7

	ar_correcter = AR_Correcter(AR_fact_file_name, AR_condition_file_name, support_threshold, confidence_threshold)

	MICS = zip(dataCenter.TEST_SENTENCEs, dataCenter.TEST_POSTAGs, dataCenter.TEST_CAPs, dataCenter.TEST_LM_SENTENCEs, dataCenter.TEST_POSCAPs, dataCenter.TEST_OUTs, dataCenter.instance_TEST)
	MICS.sort(key = lambda s: len(s[0]), reverse =True)
	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs, instance_list = zip(*MICS)
	SENTENCEs = list(SENTENCEs)
	POSTAGs = list(POSTAGs)
	CAPs = list(CAPs)
	POSCAPs = list(POSCAPs)
	LM_SENTENCEs = list(LM_SENTENCEs)
	OUTs = list(OUTs)
	instance_list = list(instance_list)
		
	assert len(SENTENCEs) == len(OUTs)
	print(len(SENTENCEs))

	batch_size = 100
	batches = len(SENTENCEs)/batch_size

	tag_outFile = open(data_file, 'w')
	count = 0
	for index in range(batches+1):
		SENTENCEs_batch = SENTENCEs[index*batch_size:(index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size:(index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size:(index+1)*batch_size]
		LM_SENTENCEs_batch = LM_SENTENCEs[index*batch_size:(index+1)*batch_size]
		POSCAPs_batch = POSCAPs[index*batch_size:(index+1)*batch_size]
		OUTs_batch = OUTs[index*batch_size:(index+1)*batch_size]
		instance_list_batch = instance_list[index*batch_size:(index+1)*batch_size]

		lm_input = single_model_predict(models[0], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), len(SENTENCEs_batch))
		pos_input = single_model_predict(models[1], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), len(SENTENCEs_batch))
		cap_input = single_model_predict(models[2], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), len(SENTENCEs_batch))

		predict_fact_batch, predict_condition_batch = ensemble_model((lm_input, pos_input, cap_input))

		for i in range(len(predict_fact_batch)):
			_, predicted_fact_tags = torch.max(predict_fact_batch[i], 1)
			_, predicted_conditions_tags = torch.max(predict_condition_batch[i], 1)
			assert len(OUTs_batch[i][0]) == len(instance_list_batch[i].OUT[0])
			fact_tags = []
			cond_tags = []
			if len(OUTs_batch[i][0]) > 15:
				continue
			for j in range(len(OUTs_batch[i][0])):
				y_predict = predicted_fact_tags[j].item()
				tag = dataCenter.ID2Tag_fact[y_predict]
				fact_tags.append(tag)
				y_predict = predicted_conditions_tags[j].item()
				tag = dataCenter.ID2Tag_condition[y_predict]
				cond_tags.append(tag)

			if AR:
				# print('using AR')
				sentence = SENTENCEs_batch[i]
				postag = POSTAGs_batch[i]
				VB_index = []
				for j in range(len(postag)):
					if postag[j].startswith('VB'):
						VB_index.append(j)
				j = 0
				while j < len(sentence):
					flag = False
					for k in range(len(sentence), j, -1):
						_A = postag[j:k]
						for kk in range(len(_A)):
							if _A[kk] == 'IN':
								_A[kk] += (':'+sentence[j+kk])
							_A[kk] += (':'+str(get_position(VB_index, j+kk)))
						_A = '\t'.join(_A)
						if _A in ar_correcter.A2B_fact:
							tags = ar_correcter.A2B_fact[_A].split('\t')
							flag = True
							fact_tags[j:k] = tags
							j = k
							break
					if not flag:
						j += 1

				IN_index = []
				for j in range(len(postag)):
					if postag[j] == 'IN':
						IN_index.append(j)

				j = 0
				while j < len(sentence):
					flag = False
					for k in range(len(sentence), j, -1):
						_A = postag[j:k]
						for kk in range(len(_A)):
							if _A[kk] == 'IN':
								_A[kk] += (':'+sentence[j+kk])
							_A[kk] += (':'+str(get_position(IN_index, j+kk)))
						_A = '\t'.join(_A)
						if _A in ar_correcter.A2B_cond:
							tags = ar_correcter.A2B_cond[_A].split('\t')
							flag = True
							cond_tags[j:k] = tags
							j = k
							break
					if not flag:
						j += 1

			if ST:
				fact_tags, corrected_fact = smooth_tag_sequence(fact_tags)
				cond_tags, corrected_cond = smooth_tag_sequence(cond_tags)

			if DEL:
				# print('using DEL')
				is_discarded_fact, fact_predicate_set = is_discarded(fact_tags)
				is_discarded_cond, cond_predicate_set = is_discarded(cond_tags)

				if is_discarded_fact or is_discarded_cond:
					continue
				if fact_predicate_set & cond_predicate_set != set():
					continue
			if STDEL:
				# print('using STDEL')
				fact_tags, corrected_fact = smooth_tag_sequence(fact_tags)
				cond_tags, corrected_cond = smooth_tag_sequence(cond_tags)
				if corrected_fact or corrected_cond:
					continue

			tag_outFile.write('===== '+str(instance_list_batch[i].paper_id)+' stmt'+str(instance_list_batch[i].stmt_id)+' =====\n')
			tag_outFile.write('WORD\t%s\n' % '\t'.join(instance_list_batch[i].SENTENCE))
			tag_outFile.write('POSTAG\t%s\n' % '\t'.join(POSTAGs_batch[i]))
			tag_outFile.write('CAP\t%s\n' % '\t'.join(CAPs_batch[i]))
			tag_outFile.write('f\t%s\n' % '\t'.join(fact_tags))
			tag_outFile.write('c\t%s\n' % '\t'.join(cond_tags))
			count += 1

	tag_outFile.write('#'+str(count)+'\n')
	tag_outFile.close()

if __name__ == '__main__':
	
	logging.debug(args)
	max_f1 = args.max_f1
	max_std = args.max_std
	min_loss = 999
	batch_size = 1000
	dim = 50
	input_size = dim
	hidden_dim = 300

	str_config = args.check_point.split('_')[-1].split('.torch')[0]
	config = [bool(int(i)) for i in str_config]
	assert len(config) == 3
	use_lm = config[0]
	use_postag = config[1]
	use_cap = config[2]

	print 'lm config', use_lm
	print 'postag config', use_postag
	print 'cap config', use_cap

	in_model_name = args.check_point
	
	out_model_string = 'hope_double_SeT'
	data_file = './data/hope_labeled'

	if args.AR:
		out_model_string += '_AR'
		data_file += '_AR'
	if args.ST:
		out_model_string += '_ST'
		data_file += '_ST'
	if args.DEL:
		out_model_string += '_DEL'
		data_file += '_DEL'
	if args.STDEL:
		out_model_string += '_STDEL'
		data_file += '_STDEL'

	if not args.enhance:
		data_file += ('_seperate_'+str_config)
	else:
		data_file += ('_enhance_'+str_config)

	out_model_name = (out_model_string).join(in_model_name.split('SeT'))
	out_file = './results/hope_evaluation_'+out_model_name.split('/')[-1].split('.torch')[0]+'.txt'

	print 'in_model_name =', in_model_name
	print 'out_model_name =', out_model_name
	print 'out_file =', out_file

	dataCenter = DataCenter(args.wordembed, args.language_model, dim, device)

	_weight_classes_fact = []
	for _id in range(len(dataCenter.ID2Tag_fact)):
		# _weight_classes_fact.append((1.0/dataCenter.Tag2Num[dataCenter.ID2Tag_fact[_id]])*1000)
		_weight_classes_fact.append(1.0)
	weight_classes_fact = torch.FloatTensor(_weight_classes_fact)
	print weight_classes_fact
	weight_classes_fact = weight_classes_fact.to(device)

	_weight_classes_condition = []
	for _id in range(len(dataCenter.ID2Tag_condition)):
		# _weight_classes_condition.append((1.0/dataCenter.Tag2Num[dataCenter.ID2Tag_condition[_id]])*1000)
		_weight_classes_condition.append(1.0)
	weight_classes_condition = torch.FloatTensor(_weight_classes_condition)
	print weight_classes_condition
	weight_classes_condition = weight_classes_condition.to(device)

	model_files = ['./models/supervised_model_SeT_AR_seperate_011000000000.torch', './models/supervised_model_SeT_AR_ST_seperate_000111000000.torch', './models/supervised_model_SeT_AR_STDEL_seperate_000000100000.torch']
	print model_files

	lm_model = single_model_load(model_files[0], device, dataCenter, args.seed, args.use_gate, args.enhance)
	pos_model = single_model_load(model_files[1], device, dataCenter, args.seed, args.use_gate, args.enhance)
	cap_model = single_model_load(model_files[2], device, dataCenter, args.seed, args.use_gate, args.enhance)

	models = [lm_model, pos_model, cap_model]

	ensemble_model = Ensemble_Net(use_lm, use_postag, use_cap, len(dataCenter.Tag2ID_fact), device, args.seed)
	ensemble_model.to(device)

	print "loading model parameters..."
	ensemble_model.load_state_dict(torch.load(in_model_name))
	print "loading done."

	udata_file = args.udata+'_part-1.tsv'
	data_file += '_'+udata_file.split('/')[-1]

	for index in range(6):
		udata_file = udata_file.replace('part'+str(index-1), 'part'+str(index))
		data_file = data_file.replace('part'+str(index-1), 'part'+str(index))

		print 'udata_file =', udata_file
		print 'data_file =', data_file

		dataCenter.loading_dataset(None, None, udata_file, None)
		auto_labeling(models, ensemble_model, dataCenter, data_file, args.AR, args.ST, args.DEL, args.STDEL)
		dataCenter.loading_dataset(args.train, None, data_file, args.eval)

		EXTRAIN_SENTENCEs, EXTRAIN_POSTAGs, EXTRAIN_CAPs, EXTRAIN_LM_SENTENCEs, EXTRAIN_POSCAPs, EXTRAIN_OUTs = shuffle(dataCenter.TEST_SENTENCEs, dataCenter.TEST_POSTAGs, dataCenter.TEST_CAPs, dataCenter.TEST_LM_SENTENCEs, dataCenter.TEST_POSCAPs, dataCenter.TEST_OUTs)
		TRAIN_SENTENCEs, TRAIN_POSTAGs, TRAIN_CAPs, TRAIN_LM_SENTENCEs, TRAIN_POSCAPs, TRAIN_OUTs = shuffle(dataCenter.TRAIN_SENTENCEs, dataCenter.TRAIN_POSTAGs, dataCenter.TRAIN_CAPs, dataCenter.TRAIN_LM_SENTENCEs, dataCenter.TRAIN_POSCAPs, dataCenter.TRAIN_OUTs)

		TRAIN_SENTENCEs.extend(EXTRAIN_SENTENCEs)
		TRAIN_POSTAGs.extend(EXTRAIN_POSTAGs)
		TRAIN_CAPs.extend(EXTRAIN_CAPs)
		TRAIN_LM_SENTENCEs.extend(EXTRAIN_LM_SENTENCEs)
		TRAIN_POSCAPs.extend(EXTRAIN_POSCAPs)
		TRAIN_OUTs.extend(EXTRAIN_OUTs)

		# for epoch in range(args.epoch):
		print 'training in extended set ..'
		max_f1, max_std, min_loss = retrain_ensemble_model(models, ensemble_model, out_file, 200, dataCenter, device, weight_classes_fact, weight_classes_condition, (TRAIN_SENTENCEs, TRAIN_POSTAGs, TRAIN_CAPs, TRAIN_LM_SENTENCEs, TRAIN_POSCAPs, TRAIN_OUTs), out_model_name, in_model_name, max_f1, max_std, min_loss, 5)
		print 'empty_cache'
		torch.cuda.empty_cache()

		print "loading model parameters..."
		ensemble_model.load_state_dict(torch.load(out_model_name))
		print "loading done."

		max_f1 = args.max_f1
		max_std = args.max_std
