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

from torch.autograd import Variable
from sklearn.utils import shuffle

from config import *
from utils import *
from Stmt_Extraction_Net import *

parser = argparse.ArgumentParser(description='Conditional Statement Extraction')

parser.add_argument('--udata', type=str, default='./self_train/udata/stmts-demo-unlabeled-small.tsv')
parser.add_argument('--check_point', type=str, default=WORKDIR+'models/best_model/SeT_AR_TC_SH_DEL_ensemble_supervised_model_111.torch',
					help='location of the best saved ensemble model')
parser.add_argument('--out_file', type=str, default='./predictions/stmts-demo-small-prediction')
parser.add_argument('--language_model', type=str, default='./models/LM/model.pt',
					help='language model checkpoint to use')
parser.add_argument('--wordembed', type=str, default='./models/WE/pubmed-vectors=50.bin',
					help='wordembedding file for words')
parser.add_argument('--seed', type=int, default=824,
					help='random seed')
parser.add_argument('--cuda', action='store_true',
					help='use CUDA')

args = parser.parse_args()

if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

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

def auto_labeling(models, ensemble_model, dataCenter, data_file):
	AR_fact_file_name = './association_rules_fact.txt'
	AR_condition_file_name = './association_rules_condition.txt'
	support_threshold = 3
	confidence_threshold = 0.7

	# ar_correcter = AR_Correcter(AR_fact_file_name, AR_condition_file_name, support_threshold, confidence_threshold)

	MICS = zip(dataCenter.TEST_SENTENCEs, dataCenter.TEST_POSTAGs, dataCenter.TEST_CAPs, dataCenter.TEST_POSCAPs, dataCenter.TEST_OUTs, dataCenter.instance_TEST)
	MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
	SENTENCEs, POSTAGs, CAPs, POSCAPs, OUTs, instance_list = zip(*MICS)
	SENTENCEs = list(SENTENCEs)
	POSTAGs = list(POSTAGs)
	CAPs = list(CAPs)
	POSCAPs = list(POSCAPs)
	OUTs = list(OUTs)
	instance_list = list(instance_list)
		
	assert len(SENTENCEs) == len(OUTs)
	print(len(SENTENCEs))

	batch_size = 50
	batches = len(SENTENCEs)//batch_size
	if len(SENTENCEs) % batch_size != 0:
		batches += 1

	nodes = []
	links = []
	graph = dict()
	graph['nodes'] = nodes
	graph['links'] = links
	node_id = 1
	node2id = dict()

	tag_outFile = open(data_file+'_tag_seqs.tsv', 'w')
	tuple_outFile = open(data_file+'_tuples.txt', 'w')
	count = 0
	f_id = 1
	c_id = 1
	for index in range(batches):
		SENTENCEs_batch = SENTENCEs[index*batch_size:(index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size:(index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size:(index+1)*batch_size]
		# LM_SENTENCEs_batch = LM_SENTENCEs[index*batch_size:(index+1)*batch_size]
		POSCAPs_batch = POSCAPs[index*batch_size:(index+1)*batch_size]
		OUTs_batch = OUTs[index*batch_size:(index+1)*batch_size]
		instance_list_batch = instance_list[index*batch_size:(index+1)*batch_size]

		LM_SENTENCEs_batch = []

		LM_hidden = dataCenter.LM_model.init_hidden(1)
		lm_data = dataCenter.lm_corpus.tokenize(SENTENCEs_batch)
		token_lm_embs = []
		for _index in range(len(lm_data)):
			token_lm_emb = []
			sentence = lm_data[_index]
			seqword = SENTENCEs_batch[_index]
			assert len(sentence) == len(seqword)
			data = sentence.view(len(sentence), -1)
			#print data.size()
			output = dataCenter.LM_model(data.to(dataCenter.device), LM_hidden)
			for index in range(len(sentence)):
				LM_emb = output[index]
				token_lm_emb.append(LM_emb)
			token_lm_emb = torch.cat(token_lm_emb, 0)
			LM_SENTENCEs_batch.append(token_lm_emb)
			assert len(instance_list_batch[_index].SENTENCE) == len(seqword)

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
			# if len(OUTs_batch[i][0]) > 15:
				# continue
			for j in range(len(OUTs_batch[i][0])):
				y_predict = predicted_fact_tags[j].item()
				tag = dataCenter.ID2Tag_fact[y_predict]
				fact_tags.append(tag)
				y_predict = predicted_conditions_tags[j].item()
				tag = dataCenter.ID2Tag_condition[y_predict]
				cond_tags.append(tag)

			stmt_str = str(instance_list_batch[i].paper_id)+' stmt'+str(instance_list_batch[i].stmt_id)
			tag_outFile.write('===== '+stmt_str+' =====\n')
			tag_outFile.write('WORD\t%s\n' % '\t'.join(instance_list_batch[i].SENTENCE))
			tag_outFile.write('POSTAG\t%s\n' % '\t'.join(POSTAGs_batch[i]))
			tag_outFile.write('CAP\t%s\n' % '\t'.join(CAPs_batch[i]))
			tag_outFile.write('f\t%s\n' % '\t'.join(fact_tags))
			tag_outFile.write('c\t%s\n' % '\t'.join(cond_tags))

			tuple_outFile.write('===== '+stmt_str+' =====\n')
			tuple_outFile.write('%s\n' % ' '.join(instance_list_batch[i].SENTENCE))

			fact_tuples = post_decoder(instance_list_batch[i].SENTENCE, fact_tags)
			condition_tuples = post_decoder(instance_list_batch[i].SENTENCE, cond_tags)
			for fact_tuple in fact_tuples:
				assert len(fact_tuple) == 5
				_1C, _1A, predicate, _3C, _3A = fact_tuple
				if predicate != 'NIL':
					predicate = predicate[0]+'#'+str(predicate[1])

				# subject nodes generation
				if _1C != 'NIL':
					_1C = _1C[0]+'#'+str(_1C[1])

				if _1A == 'NIL':
					subject = _1C
				else:
					_1A = _1A[0]+'#'+str(_1A[1])
					subject = '{'+_1C+':'+_1A+'}'

				# object nodes generation
				if _3C != 'NIL':
					_3C = _3C[0]+'#'+str(_3C[1])

				if _3A == 'NIL':
					_object = _3C
				else:
					_3A = _3A[0]+'#'+str(_3A[1])
					_object = '{'+_3C+':'+_3A+'}'
				tuple_outFile.write('f%d: (%s, %s, %s)\n' % (f_id,subject,predicate,_object))
				f_id += 1

			for condition_tuple in condition_tuples:
				assert len(condition_tuple) == 5
				_1C, _1A, predicate, _3C, _3A = condition_tuple
				if predicate != 'NIL':
					predicate = predicate[0]+'#'+str(predicate[1])

				# subject nodes generation
				if _1C != 'NIL':
					_1C = _1C[0]+'#'+str(_1C[1])

				if _1A == 'NIL':
					subject = _1C
				else:
					_1A = _1A[0]+'#'+str(_1A[1])
					subject = '{'+_1C+':'+_1A+'}'

				# object nodes generation
				if _3C != 'NIL':
					_3C = _3C[0]+'#'+str(_3C[1])

				if _3A == 'NIL':
					_object = _3C
				else:
					_3A = _3A[0]+'#'+str(_3A[1])
					_object = '{'+_3C+':'+_3A+'}'
				
				tuple_outFile.write('c%d: (%s, %s, %s)\n' % (c_id,subject,predicate,_object))
				c_id += 1

				count += 1
				if count % 1000 == 0:
					print(count,f_id-1,c_id-1, 'done')

	tag_outFile.write('#'+str(count)+'\n')
	tag_outFile.close()
	tuple_outFile.write('#'+str(count)+'\n')
	tuple_outFile.close()

if __name__ == '__main__':
	
	logging.debug(args)
	dim = 50

	str_config = args.check_point.split('_')[-1].split('.torch')[0]
	config = [bool(int(i)) for i in str_config]
	assert len(config) == 3
	use_lm = config[0]
	use_postag = config[1]
	use_cap = config[2]

	print('lm config', use_lm)
	print('postag config', use_postag)
	print('cap config', use_cap)

	in_model_name = args.check_point
	
	print('in_model_name =', in_model_name)
	print('out_file =', args.out_file)

	dataCenter = DataCenter(args.wordembed, args.language_model, dim, device)

	model_files = [WORKDIR+'models/best_model/supervised_model_SeT_AR_SH_011000000.torch', WORKDIR+'models/best_model/supervised_model_SeT_AR_TC_SH_000111000.torch', WORKDIR+'models/best_model/supervised_model_SeT_AR_TCDEL_SH_000000100.torch']
	print(model_files)

	lm_model = single_model_load(model_files[0], device, dataCenter, args.seed, False, False)
	pos_model = single_model_load(model_files[1], device, dataCenter, args.seed, False, False)
	cap_model = single_model_load(model_files[2], device, dataCenter, args.seed, False, False)

	models = [lm_model, pos_model, cap_model]

	ensemble_model = Ensemble_Net(use_lm, use_postag, use_cap, len(dataCenter.Tag2ID_fact), device, args.seed)
	ensemble_model.to(device)

	print("loading model parameters...")
	ensemble_model.load_state_dict(torch.load(in_model_name))
	print("loading done.")

	# dataCenter.loading_dataset(None, None, None, './data/stmts-eval.tsv')

	# evaluation_ensemble(models, args.out_file, dataCenter, 0, 0, 0, 0, './models/no', 5, True, None, None, ensemble_model, device, False, False)

	udata_file = args.udata

	dataCenter.loading_dataset(None, None, udata_file, None, True)
	auto_labeling(models, ensemble_model, dataCenter, args.out_file)
