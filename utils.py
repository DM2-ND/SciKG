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
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from config import *
from load_pretrained_word_embeddings import *
from Stmt_Extraction_Net import *

class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []

	def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
		return self.word2idx[word]

	def __len__(self):
		return len(self.idx2word)

class Corpus(object):
	def __init__(self):
		self.dictionary = Dictionary()
		self.load_dictionary()

	def load_dictionary(self):
		print('loading dictionary for language model ...')
		index = 0
		filepath1 = WORKDIR+'resources/LM_dictionary1.txt'
		filepath2 = WORKDIR+'resources/LM_dictionary2.txt'
		filepath3 = WORKDIR+'resources/LM_dictionary3.txt'
		with io.open(filepath1, 'r', encoding="utf8") as f:
			tokens = 0
			for line in f:
				words = line.split() + ['<eos>']
				tokens += len(words)
				for word in words:
					self.dictionary.add_word(word)
				index += 1
				if index%100000 == 0:
					print(index, 'done.')
		with io.open(filepath2, 'r', encoding="utf8") as f:
			tokens = 0
			for line in f:
				words = line.split() + ['<eos>']
				tokens += len(words)
				for word in words:
					self.dictionary.add_word(word)
				index += 1
				if index%10000 == 0:
					print(index, 'done.')
		with io.open(filepath3, 'r', encoding="utf8") as f:
			tokens = 0
			for line in f:
				words = line.split() + ['<eos>']
				tokens += len(words)
				for word in words:
					self.dictionary.add_word(word)
				index += 1
				if index%10000 == 0:
					print(index, 'done.')
		print('done.')

	def tokenize(self, seqword_list):
		ids_list = []
		index = 0
		word_set = set()
		for seqword in seqword_list:
			tokens = len(seqword)
			ids = torch.LongTensor(tokens)
			token = 0
			for word in seqword:
				if word not in self.dictionary.word2idx:
					ids[token] = self.dictionary.word2idx['<eos>']
					word_set.add(word)
				else:
					ids[token] = self.dictionary.word2idx[word]
				token += 1
			ids_list.append(ids)
			index += 1
			if index%10000 == 0:
				print(index, 'done.')
		return ids_list

class Instance(object):
	"""docstring for Instance"""
	def __init__(self, paper_id, stmt_id, multi_input, multi_output):
		super(Instance, self).__init__()
		self.paper_id = paper_id
		self.stmt_id = stmt_id
		self.multi_input = multi_input
		self.multi_output = multi_output
		self.SENTENCE = None
		self.POSTAG = None
		self.CAP = None
		self.POSCAP = None
		self.LM_SENTENCE = None
		self.OUT = None

class DataCenter(object):
	"""docstring for Instance"""
	def __init__(self, wordembed_File, language_model_File, dim, device, lm_type='normal'):
		super(DataCenter, self).__init__()

		self.oov_set = set()
		self.device = device
		self.lm_type = lm_type

		self.POS2ID, self.ID2POS = self.getTag2ID(WORKDIR+'resources/PosTag2ID.txt')
		self.CAP2ID, self.ID2CAP = self.getTag2ID(WORKDIR+'resources/CAPTag2ID.txt')
		self.POSCAP2ID, self.ID2POSCAP = self.getTag2ID(WORKDIR+'resources/POSCAPTag2ID_new.txt')
		self.Tag2ID_fact, self.ID2Tag_fact = self.getTag2ID(WORKDIR+'resources/OutTag2ID_fact.txt')
		self.Tag2ID_condition, self.ID2Tag_condition = self.getTag2ID(WORKDIR+'resources/OutTag2ID_condition.txt')

		self.Tag2Num = dict()

		wv = Gensim(wordembed_File, dim)
		self.word2vec = wv.word2vec_dict

		print('loading language model... ...')
		if lm_type == 'normal':
			with open(language_model_File, 'rb') as f:
				# self.LM_model = torch.load(f, map_location='cpu')
				if torch.cuda.is_available():
					self.LM_model = torch.load(f).to(self.device)
				else:
					self.LM_model = torch.load(f, map_location='cpu').to(self.device)
			self.lm_corpus = Corpus()
		else:
			assert lm_type in ['bert-base', 'bert-large']
			self.LM_model = BertModel.from_pretrained(language_model_File)
			self.tokenizer = BertTokenizer.from_pretrained(lm_type+'-uncased')

		self.PAD = '<pad>'
		self.WordEmbedding = [self.word2vec[self.PAD].view(1, -1),]
		self.Word2ID = dict()
		self.ID2Word = dict()
		self.Word2ID[self.PAD] = 0
		self.ID2Word[0] = self.PAD

		for word in self.word2vec:
			if word == self.PAD or word in self.Word2ID:
				continue
			_id = len(self.WordEmbedding)
			self.Word2ID[word] = _id
			self.ID2Word[_id] = word
			self.WordEmbedding.append(self.word2vec[word].view(1, -1))
		self.WordEmbedding = torch.cat(self.WordEmbedding)

		print(self.WordEmbedding.size())

		self.TRAIN_SENTENCEs = []
		self.TRAIN_POSTAGs = []
		self.TRAIN_CAPs = []
		self.TRAIN_POSCAPs = []
		self.TRAIN_LM_SENTENCEs = []
		self.TRAIN_OUTs = []

		self.VALID_SENTENCEs = []
		self.VALID_POSTAGs = []
		self.VALID_CAPs = []
		self.VALID_POSCAPs = []
		self.VALID_LM_SENTENCEs = []
		self.VALID_OUTs = []

		self.TEST_SENTENCEs = []
		self.TEST_POSTAGs = []
		self.TEST_CAPs = []
		self.TEST_POSCAPs = []
		self.TEST_LM_SENTENCEs = []
		self.TEST_OUTs = []

		self.EVAL_SENTENCEs = []
		self.EVAL_POSTAGs = []
		self.EVAL_CAPs = []
		self.EVAL_POSCAPs = []
		self.EVAL_LM_SENTENCEs = []
		self.EVAL_OUTs = []

		self.max_outpus = 0

		self.instance_TRAIN = []
		self.instance_VALID = []
		self.instance_TEST = []
		self.instance_EVAL = []

		self.paper_id_set_TRAIN = set()
		self.paper_id_set_VALID = set()
		self.paper_id_set_TEST = set()
		self.paper_id_set_EVAL = set()

	def getTag2ID(self, fileName):
		tag2ID = dict()
		ID2Tag = dict()
		with open(fileName, 'r') as f:
			for line in f:
				tag, _id = line.strip().split(' ')
				tag2ID[tag] = int(_id)
				ID2Tag[int(_id)] = tag
		return tag2ID, ID2Tag

	def _add_instance(self, dataset_type, paper_id, stmt_id, multi_input, multi_output, attr_tuple):
		if len(multi_input) == 0:
			return

		SENTENCEs, POSTAGs, CAPs, OUTs = attr_tuple
		instance = Instance(paper_id, stmt_id, multi_input, multi_output)
		senLen = len(instance.multi_input[0][-1])
		for _input in instance.multi_input:
			seq_name = _input[0]
			seq = _input[1]
			if seq_name == 'WORD':
				sentence = []
				for word in seq:
					word = word.lower()
					if word not in self.word2vec:
						sentence.append('<unk>')
						self.oov_set.add(word)
					else:
						sentence.append(word)
				assert len(sentence) == senLen
				SENTENCEs.append(sentence)
				instance.SENTENCE = seq
				# formatted_anno_file.write('WORD:\t'+'\t'.join(sentence)+'\n')
			elif seq_name == 'POSTAG':
				assert len(seq) == senLen
				POSTAGs.append(seq)
				instance.POSTAG = seq
				# formatted_anno_file.write('POSTAG:\t'+'\t'.join(seq)+'\n')
			else:
				assert len(seq) == senLen
				CAPs.append(seq)
				instance.CAP = seq
				# formatted_anno_file.write('CAP:\t'+'\t'.join(seq)+'\n')

		# print 'OUT:'
		facts_out = [self.Tag2ID_fact['O']] * senLen
		conditions_out = [self.Tag2ID_condition['O']] * senLen

		outs = []
		for _output in instance.multi_output:
			key = _output[0]
			seq = _output[1]
			for index in range(len(seq)):
				tag = seq[index]
				# print tag,
				if key.startswith('f'):
					if tag != 'O':
						facts_out[index] = self.Tag2ID_fact[tag]
						# tag_id = self.Tag2ID_fact[tag]
						# if facts_out[index] == self.Tag2ID_fact['O']:
						# 	facts_out[index] = dict()
						# if tag_id in facts_out[index]:
						# 	facts_out[index][tag_id] += 1 
						# else:
						# 	facts_out[index][tag_id] = 1
				else:
					if tag != 'O':
						conditions_out[index] = self.Tag2ID_condition[tag]
						# tag_id = self.Tag2ID_condition[tag]
						# if conditions_out[index] == self.Tag2ID_condition['O']:
						# 	conditions_out[index] = dict()
						# if tag_id in conditions_out[index]:
						# 	conditions_out[index][tag_id] += 1 
						# else:
						# 	conditions_out[index][tag_id] = 1
						
		for index in range(len(facts_out)):
			tag_id_fact = facts_out[index]
			tag_id_condition = conditions_out[index]
			if dataset_type == 'TRAIN':
				self.count_tag(self.ID2Tag_fact[tag_id_fact])
				self.count_tag(self.ID2Tag_condition[tag_id_condition])
			# if tag_id_fact != self.Tag2ID_fact['O']:
			# 	tag_id_fact = sorted(tag_id_fact.iteritems(), key=lambda (k,v): (v,k), reverse=True)[0][0]
			# 	facts_out[index] = tag_id_fact
			# 	if dataset_type == 'TRAIN':
			# 		self.count_tag(self.ID2Tag_fact[tag_id_fact])
			# else:
			# 	if dataset_type == 'TRAIN':
			# 		self.count_tag(self.ID2Tag_fact[tag_id_fact])

			# if tag_id_condition != self.Tag2ID_condition['O']:
			# 	tag_id_condition = sorted(tag_id_condition.iteritems(), key=lambda (k,v): (v,k), reverse=True)[0][0]
			# 	conditions_out[index] = tag_id_condition
			# 	if dataset_type == 'TRAIN':
			# 		self.count_tag(self.ID2Tag_condition[tag_id_condition])
			# else:
			# 	if dataset_type == 'TRAIN':
			# 		self.count_tag(self.ID2Tag_condition[tag_id_condition])
				
		assert len(facts_out) == len(conditions_out) == senLen

		# formatted_anno_file.write('f:\t'+'\t'.join([self.ID2Tag_fact[tag_id] for tag_id in facts_out])+'\n')
		# formatted_anno_file.write('c:\t'+'\t'.join([self.ID2Tag_condition[tag_id] for tag_id in conditions_out])+'\n')

		outs = [facts_out, conditions_out]
		OUTs.append(outs)
		instance.OUT = outs

		assert len(SENTENCEs) == len(POSTAGs) ==len(CAPs) == len(OUTs)
		if len(SENTENCEs) % 10000 == 0:
			print(len(SENTENCEs), 'done')

		instance_list = getattr(self, 'instance_'+dataset_type)
		instance_list.append(instance)

	def count_tag(self, tag):
		if tag not in self.Tag2Num:
			self.Tag2Num[tag] = 0
		self.Tag2Num[tag] += 1
		

	def _loading_dataset(self, dataset_type, dataFile, no_LM):

		SENTENCEs = getattr(self, dataset_type+'_SENTENCEs')
		POSTAGs = getattr(self, dataset_type+'_POSTAGs')
		CAPs = getattr(self, dataset_type+'_CAPs')
		POSCAPs = getattr(self, dataset_type+'_POSCAPs')
		LM_SENTENCEs = getattr(self, dataset_type+'_LM_SENTENCEs')
		OUTs = getattr(self, dataset_type+'_OUTs')
		instance_list = getattr(self, 'instance_'+dataset_type)

		del SENTENCEs[:]
		del POSTAGs[:]
		del CAPs[:]
		del POSCAPs[:]
		del LM_SENTENCEs[:]
		del OUTs[:]
		del instance_list[:]

		attr_tuple = (SENTENCEs, POSTAGs, CAPs, OUTs)

		logging.debug('loading '+dataset_type+' data from '+dataFile)

		paper_id_set = getattr(self, 'paper_id_set_'+dataset_type)
		paper_id = 'none'
		stmt_id = '0'
		multi_input = []
		multi_output = []
		previous = False

		# formatted_anno_file = open(dataFile.replace('.tsv', '_formatted.tsv'), 'w')

		with open(dataFile, 'r') as fd:
			for line in fd:
				if line.startswith('=====') or line.startswith('#'):
					# conclude the previous instance
					if previous:
						self._add_instance(dataset_type, paper_id, stmt_id, multi_input, multi_output, attr_tuple)

					# start a new instance
					if not line.startswith('====='):
						continue
					paper_id = line.strip().split('===== ')[-1].split(' stmt')[0]
					paper_id_set.add(paper_id)
					stmt_id = line.split('stmt')[-1].split(' =====')[0]
					# logging.debug('doing the paper '+paper_id+', stmt '+stmt_id)
					multi_input = []
					multi_output = []
					previous = True
					# formatted_anno_file.write(line)
					continue
				line_list = line.strip('\n').split('\t')
				seq_name = line_list[0]
				seq = line_list[1:]
				if seq_name in ['WORD', 'POSTAG', 'CAP']:
					multi_input.append((seq_name, seq))
				else:
					multi_output.append((seq_name, seq))

		# formatted_anno_file.close()

		instance_list = getattr(self, 'instance_'+dataset_type)
		for i in range(len(CAPs)):
			CAP = CAPs[i]
			POSTAG = POSTAGs[i]
			POSCAP = []
			assert len(CAP) == len(POSTAG)
			for j in range(len(CAP)):
				# if CAP[j] == 'O':
				# 	if POSTAG[j] not in self.POS2ID:
				# 	 	POSCAP.append('SYM') 
				# 	else:
				# 		POSCAP.append(POSTAG[j])
				# else:
				# 	POSCAP.append(CAP[j])
				if POSTAG[j] not in self.POS2ID:
					POSCAP.append('SYM-'+CAP[j])
				else:
					POSCAP.append(POSTAG[j]+'-'+CAP[j])
			assert len(CAP) == len(POSTAG) == len(POSCAP)
			POSCAPs.append(POSCAP)
			assert instance_list[i].CAP == CAP
			instance_list[i].POSCAP = POSCAP

		print(len(SENTENCEs), len(POSTAGs), len(CAPs), len(POSCAPs))
		assert len(CAPs) == len(POSCAPs)
		
		self.LM_model.eval()

		if no_LM:
			return
		print('applying the language model ...')
		if self.lm_type == 'normal':
			LM_hidden = self.LM_model.init_hidden(1)
			lm_data = self.lm_corpus.tokenize(SENTENCEs)
			token_lm_embs = []
			for _index in range(len(lm_data)):
				token_lm_emb = []
				sentence = lm_data[_index]
				seqword = SENTENCEs[_index]
				assert len(sentence) == len(seqword)
				data = sentence.view(len(sentence), -1)
				#print data.size()
				output = self.LM_model(data.to(self.device), LM_hidden)
				# output = self.LM_model(data, LM_hidden)
				for index in range(len(sentence)):
					LM_emb = output[index]
					token_lm_emb.append(LM_emb)
				token_lm_emb = torch.cat(token_lm_emb, 0)
				LM_SENTENCEs.append(token_lm_emb)
				assert len(instance_list[_index].SENTENCE) == len(seqword)
				instance_list[_index].LM_SENTENCE = token_lm_emb
				if _index % 10000 == 0:
					print(_index)
		else:
			oov = set()
			for _index in range(len(SENTENCEs)):
				seqword = SENTENCEs[_index]
				new_seqword = []
				assert '[UNK]' in self.tokenizer.vocab
				for w in seqword:
					assert w == w.lower()
					if w not in self.tokenizer.vocab:
						new_seqword.append('[UNK]')
						oov.add(w)
					else:
						new_seqword.append(w)
				assert len(new_seqword) == len(seqword)
				indexed_tokens = self.tokenizer.convert_tokens_to_ids(new_seqword)
				tokens_tensor = torch.tensor([indexed_tokens])
				encoded_layers, _ = self.LM_model(tokens_tensor, token_type_ids=None, output_all_encoded_layers=False)
				token_lm_emb = encoded_layers[0]
				LM_SENTENCEs.append(token_lm_emb.to(self.device))
				# LM_SENTENCEs.append(token_lm_emb)
				assert len(instance_list[_index].SENTENCE) == len(seqword)
				instance_list[_index].LM_SENTENCE = token_lm_emb
				if _index % 10000 == 0:
					print(_index)
			print(len(oov))
		# assert len(SENTENCEs) == len(LM_SENTENCEs)
		# print(LM_SENTENCEs[0].size())

		print('done.')

	# def loading_dataset(self, trainFile, validFile, testFile):
	def loading_dataset(self, trainFile, validFile, testFile, evalFile, no_LM=False):

		if trainFile != None:
			self._loading_dataset('TRAIN', trainFile, no_LM)
			# for pos in self.POS2ID:
			# 	print(pos, self.POS2ID[pos])
			# for cap in self.CAP2ID:
			# 	print(cap, self.CAP2ID[cap])
			# for tag in self.Tag2ID_fact:
			# 	print(tag, self.Tag2ID_fact[tag], 
			# 	print(self.Tag2Num[tag] if tag in self.Tag2Num else ''
			# for tag in self.Tag2ID_condition:
			# 	print(tag, self.Tag2ID_condition[tag],
			# 	print(self.Tag2Num[tag] if tag in self.Tag2Num else ''

		if evalFile != None:
			self._loading_dataset('EVAL', evalFile, no_LM)

		if validFile != None:
			self._loading_dataset('VALID', validFile, no_LM)

		if testFile != None:
			self._loading_dataset('TEST', testFile, no_LM)

	def get_evaluation(self, valid_prop):

		VALID_SENTENCEs = []
		VALID_POSTAGs = []
		VALID_CAPs = []
		VALID_POSCAPs = []
		VALID_LM_SENTENCEs = []
		VALID_OUTs = []
		VALID_instances = []

		TEST_SENTENCEs = []
		TEST_POSTAGs = []
		TEST_CAPs = []
		TEST_POSCAPs = []
		TEST_LM_SENTENCEs = []
		TEST_OUTs = []
		TEST_instances = []

		#print range(len(self.EVAL_SENTENCEs))
		id_list = random.sample(range(len(self.EVAL_SENTENCEs)), int(len(self.EVAL_SENTENCEs)*valid_prop))
		#print id_list
		for index in range(len(self.EVAL_SENTENCEs)):
			if index not in id_list:
				TEST_SENTENCEs.append(self.EVAL_SENTENCEs[index])
				TEST_POSTAGs.append(self.EVAL_POSTAGs[index])
				TEST_CAPs.append(self.EVAL_CAPs[index])
				TEST_POSCAPs.append(self.EVAL_POSCAPs[index])
				TEST_LM_SENTENCEs.append(self.EVAL_LM_SENTENCEs[index])
				TEST_OUTs.append(self.EVAL_OUTs[index])
				TEST_instances.append(self.instance_EVAL[index])
			else:
				VALID_SENTENCEs.append(self.EVAL_SENTENCEs[index])
				VALID_POSTAGs.append(self.EVAL_POSTAGs[index])
				VALID_CAPs.append(self.EVAL_CAPs[index])
				VALID_POSCAPs.append(self.EVAL_POSCAPs[index])
				VALID_LM_SENTENCEs.append(self.EVAL_LM_SENTENCEs[index])
				VALID_OUTs.append(self.EVAL_OUTs[index])
				VALID_instances.append(self.instance_EVAL[index])

		VALID_DATA = (VALID_SENTENCEs, VALID_POSTAGs, VALID_CAPs, VALID_POSCAPs, VALID_LM_SENTENCEs, VALID_OUTs, VALID_instances)

		TEST_DATA = (TEST_SENTENCEs, TEST_POSTAGs, TEST_CAPs, TEST_POSCAPs, TEST_LM_SENTENCEs, TEST_OUTs, TEST_instances)

		# for data in VALID_DATA:
		# 	print(len(data),

		# for data in TEST_DATA:
		# 	print(len(data),

		return VALID_DATA, TEST_DATA

	def load_prior_tag(self, dataset_type, prior_tag_fact, prior_tag_condition):
		print('loading the prior tag distribution for '+dataset_type+' data ...')
		SENTENCEs = getattr(self, dataset_type+'_SENTENCEs')
		OUTs = getattr(self, dataset_type+'_OUTs')

		class_num = len(self.Tag2ID_fact)
		prior_tag_distrib_list_fact = []
		fbin = open(prior_tag_fact, 'rb')
		d_str = fbin.read()
		d_len = len(d_str)/4
		float_tuple = struct.unpack(str(d_len)+'f',d_str)
		print(len(float_tuple)/class_num, len(float_tuple)%class_num)
		float_index = 0
		tuple_len = len(float_tuple)
		while float_index < tuple_len:
			distrib_prior = []
			for i in range(class_num):
				distrib_prior.append(float_tuple[float_index])
				float_index += 1
			assert len(distrib_prior) == class_num
			prior_tag_distrib_list_fact.append(distrib_prior)
			if (float_index/11) % 100000 == 0:
				print(float_index/11, 'done')
		fbin.close()

		class_num = len(self.Tag2ID_condition)
		prior_tag_distrib_list_condition = []
		fbin = open(prior_tag_condition, 'rb')
		d_str = fbin.read()
		d_len = len(d_str)/4
		float_tuple = struct.unpack(str(d_len)+'f',d_str)
		print(len(float_tuple)/class_num, len(float_tuple)%class_num)
		float_index = 0
		tuple_len = len(float_tuple)
		while float_index < tuple_len:
			distrib_prior = []
			for i in range(class_num):
				distrib_prior.append(float_tuple[float_index])
				float_index += 1
			assert len(distrib_prior) == class_num
			prior_tag_distrib_list_condition.append(distrib_prior)
			if (float_index/11) % 100000 == 0:
				print(float_index/11, 'done')
		fbin.close()

		print('length of prior_tag_distrib_list_fact:', len(prior_tag_distrib_list_fact))
		print('length of prior_tag_distrib_list_condition:', len(prior_tag_distrib_list_condition))

		token_index = 0
		for _index in range(len(SENTENCEs)):
			token_tag_distrib = []
			seqword = SENTENCEs[_index]
			for index in range(len(seqword)):
				token_tag_distrib.append((prior_tag_distrib_list_fact[token_index], prior_tag_distrib_list_condition[token_index]))
				token_index += 1
			token_tag_distrib = zip(*token_tag_distrib)
			token_tag_distrib[0] = list(token_tag_distrib[0])
			token_tag_distrib[1] = list(token_tag_distrib[1])
			OUTs[_index] = token_tag_distrib
		assert len(SENTENCEs) == len(OUTs)
		print(token_index)
		print('done.')

def getStmtLabel(docid_filelabel):

	# load concepts, attributes, predicates, and stmts

	docid2struc = {}

	for [docid,filelabel] in docid_filelabel:

		nid2tuple = {}
		hid2tuple = {}
		fid2tuple = {}
		cid2tuple = {}
		sid2stmts = {}

		fr = open(filelabel,'r')
		for line in fr:
			text = line.strip()
			if text == '': continue
			head = text[0]
			if head == '#':
				continue
			elif head == 'n':
				pos = text.find('[')
				arr = text[pos+1:-1].split(',')
				assert(len(arr) == 3 and arr[1] == 'as')
				_id = text[:pos-1]
				assert(not _id in nid2tuple)
				nid2tuple[_id] = [['C',arr[0]],arr[1],['C',arr[2]]]
			elif head == 'h':
				pos = text.find('[')
				arr = text[pos+1:-1].split(',')
				assert(len(arr) == 3 and arr[1] == 'contain')
				_id = text[:pos-1]
				assert(not _id in hid2tuple)		
				hid2tuple[_id] = [['C',arr[0]],arr[1],['C',arr[2]]]
			elif head == 'f':
				pos = text.find('[')
				arr = text[pos+1:-1].split(',')
				assert(len(arr) == 3)
				_tuple = [[],'',[]]
				for i in [0,2]:
					if ':' in arr[i]:
						_arr = arr[i][1:-1].split(':')
						assert(len(_arr) == 2)
						_tuple[i] = ['A',_arr[0],_arr[1]]
					else:
						if arr[i] == 'NIL':
							_tuple[i] = ['N',arr[i]]
						else:
							_tuple[i] = ['C',arr[i]]
				_tuple[1] = arr[1]
				_id = text[:pos-1]
				try:
					assert(not _id in fid2tuple)  
				except:
					print(text)
					sys.exit(1)
				fid2tuple[_id] = _tuple
			elif head == 'c':
				pos = text.find('[')
				arr = text[pos+1:-1].split(',')
				assert(len(arr) == 3)
				_tuple = [[],'',[]]		
				for i in [0,2]:
					if ':' in arr[i]:
						_arr = arr[i][1:-1].split(':')
						assert(len(_arr) == 2)
						_tuple[i] = ['A',_arr[0],_arr[1]]
					else:
						if arr[i] == 'NIL':
							_tuple[i] = ['N',arr[i]]
						else:
							_tuple[i] = ['C',arr[i]]
				_tuple[1] = arr[1]
				_id = text[:pos-1]
				assert(not _id in cid2tuple)
				cid2tuple[_id] = _tuple
			elif head == 's':
				if text[:4] == 'stmt':
					arr = text.split(' ')
					stmt = [[],[],'NIL']
					assert(arr[1] == '=')
					for i in range(2,len(arr)):
						_id = arr[i]
						if _id[0] == 'f':
							assert(_id in fid2tuple)
							stmt[0].append(_id)
						elif _id[0] == 'c':
							assert(_id in cid2tuple)
							stmt[1].append(_id)
						elif _id[0] == '(' and _id[-1] == ')':
							stmt[2] = _id[1:-1]
						else:
							assert(False)
					sid = int(arr[0][4:])
					if not sid in sid2stmts:
						sid2stmts[sid] = []
					sid2stmts[sid].append(stmt)
				elif text[:4] == 's???':
					continue
				else:
					assert(False)
			else:
				assert(False)
		fr.close()

		docid2struc[docid] = [nid2tuple,hid2tuple,fid2tuple,cid2tuple,sid2stmts]

	return docid2struc

def parsing(text):
	seqword,seqpostag,seqanno = [],[],[]
	elems = text.split(' ')
	n = len(elems)
	for i in range(n):
		elem = elems[i]
		if elem.startswith('$C'):
			_arr = elem.split(':')
			phrase = _arr[1]
			arrphrase = phrase.split('_')
			arrpostag = _arr[2].split('_')
			_n = len(arrphrase)
			for j in range(_n):
				seqword.append(arrphrase[j].lower())
				seqpostag.append(arrpostag[j])
				if j == 0:
					seqanno.append('B-C')
				else:
					seqanno.append('I-C')
		elif elem.startswith('$A'):
			_arr = elem.split(':')
			arrphrase = _arr[1].split('_')
			arrpostag = _arr[2].split('_')
			_n = len(arrphrase)
			for j in range(_n):
				seqword.append(arrphrase[j].lower())
				seqpostag.append(arrpostag[j])
				if j == 0:
					seqanno.append('B-A')
				else:
					seqanno.append('I-A')
		elif elem.startswith('$P'):
			_arr = elem.split(':')
			arrphrase = _arr[1].split('_')
			arrpostag = _arr[2].split('_')
			_n = len(arrphrase)
			for j in range(_n):
				seqword.append(arrphrase[j].lower())
				seqpostag.append(arrpostag[j])
				if j == 0:
					seqanno.append('B-P')
				else:
					seqanno.append('I-P')
		else:
			_arr = elem.split(':')
			seqword.append(_arr[0].lower())
			seqpostag.append(_arr[1])
			seqanno.append('O')
	assert len(seqword) == len(seqpostag) == len(seqanno)
	return seqword, seqpostag, seqanno

def getTag2ID(fileName):
	tag2ID = dict()
	with open(fileName, 'r') as f:
		for line in f:
			tag, _id = line.strip().split(' ')
			tag2ID[tag] = int(_id)
	return tag2ID

def getOneHot(index, lenth):
	assert index < lenth
	vector = np.asarray([0]*lenth)
	vector[index] = 1
	return vector

class AR_Correcter(object):
	"""docstring for AR_Correcter"""
	def __init__(self, AR_fact_file_name, AR_condition_file_name, support_threshold, confidence_threshold):
		super(AR_Correcter, self).__init__()
		self.A2B_fact = dict()
		self.A2conf_fact = dict()
		self.A2B_cond = dict()
		self.A2conf_cond = dict()

		self._load_AR_file(AR_fact_file_name, AR_condition_file_name, support_threshold, confidence_threshold)

	def _load_AR_file(self, AR_fact_file_name, AR_condition_file_name, support_threshold, confidence_threshold):
		fi = open(AR_fact_file_name, 'r')
		ci = open(AR_condition_file_name, 'r')

		for line in fi:
			A_B, support, confidence = line.strip().split('#')
			support = int(support)
			confidence = float(confidence)
			A, B = A_B.split('-->')

			if support < support_threshold or confidence < confidence_threshold:
				continue

			if not self._is_good_rule(A.split('\t'), B.split('\t')):
				continue

			if A in self.A2B_fact:
				if self.A2conf_fact[A] < confidence:
					self.A2B_fact[A] = B
					self.A2conf_fact[A] = confidence
			else:
				self.A2B_fact[A] = B
				self.A2conf_fact[A] = confidence

		for A in self.A2B_fact:
			print(A, self.A2B_fact[A], self.A2conf_fact[A])

		
		for line in ci:
			A_B, support, confidence = line.strip().split('#')
			support = int(support)
			confidence = float(confidence)
			A, B = A_B.split('-->')

			if support < support_threshold or confidence < confidence_threshold:
				continue

			if not self._is_good_rule(A.split('\t'), B.split('\t')):
				continue

			if A in self.A2B_cond:
				if self.A2conf_cond[A] < confidence:
					self.A2B_cond[A] = B
					self.A2conf_cond[A] = confidence
			else:
				self.A2B_cond[A] = B
				self.A2conf_cond[A] = confidence

		for A in self.A2B_cond:
			print(A, self.A2B_cond[A], self.A2conf_cond[A])

		fi.close()
		ci.close()

	def _is_good_rule(self, pos_sequence, tag_sequence):
		role_set = set()
		for tag in tag_sequence:
			if tag == 'O':
				continue
			role = tag[3]
			role_set.add(role)
		if len(role_set) < 2 or ('2' not in role_set):
			return False
		return True

def smooth_tag_sequence(tag_sequence):
	new_tag_sequence = ['O']
	index = 0
	lenth = len(tag_sequence)
	flag = False
	while index < lenth:
		tag = tag_sequence[index]

		if tag == 'O':
			new_tag = 'O'
		elif not tag.endswith('2P') and not tag.endswith('A'):
			if new_tag_sequence[-1].endswith('2P') or new_tag_sequence[-1].endswith('A'):
				new_tag = 'B'+tag[1:]
			elif new_tag_sequence[-1].startswith('B') or new_tag_sequence[-1].startswith('I'):
				new_tag = 'I'+new_tag_sequence[-1][1:]
			else:
				new_tag = 'B'+tag[1:]
		elif tag.endswith('2P'):
			if new_tag_sequence[-1].endswith('2P'):
				assert tag[1:] == new_tag_sequence[-1][1:]
				new_tag = 'I'+new_tag_sequence[-1][1:]
			else:
				new_tag = 'B'+tag[1:]
		else:
			assert tag.endswith('A')
			if new_tag_sequence[-1].endswith('A'):
				new_tag = 'I'+new_tag_sequence[-1][1:]
			else:
				new_tag = 'B'+tag[1:]

		if new_tag != tag:
			flag = True

		new_tag_sequence.append(new_tag)
		index += 1

	assert len(new_tag_sequence[1:]) == len(tag_sequence)
	return new_tag_sequence[1:], flag

def is_discarded(tag_sequence):
	role_set = set()
	role_type_set = set()
	predicate_set = set()
	for index in range(len(tag_sequence)):
		tag = tag_sequence[index]
		if tag == 'O':
			continue
		if '2P' in tag:
			predicate_set.add(index)
		role = tag[3]
		role_type = tag[3:]
		role_set.add(role)
		role_type_set.add(role_type)

	if len(role_set) < 3: # or '2P' not in role_type_set:
		return True, predicate_set

	if '1A' in role_type_set and '1C' not in role_type_set:
		return True, predicate_set

	if '3A' in role_type_set and '3C' not in role_type_set:
		return True, predicate_set

	return False, predicate_set

class Metrics(object):
	"""docstring for Metrics"""
	def __init__(self):
		super(Metrics, self).__init__()
		self.TP = 0
		self.FP = 0
		self.TN = 0
		self.FN = 0

	def Precision(self):
		if (self.TP == self.FP) and (self.TP == 0):
			return 0
		return float(self.TP) / (self.TP + self.FP)

	def Recall(self):
		if (self.TP == self.FN) and (self.FN == 0):
			return 0
		return float(self.TP) / (self.TP + self.FN)

	def F_1(self):
		precision = self.Precision()
		recall = self.Recall()
		if (precision == recall) and (precision == 0):
			return 0
		return 2 * (precision * recall) / (precision + recall)

# def _get_precision_recall(matrix):
# 	precisions = np.asarray(matrix)
# 	precisions = np.asarray([precisions[i].max() for i in range(len(precisions))])
# 	precision = precisions.sum()/float(len(predicted_tuples)*5)

# 	recalls = np.asarray(matrix).transpose()
# 	recalls = np.asarray([recalls[i].max() for i in range(len(recalls))])
# 	recall = recalls.sum()/float(len(truth_tuples)*5)

# 	return (precision, recall, precisions, recalls)

def match_score(truth_tuples, predicted_tuples):
	if len(truth_tuples) == 0:
		truth_tuples = [['NIL', 'NIL', 'NIL', 'NIL', 'NIL']]
	if len(predicted_tuples) == 0:
		predicted_tuples = [['NIL', 'NIL', 'NIL', 'NIL', 'NIL']]

	matrix = []
	# matrix_1C = []
	# matrix_1A = []
	# matrix_2P = []
	# matrix_3C = []
	# matrix_3A = []

	for predicted_tuple in predicted_tuples:
		scores = []
		# scores_1C = []
		# scores_1A = []
		# scores_2P = []
		# scores_3C = []
		# scores_3A = []
		for truth_tuple in truth_tuples:
			assert len(truth_tuple) == 5 and len(predicted_tuple) == 5
			score = 0
			# score_1C = 0
			# score_1A = 0
			# score_2P = 0
			# score_3C = 0
			# score_3A = 0
			for index in range(len(truth_tuple)):
				t_part = truth_tuple[index]
				p_part = predicted_tuple[index]
				if t_part == p_part:
					score += 1
					# if index == 0:
					# 	score_1C += 1
					# elif index == 1:
					# 	score_1A += 1
					# elif index == 1:
					# 	score_2P += 1
					# elif index == 1:
					# 	score_3C += 1
					# else:
					# 	score_3A += 1
			scores.append(score)
			# scores_1C.append(score_1C)
			# scores_1A.append(score_1A)
			# scores_2P.append(score_2P)
			# scores_3C.append(score_3C)
			# scores_3A.append(score_3A)
		matrix.append(scores)
		# matrix_1C.append(scores_1C)
		# matrix_1A.append(scores_1A)
		# matrix_2P.append(scores_2P)
		# matrix_3C.append(scores_3C)
		# matrix_3A.append(scores_3A)

	precisions = np.asarray(matrix)
	precisions = np.asarray([precisions[i].max() for i in range(len(precisions))])
	precision = precisions.sum()/float(len(predicted_tuples)*5)

	recalls = np.asarray(matrix).transpose()
	recalls = np.asarray([recalls[i].max() for i in range(len(recalls))])
	recall = recalls.sum()/float(len(truth_tuples)*5)

	return precision, recall, precisions, recalls

	#return _get_precision_recall(matrix) , _get_precision_recall(matrix_1C), _get_precision_recall(matrix_1A), _get_precision_recall(matrix_2P), _get_precision_recall(matrix_3C), _get_precision_recall(matrix_3A)

def is_blocked(start, end, predicate_set):
	if start > end:
		return True
	for predicate in predicate_set:
		if predicate[1] > start and predicate[1] < end:
			return True

# tuple from tag sequence
def post_decoder(words, predicted_fact_tags, ID2Tag=None):
	facts = []

	f1_set = set()
	f1a_set = set()
	f2_set = set()
	f3_set = set()
	f3a_set = set()

	index = 0
	while index < len(words):
		if type(predicted_fact_tags[index]) == type(1):
			tagID = predicted_fact_tags[index]
			tag = ID2Tag[tagID]
		else:
			tag = predicted_fact_tags[index]

		if tag.startswith('B-'):
			string_tag = tag
			string = words[index]
			string_start = index
			index += 1
			if index < len(words):
				if type(predicted_fact_tags[index]) == type(1):
					tagID = predicted_fact_tags[index]
					tag = ID2Tag[tagID]
				else:
					tag = predicted_fact_tags[index]
				while tag.startswith('I'):
					string += ('_' + words[index])
					index += 1
					if index < len(words):
						if type(predicted_fact_tags[index]) == type(1):
							tagID = predicted_fact_tags[index]
							tag = ID2Tag[tagID]
						else:
							tag = predicted_fact_tags[index]
					else:
						break
			string_end = index
			if string_tag.endswith('1C'):
				f1_set.add((string, string_start, string_end))
			elif string_tag.endswith('1A'):
				f1a_set.add((string, string_start, string_end))
				#f1a_set = set()
			elif string_tag.endswith('2P'):
				f2_set.add((string, string_start, string_end))
			elif string_tag.endswith('3C'):
				f3_set.add((string, string_start, string_end))
			elif string_tag.endswith('3A'):
				f3a_set.add((string, string_start, string_end))
				#f3a_set = set()
			else:
				print('error!', string, string_tag, string_start, string_end)
				sys.exit(1)

		else:
			index += 1

	MIN = 30
	subject2predicate = dict()
	for subject in f1_set:
		min_dis = MIN
		t_predicate = None
		for predicate in f2_set: 
			if is_blocked(subject[1], predicate[1], f2_set):
				continue
			dis = predicate[1]-subject[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_predicate = predicate
		subject2predicate[subject] = t_predicate
	# print subject2predicate

	object2predicate = dict()
	for _object in f3_set:
		min_dis = MIN
		t_predicate = None
		for predicate in f2_set:
			if is_blocked(predicate[1], _object[1], f2_set):
				continue
			dis = _object[1]-predicate[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_predicate = predicate
		object2predicate[_object] = t_predicate
	# print object2predicate

	predicate2subject = dict()
	for predicate in f2_set:
		min_dis = MIN
		t_subject = None
		for subject in f1_set:
			if is_blocked(subject[1], predicate[1], f2_set):
				continue
			dis = predicate[1]-subject[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_subject = subject
		predicate2subject[predicate] = t_subject
	# print predicate2subject

	predicate2object = dict()
	for predicate in f2_set:
		min_dis = MIN
		t_object = None
		for _object in f3_set:
			if is_blocked(predicate[1], _object[1], f2_set):
				continue
			dis = _object[1]-predicate[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_object = _object
		predicate2object[predicate] = t_object
	# print predicate2object

	subject2object = dict()
	for subject in f1_set:
		min_dis = MIN
		t_object = None
		for _object in f3_set:
			dis = _object[1]-subject[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_object = _object
		subject2object[subject] = t_object
	# print subject2object

	object2subject = dict()
	for _object in f3_set:
		min_dis = MIN
		t_subject = None
		for subject in f1_set:
			dis = _object[1]-subject[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_subject = subject
		object2subject[_object] = t_subject
	# print object2subject

	attrib2subject = dict()
	for attrib in f1a_set:
		min_dis = 3
		t_subject = None
		for subject in f1_set:
			dis = subject[1] - attrib[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_subject = subject
		attrib2subject[attrib] = t_subject
	# print attrib2subject

	attrib12predicate = dict()
	for attrib in f1a_set:
		min_dis = 5
		t_predicate = None
		for predicate in f2_set:
			dis = predicate[1] - attrib[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_predicate = predicate
		attrib12predicate[attrib] = t_predicate

	attrib32predicate = dict()
	for attrib in f3a_set:
		min_dis = 5
		t_predicate = None
		for predicate in f2_set:
			dis = attrib[1] - predicate[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_predicate = predicate
		attrib32predicate[attrib] = t_predicate

	attrib2object = dict()
	for attrib in f3a_set:
		min_dis = 3
		t_object = None
		for _object in f3_set:
			dis = _object[1] - attrib[2]
			if dis < min_dis and dis >= 0:
				min_dis = dis
				t_object = _object
		attrib2object[attrib] = t_object
	# print attrib2object

	sets = [f1_set, f2_set, f3_set]
	for _set in sets:
		_set.add('NIL')

	candidate_facts = itertools.product(f1_set, set(['NIL']), f2_set, f3_set, set(['NIL']))

	for fact in candidate_facts:
		# print 'candidate_facts:', fact
		fact = list(fact)
		subject = fact[0]
		predicate = fact[2]
		_object = fact[3]

		if subject == 'NIL' and _object == 'NIL':
			if predicate!='NIL' and (predicate2subject[predicate]==None and predicate2object[predicate]==None):
				facts.append(fact)
			else:
				continue
		elif predicate == 'NIL':
			if subject == 'NIL' or _object == 'NIL':
				continue
			elif (subject2object[subject] != _object) and (object2subject[_object] != subject):
				continue
			elif (subject2predicate[subject] != None) or (object2predicate[_object] != None):
				continue
			else:
				facts.append(fact)
		else:
			if subject == 'NIL' and (predicate2subject[predicate]!=None or object2subject[_object]!=None):
				continue
			elif _object == 'NIL' and (predicate2object[predicate]!=None or subject2object[subject]!=None):
				continue
			else:
				if subject != 'NIL' and (subject2predicate[subject] != predicate) and (predicate2subject[predicate] != subject):
					#print '1'
					continue
				elif _object != 'NIL' and (object2predicate[_object] != predicate) and (predicate2object[predicate] != _object):
					#print '2'
					continue
				elif (subject != 'NIL' and _object != 'NIL') and (subject2object[subject] != _object) and (object2subject[_object] != subject):
					#print '3'
					continue
				else:
					facts.append(fact)
	
	extend_facts = []
	for attrib in f1a_set:
		subject = attrib2subject[attrib]
		if subject == None:
			predicate = attrib12predicate[attrib]
			for fact in facts:
				t_predicate = fact[2]
				if t_predicate != predicate:
					continue
				if fact[0] == 'NIL' and fact[1] == 'NIL':
					fact[1] = attrib
			continue
		for fact in facts:
			if fact[2][0] == 'in':
				continue
			t_subject = fact[0]
			if t_subject != subject:
				continue
			if fact[1] == 'NIL':
				fact[1] = attrib
			elif fact[1] != attrib:
				new_fact = copy.deepcopy(fact)
				new_fact[1] = attrib
				extend_facts.append(new_fact)
			for _fact in facts:
				if _fact == fact:
					continue
				if _fact[2:] == fact[2:] and _fact[0] != 'NIL':
					if _fact[0][1] - subject[2] < 0 or _fact[0][1] - subject[2] > 3:
						continue
					if _fact[1] == 'NIL':
						_fact[1] = attrib
					# elif _fact[1] != attrib:
					# 	new_fact = copy.deepcopy(_fact)
					# 	new_fact[1] = attrib
					# 	extend_facts.append(new_fact)
	facts.extend(extend_facts)

	extend_facts = []
	for attrib in f3a_set:
		_object = attrib2object[attrib]
		if _object == None:
			predicate = attrib32predicate[attrib]
			for fact in facts:
				t_predicate = fact[2]
				if t_predicate != predicate:
					continue
				if fact[3] == 'NIL' and fact[4] == 'NIL':
					fact[4] = attrib
			continue
		for fact in facts:
			t_object = fact[3]
			if t_object != _object:
				continue
			if fact[4] == 'NIL':
				fact[4] = attrib
			elif fact[4] != attrib:
				new_fact = copy.deepcopy(fact)
				new_fact[4] = attrib
				extend_facts.append(new_fact)
			for _fact in facts:
				if _fact == fact:
					continue
				if _fact[:2] == fact[:2] and _fact[3] != 'NIL':
					if _fact[3][1] - _object[2] < 0 or _fact[3][1] - _object[2] > 3:
						continue
					if _fact[4] == 'NIL':
						_fact[4] = attrib
					# elif _fact[4] != attrib:
					# 	new_fact = copy.deepcopy(_fact)
					# 	new_fact[4] = attrib
					# 	extend_facts.append(new_fact)
	facts.extend(extend_facts)

	return facts

# def post_decoder(words, predicted_fact_tags, ID2Tag=None):
# 	facts = []

# 	f1_set = set()
# 	f1a_set = set()
# 	f2_set = set()
# 	f3_set = set()
# 	f3a_set = set()

# 	index = 0
# 	while index < len(words):
# 		if type(predicted_fact_tags[index]) == type(1):
# 			tagID = predicted_fact_tags[index]
# 			tag = ID2Tag[tagID]
# 		else:
# 			tag = predicted_fact_tags[index]

# 		if tag.startswith('B-'):
# 			string_tag = tag
# 			string = words[index].lower()
# 			string_start = index
# 			index += 1
# 			if index < len(words):
# 				if type(predicted_fact_tags[index]) == type(1):
# 					tagID = predicted_fact_tags[index]
# 					tag = ID2Tag[tagID]
# 				else:
# 					tag = predicted_fact_tags[index]
# 				while tag.startswith('I'):
# 					string += ('_' + words[index].lower())
# 					index += 1
# 					if index < len(words):
# 						if type(predicted_fact_tags[index]) == type(1):
# 							tagID = predicted_fact_tags[index]
# 							tag = ID2Tag[tagID]
# 						else:
# 							tag = predicted_fact_tags[index]
# 					else:
# 						break
# 			string_end = index
# 			if string_tag.endswith('1C'):
# 				f1_set.add((string, string_start, string_end))
# 			elif string_tag.endswith('1A'):
# 				f1a_set.add((string, string_start, string_end))
# 				#f1a_set = set()
# 			elif string_tag.endswith('2P'):
# 				f2_set.add((string, string_start, string_end))
# 			elif string_tag.endswith('3C'):
# 				f3_set.add((string, string_start, string_end))
# 			elif string_tag.endswith('3A'):
# 				f3a_set.add((string, string_start, string_end))
# 				#f3a_set = set()
# 			else:
# 				print('error!', string, string_tag, string_start, string_end)
# 				sys.exit(1)

# 		else:
# 			index += 1

# 	MIN = 120
# 	subject2predicate = dict()
# 	for subject in f1_set:
# 		min_dis = MIN
# 		t_predicate = None
# 		for predicate in f2_set:
# 			dis = math.fabs(predicate[1]-subject[1])
# 			if dis < min_dis:
# 				min_dis = dis
# 				t_predicate = predicate
# 		subject2predicate[subject] = t_predicate
# 	# print subject2predicate

# 	object2predicate = dict()
# 	for _object in f3_set:
# 		min_dis = MIN
# 		t_predicate = None
# 		for predicate in f2_set:
# 			dis = math.fabs(predicate[1]-_object[1])
# 			if dis < min_dis:
# 				min_dis = dis
# 				t_predicate = predicate
# 		object2predicate[_object] = t_predicate
# 	# print object2predicate

# 	predicate2subject = dict()
# 	for predicate in f2_set:
# 		min_dis = MIN
# 		t_subject = None
# 		for subject in f1_set:
# 			dis = math.fabs(predicate[1]-subject[1])
# 			if dis < min_dis:
# 				min_dis = dis
# 				t_subject = subject
# 		predicate2subject[predicate] = t_subject
# 	# print predicate2subject

# 	predicate2object = dict()
# 	for predicate in f2_set:
# 		min_dis = MIN
# 		t_object = None
# 		for _object in f3_set:
# 			dis = math.fabs(predicate[1]-_object[1])
# 			if dis < min_dis:
# 				min_dis = dis
# 				t_object = _object
# 		predicate2object[predicate] = t_object
# 	# print predicate2object

# 	subject2object = dict()
# 	for subject in f1_set:
# 		min_dis = 10000
# 		t_object = None
# 		for _object in f3_set:
# 			dis = math.fabs(subject[1]-_object[1])
# 			if dis < min_dis:
# 				min_dis = dis
# 				t_object = _object
# 		subject2object[subject] = t_object
# 	# print subject2object

# 	object2subject = dict()
# 	for _object in f3_set:
# 		min_dis = 10000
# 		t_subject = None
# 		for subject in f1_set:
# 			dis = math.fabs(_object[1]-subject[1])
# 			if dis < min_dis:
# 				min_dis = dis
# 				t_subject = subject
# 		object2subject[_object] = t_subject
# 	# print object2subject

# 	attrib2subject = dict()
# 	for attrib in f1a_set:
# 		min_dis = 10
# 		t_subject = None
# 		for subject in f1_set:
# 			dis = math.fabs(attrib[1]-subject[1])
# 			if dis < min_dis:
# 				min_dis = dis
# 				t_subject = subject
# 		attrib2subject[attrib] = t_subject
# 	# print attrib2subject

# 	attrib2object = dict()
# 	for attrib in f3a_set:
# 		min_dis = 10
# 		t_object = None
# 		for _object in f3_set:
# 			dis = math.fabs(attrib[1]-_object[1])
# 			if dis < min_dis:
# 				min_dis = dis
# 				t_object = _object
# 		attrib2object[attrib] = t_object
# 	# print attrib2object

# 	sets = [f1_set, f2_set, f3_set]
# 	for _set in sets:
# 		_set.add('NIL')

# 	candidate_facts = itertools.product(f1_set, set(['NIL']), f2_set, f3_set, set(['NIL']))

# 	for fact in candidate_facts:
# 		# print 'candidate_facts:', fact
# 		fact = list(fact)
# 		subject = fact[0]
# 		predicate = fact[2]
# 		_object = fact[3]

# 		if subject == 'NIL' and _object == 'NIL':
# 			continue
# 		elif predicate == 'NIL':
# 			if subject == 'NIL' or _object == 'NIL':
# 				continue
# 			elif (subject2object[subject] != _object) and (object2subject[_object] != subject):
# 				continue
# 			elif (subject2predicate[subject] != None) or (object2predicate[_object] != None):
# 				continue
# 			else:
# 				facts.append(fact)
# 		else:
# 			if subject == 'NIL' and (predicate2subject[predicate]!=None or object2subject[_object]!=None):
# 				continue
# 			elif _object == 'NIL' and (predicate2object[predicate]!=None or subject2object[subject]!=None):
# 				continue
# 			else:
# 				if subject != 'NIL' and (subject2predicate[subject] != predicate) and (predicate2subject[predicate] != subject):
# 					#print '1'
# 					continue
# 				elif _object != 'NIL' and (object2predicate[_object] != predicate) and (predicate2object[predicate] != _object):
# 					#print '2'
# 					continue
# 				elif (subject != 'NIL' and _object != 'NIL') and (subject2object[subject] != _object) and (object2subject[_object] != subject):
# 					#print '3'
# 					continue
# 				else:
# 					facts.append(fact)
	
# 	extend_facts = []
# 	for attrib in f1a_set:
# 		subject = attrib2subject[attrib]
# 		for fact in facts:
# 			t_subject = fact[0]
# 			if t_subject != subject:
# 				continue
# 			if fact[1] == 'NIL':
# 				fact[1] = attrib
# 			elif fact[1] != attrib:
# 				new_fact = copy.deepcopy(fact)
# 				new_fact[1] = attrib
# 				extend_facts.append(new_fact)
# 			for _fact in facts:
# 				if _fact == fact:
# 					continue
# 				if _fact[2:] == fact[2:]:
# 					if _fact[1] == 'NIL':
# 						_fact[1] = attrib
# 					elif _fact[1] != attrib:
# 						new_fact = copy.deepcopy(_fact)
# 						new_fact[1] = attrib
# 						extend_facts.append(new_fact)
# 	facts.extend(extend_facts)

# 	extend_facts = []
# 	for attrib in f3a_set:
# 		_object = attrib2object[attrib]
# 		for fact in facts:
# 			t_object = fact[3]
# 			if t_object != _object:
# 				continue
# 			if fact[4] == 'NIL':
# 				fact[4] = attrib
# 			elif fact[4] != attrib:
# 				new_fact = copy.deepcopy(fact)
# 				new_fact[4] = attrib
# 				extend_facts.append(new_fact)
# 			for _fact in facts:
# 				if _fact == fact:
# 					continue
# 				if _fact[:2] == fact[:2]:
# 					if _fact[4] == 'NIL':
# 						_fact[4] = attrib
# 					elif _fact[4] != attrib:
# 						new_fact = copy.deepcopy(_fact)
# 						new_fact[4] = attrib
# 						extend_facts.append(new_fact)
# 	facts.extend(extend_facts)

# 	return facts

def revise_max(distribs, tags, threshold):
	assert len(distribs) == len(tags)
	for index in range(len(tags)):
		if distribs[index].item() > torch.log(torch.Tensor([threshold])).item():
			continue
		tags[index] = 0

def get_f1(precision, recall):
	if (precision == recall) and (precision == 0):
		return 0
	return 2 * (precision * recall) / (precision + recall)

def metric_to_list(metric):
	metric_list = []
	for tag in sorted(metric):
		tag2metric = [metric[tag].Precision()*100, metric[tag].Recall()*100, metric[tag].F_1()*100]
		metric_list.append(tag2metric)
	return metric_list

def tag2tag_to_list(tag2tag):
	tag2tag_list = []
	for tag in sorted(tag2tag):
		row = []
		for vtag in sorted(tag2tag[tag]):
			row.append(tag2tag[tag][vtag])
		tag2tag_list.append(row)
	return tag2tag_list

def evaluation(model, file_name, dataCenter, threshold_fact, threshold_cond, max_f1, max_std, out_model_name, num_pass, just_eval, weight_classes_fact, weight_classes_condition, write_prediction=False, file_name2=None, just_PR = False):

	valid_losses = []
	valid_tag_levels = []
	valid_tuple_levels = []
	valid_tag2metric_list = []
	valid_tag2tag_fact_list = []
	valid_tag2tag_cond_list = []

	test_losses = []
	test_tag_levels = []
	test_tuple_levels = []
	test_tag2metric_list = []
	test_tag2tag_fact_list = []
	test_tag2tag_cond_list = []

	for i in range(num_pass):
		VALID_DATA, TEST_DATA = dataCenter.get_evaluation(1.0/num_pass)
		loss, valid_tag_level, valid_Metrics, valid_tuple_level= _evaluation(model, dataCenter, VALID_DATA, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition, write_prediction, file_name2)
		valid_losses.append(loss)
		valid_tag_levels.append(valid_tag_level)
		valid_tuple_levels.append(valid_tuple_level)
		Tag2Metrics, tag2tag_fact, tag2tag_cond = valid_Metrics
		valid_tag2metric_list.append(metric_to_list(Tag2Metrics))
		valid_tag2tag_fact_list.append(tag2tag_to_list(tag2tag_fact))
		valid_tag2tag_cond_list.append(tag2tag_to_list(tag2tag_cond))

		if len(TEST_DATA[0]) == 0:
			continue

		loss, test_tag_level, test_Metrics, test_tuple_level= _evaluation(model, dataCenter, TEST_DATA, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition, write_prediction, file_name2)
		test_losses.append(loss)
		test_tag_levels.append(test_tag_level)
		test_tuple_levels.append(test_tuple_level)
		Tag2Metrics, tag2tag_fact, tag2tag_cond = test_Metrics
		test_tag2metric_list.append(metric_to_list(Tag2Metrics))
		test_tag2tag_fact_list.append(tag2tag_to_list(tag2tag_fact))
		test_tag2tag_cond_list.append(tag2tag_to_list(tag2tag_cond))

	valid_loss_mean = np.asarray(valid_losses).mean(0)
	valid_loss_std = np.asarray(valid_losses).std(0)
	valid_tag_levels_mean = np.asarray(valid_tag_levels).mean(0)
	valid_tag_levels_std = np.asarray(valid_tag_levels).std(0)

	valid_tag2metric_mean = np.asarray(valid_tag2metric_list).mean(0)
	valid_tag2metric_std = np.asarray(valid_tag2metric_list).std(0)
	valid_tag2tag_fact_mean = np.asarray(valid_tag2tag_fact_list).mean(0)
	valid_tag2tag_fact_std = np.asarray(valid_tag2tag_fact_list).std(0)
	valid_tag2tag_cond_mean = np.asarray(valid_tag2tag_cond_list).mean(0)
	valid_tag2tag_cond_std = np.asarray(valid_tag2tag_cond_list).std(0)

	valid_tuple_levels_mean = np.asarray(valid_tuple_levels).mean(0)
	valid_tuple_levels_std = np.asarray(valid_tuple_levels).std(0)

	if len(TEST_DATA[0]) != 0:
		test_loss_mean = np.asarray(test_losses).mean(0)
		test_loss_std = np.asarray(test_losses).std(0)
		test_tag_levels_mean = np.asarray(test_tag_levels).mean(0)
		test_tag_levels_std = np.asarray(test_tag_levels).std(0)

		test_tag2metric_mean = np.asarray(test_tag2metric_list).mean(0)
		test_tag2metric_std = np.asarray(test_tag2metric_list).std(0)
		test_tag2tag_fact_mean = np.asarray(test_tag2tag_fact_list).mean(0)
		test_tag2tag_fact_std = np.asarray(test_tag2tag_fact_list).std(0)
		test_tag2tag_cond_mean = np.asarray(test_tag2tag_cond_list).mean(0)
		test_tag2tag_cond_std = np.asarray(test_tag2tag_cond_list).std(0)

		test_tuple_levels_mean = np.asarray(test_tuple_levels).mean(0)
		test_tuple_levels_std = np.asarray(test_tuple_levels).std(0)
	
	# print valid_loss_mean, valid_loss_std, test_loss_mean, test_loss_std
	print valid_tag_levels_mean[-1][-1], valid_tag_levels_std[-1][-1]

	macro_F1 = valid_tag_levels_mean[-1][-1]
	macro_std = valid_tag_levels_std[-1][-1]

	# if (macro_F1-macro_std) > (max_f1-max_std) and (macro_F1+macro_std) > (max_f1+max_std):
	# if (macro_F1-macro_std) > (max_f1+max_std):
	# if valid_loss_mean < min_loss:
	if just_PR:
		return
	if macro_F1 > max_f1:
		max_f1 = macro_F1
		max_std = macro_std
		better = True
		print(max_f1, max_std)
		if not just_eval:
			print('saving model ...')
			torch.save(model.state_dict(), out_model_name)
			print('saving done.')

		fo = open(file_name, 'w')
		if len(TEST_DATA[0]) != 0:
			for i in range(len(test_tag_levels_mean)):
				for j in range(len(test_tag_levels_mean[i])):
					fo.write('%.2f+/-%.2f\t' % (test_tag_levels_mean[i][j], test_tag_levels_std[i][j]))
			for i in range(len(test_tuple_levels_mean)):
				for j in range(len(test_tuple_levels_mean[i])):
					fo.write('%.2f+/-%.2f\t' % (test_tuple_levels_mean[i][j], test_tuple_levels_std[i][j]))

		for i in range(len(valid_tag_levels_mean)):
			for j in range(len(valid_tag_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (valid_tag_levels_mean[i][j], valid_tag_levels_std[i][j]))
		for i in range(len(valid_tuple_levels_mean)):
			for j in range(len(valid_tuple_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (valid_tuple_levels_mean[i][j], valid_tuple_levels_std[i][j]))

		fo.write('\n')

		i = 0
		assert len(Tag2Metrics) == len(valid_tag2metric_mean)
		for tag in sorted(Tag2Metrics):
			assert len(valid_tag2metric_mean[i]) == 3
			# print '%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f' % (tag, valid_tag2metric_mean[i][0], valid_tag2metric_std[i][0], valid_tag2metric_mean[i][1], valid_tag2metric_std[i][1], valid_tag2metric_mean[i][2], valid_tag2metric_std[i][2])
			fo.write('%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f\n' % (tag, valid_tag2metric_mean[i][0], valid_tag2metric_std[i][0], valid_tag2metric_mean[i][1], valid_tag2metric_std[i][1], valid_tag2metric_mean[i][2], valid_tag2metric_std[i][2]))
			i += 1


		string = '-'
		for vtag in sorted(tag2tag_fact['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')

		i = 0
		assert len(tag2tag_fact) == len(valid_tag2tag_fact_mean) == len(valid_tag2tag_fact_std)
		for tag in sorted(tag2tag_fact):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_fact[tag]):
				mean = valid_tag2tag_fact_mean[i][j]
				std = valid_tag2tag_fact_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		string = '-'
		for vtag in sorted(tag2tag_cond['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')
		i = 0
		assert len(tag2tag_cond) == len(valid_tag2tag_cond_mean) == len(valid_tag2tag_cond_std)
		for tag in sorted(tag2tag_cond):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_cond[tag]):
				mean = valid_tag2tag_cond_mean[i][j]
				std = valid_tag2tag_cond_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		if len(TEST_DATA[0]) == 0:
			fo.close()
			return max_f1, max_std

		i = 0
		assert len(Tag2Metrics) == len(test_tag2metric_mean)
		for tag in sorted(Tag2Metrics):
			assert len(test_tag2metric_mean[i]) == 3
			# print '%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f' % (tag, test_tag2metric_mean[i][0], test_tag2metric_std[i][0], test_tag2metric_mean[i][1], test_tag2metric_std[i][1], test_tag2metric_mean[i][2], test_tag2metric_std[i][2])
			fo.write('%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f\n' % (tag, test_tag2metric_mean[i][0], test_tag2metric_std[i][0], test_tag2metric_mean[i][1], test_tag2metric_std[i][1], test_tag2metric_mean[i][2], test_tag2metric_std[i][2]))
			i += 1


		string = '-'
		for vtag in sorted(tag2tag_fact['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')

		i = 0
		assert len(tag2tag_fact) == len(test_tag2tag_fact_mean) == len(test_tag2tag_fact_std)
		for tag in sorted(tag2tag_fact):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_fact[tag]):
				mean = test_tag2tag_fact_mean[i][j]
				std = test_tag2tag_fact_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		string = '-'
		for vtag in sorted(tag2tag_cond['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')
		i = 0
		assert len(tag2tag_cond) == len(test_tag2tag_cond_mean) == len(test_tag2tag_cond_std)
		for tag in sorted(tag2tag_cond):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_cond[tag]):
				mean = test_tag2tag_cond_mean[i][j]
				std = test_tag2tag_cond_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		fo.close()

	return max_f1, max_std

def evaluation_ensemble(models, file_name, dataCenter, threshold_fact, threshold_cond, max_f1, max_std, out_model_name, num_pass, just_eval, weight_classes_fact, weight_classes_condition, ensemble_model, device, is_two=False, is_filter=False, just_PR=False, write_prediction=False, out_file=None):

	valid_losses = []
	valid_tag_levels = []
	valid_tuple_levels = []
	valid_tag2metric_list = []
	valid_tag2tag_fact_list = []
	valid_tag2tag_cond_list = []

	test_losses = []
	test_tag_levels = []
	test_tuple_levels = []
	test_tag2metric_list = []
	test_tag2tag_fact_list = []
	test_tag2tag_cond_list = []

	for i in range(num_pass):
		VALID_DATA, TEST_DATA = dataCenter.get_evaluation(1.0/num_pass)
		if is_two:
			loss, valid_tag_level, valid_Metrics, valid_tuple_level = _evaluation_two_ensemble(models, dataCenter, VALID_DATA, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition, ensemble_model, device)
		else:
			loss, valid_tag_level, valid_Metrics, valid_tuple_level = _evaluation_ensemble(models, dataCenter, VALID_DATA, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition, ensemble_model, device, is_filter, write_prediction, out_file)
		valid_losses.append(loss)
		valid_tag_levels.append(valid_tag_level)
		valid_tuple_levels.append(valid_tuple_level)
		Tag2Metrics, tag2tag_fact, tag2tag_cond = valid_Metrics
		valid_tag2metric_list.append(metric_to_list(Tag2Metrics))
		valid_tag2tag_fact_list.append(tag2tag_to_list(tag2tag_fact))
		valid_tag2tag_cond_list.append(tag2tag_to_list(tag2tag_cond))

		if len(TEST_DATA[0]) == 0:
			continue

		if is_two:
			loss, test_tag_level, test_Metrics, test_tuple_level= _evaluation_two_ensemble(models, dataCenter, TEST_DATA, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition, ensemble_model, device)
		else:
			loss, test_tag_level, test_Metrics, test_tuple_level= _evaluation_ensemble(models, dataCenter, TEST_DATA, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition, ensemble_model, device, is_filter, write_prediction, out_file)
		test_losses.append(loss)
		test_tag_levels.append(test_tag_level)
		test_tuple_levels.append(test_tuple_level)
		Tag2Metrics, tag2tag_fact, tag2tag_cond = test_Metrics
		test_tag2metric_list.append(metric_to_list(Tag2Metrics))
		test_tag2tag_fact_list.append(tag2tag_to_list(tag2tag_fact))
		test_tag2tag_cond_list.append(tag2tag_to_list(tag2tag_cond))

	valid_loss_mean = np.asarray(valid_losses).mean(0)
	valid_loss_std = np.asarray(valid_losses).std(0)
	valid_tag_levels_mean = np.asarray(valid_tag_levels).mean(0)
	valid_tag_levels_std = np.asarray(valid_tag_levels).std(0)

	valid_tag2metric_mean = np.asarray(valid_tag2metric_list).mean(0)
	valid_tag2metric_std = np.asarray(valid_tag2metric_list).std(0)
	valid_tag2tag_fact_mean = np.asarray(valid_tag2tag_fact_list).mean(0)
	valid_tag2tag_fact_std = np.asarray(valid_tag2tag_fact_list).std(0)
	valid_tag2tag_cond_mean = np.asarray(valid_tag2tag_cond_list).mean(0)
	valid_tag2tag_cond_std = np.asarray(valid_tag2tag_cond_list).std(0)

	valid_tuple_levels_mean = np.asarray(valid_tuple_levels).mean(0)
	valid_tuple_levels_std = np.asarray(valid_tuple_levels).std(0)

	if len(TEST_DATA[0]) != 0:
		test_loss_mean = np.asarray(test_losses).mean(0)
		test_loss_std = np.asarray(test_losses).std(0)
		test_tag_levels_mean = np.asarray(test_tag_levels).mean(0)
		test_tag_levels_std = np.asarray(test_tag_levels).std(0)

		test_tag2metric_mean = np.asarray(test_tag2metric_list).mean(0)
		test_tag2metric_std = np.asarray(test_tag2metric_list).std(0)
		test_tag2tag_fact_mean = np.asarray(test_tag2tag_fact_list).mean(0)
		test_tag2tag_fact_std = np.asarray(test_tag2tag_fact_list).std(0)
		test_tag2tag_cond_mean = np.asarray(test_tag2tag_cond_list).mean(0)
		test_tag2tag_cond_std = np.asarray(test_tag2tag_cond_list).std(0)

		test_tuple_levels_mean = np.asarray(test_tuple_levels).mean(0)
		test_tuple_levels_std = np.asarray(test_tuple_levels).std(0)
	
	# print valid_loss_mean, valid_loss_std, test_loss_mean, test_loss_std
	

	micro_F1 = valid_tag_levels_mean[-1][2]
	micro_std = valid_tag_levels_std[-1][2]
	macro_F1 = valid_tag_levels_mean[-1][-1]
	macro_std = valid_tag_levels_std[-1][-1]

	print macro_F1, macro_std

	# if (macro_F1-macro_std) > (max_f1-max_std) and (macro_F1+macro_std) > (max_f1+max_std):
	# if (macro_F1-macro_std) > (max_f1+max_std):
	# if valid_loss_mean < min_loss:
	if just_PR:
		return
	if macro_F1 > max_f1:
		max_f1 = macro_F1
		max_std = macro_std
		better = True
		print(max_f1, max_std)
		if not just_eval:
			print('saving model ...')
			torch.save(ensemble_model.state_dict(), out_model_name)
			print('saving done.')

		fo = open(file_name, 'w')
		for i in range(len(test_tag_levels_mean)):
			for j in range(len(test_tag_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (test_tag_levels_mean[i][j], test_tag_levels_std[i][j]))
		for i in range(len(test_tuple_levels_mean)):
			for j in range(len(test_tuple_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (test_tuple_levels_mean[i][j], test_tuple_levels_std[i][j]))

		for i in range(len(valid_tag_levels_mean)):
			for j in range(len(valid_tag_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (valid_tag_levels_mean[i][j], valid_tag_levels_std[i][j]))
		for i in range(len(valid_tuple_levels_mean)):
			for j in range(len(valid_tuple_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (valid_tuple_levels_mean[i][j], valid_tuple_levels_std[i][j]))

		fo.write('\n')

		i = 0
		assert len(Tag2Metrics) == len(valid_tag2metric_mean)
		for tag in sorted(Tag2Metrics):
			assert len(valid_tag2metric_mean[i]) == 3
			# print '%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f' % (tag, valid_tag2metric_mean[i][0], valid_tag2metric_std[i][0], valid_tag2metric_mean[i][1], valid_tag2metric_std[i][1], valid_tag2metric_mean[i][2], valid_tag2metric_std[i][2])
			fo.write('%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f\n' % (tag, valid_tag2metric_mean[i][0], valid_tag2metric_std[i][0], valid_tag2metric_mean[i][1], valid_tag2metric_std[i][1], valid_tag2metric_mean[i][2], valid_tag2metric_std[i][2]))
			i += 1


		string = '-'
		for vtag in sorted(tag2tag_fact['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')

		i = 0
		assert len(tag2tag_fact) == len(valid_tag2tag_fact_mean) == len(valid_tag2tag_fact_std)
		for tag in sorted(tag2tag_fact):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_fact[tag]):
				mean = valid_tag2tag_fact_mean[i][j]
				std = valid_tag2tag_fact_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		string = '-'
		for vtag in sorted(tag2tag_cond['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')
		i = 0
		assert len(tag2tag_cond) == len(valid_tag2tag_cond_mean) == len(valid_tag2tag_cond_std)
		for tag in sorted(tag2tag_cond):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_cond[tag]):
				mean = valid_tag2tag_cond_mean[i][j]
				std = valid_tag2tag_cond_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		i = 0
		assert len(Tag2Metrics) == len(test_tag2metric_mean)
		for tag in sorted(Tag2Metrics):
			assert len(test_tag2metric_mean[i]) == 3
			# print '%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f' % (tag, test_tag2metric_mean[i][0], test_tag2metric_std[i][0], test_tag2metric_mean[i][1], test_tag2metric_std[i][1], test_tag2metric_mean[i][2], test_tag2metric_std[i][2])
			fo.write('%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f\n' % (tag, test_tag2metric_mean[i][0], test_tag2metric_std[i][0], test_tag2metric_mean[i][1], test_tag2metric_std[i][1], test_tag2metric_mean[i][2], test_tag2metric_std[i][2]))
			i += 1


		string = '-'
		for vtag in sorted(tag2tag_fact['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')

		i = 0
		assert len(tag2tag_fact) == len(test_tag2tag_fact_mean) == len(test_tag2tag_fact_std)
		for tag in sorted(tag2tag_fact):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_fact[tag]):
				mean = test_tag2tag_fact_mean[i][j]
				std = test_tag2tag_fact_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		string = '-'
		for vtag in sorted(tag2tag_cond['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')
		i = 0
		assert len(tag2tag_cond) == len(test_tag2tag_cond_mean) == len(test_tag2tag_cond_std)
		for tag in sorted(tag2tag_cond):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_cond[tag]):
				mean = test_tag2tag_cond_mean[i][j]
				std = test_tag2tag_cond_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		fo.close()

	return max_f1, max_std

def single_evaluation(fact_model, cond_model, file_name, dataCenter, threshold_fact, threshold_cond, max_f1, out_model_name, num_pass):
	valid_tag_levels = []
	valid_tuple_levels = []
	valid_tag2metric_list = []
	valid_tag2tag_fact_list = []
	valid_tag2tag_cond_list = []

	test_tag_levels = []
	test_tuple_levels = []
	test_tag2metric_list = []
	test_tag2tag_fact_list = []
	test_tag2tag_cond_list = []

	for i in range(num_pass):
		VALID_DATA, TEST_DATA = dataCenter.get_evaluation(1.0/num_pass)
		loss, valid_tag_level, valid_Metrics, valid_tuple_level= _single_evaluation(fact_model, cond_model, dataCenter, VALID_DATA, threshold_fact, threshold_cond)
		valid_tag_levels.append(valid_tag_level)
		valid_tuple_levels.append(valid_tuple_level)
		Tag2Metrics, tag2tag_fact, tag2tag_cond = valid_Metrics
		valid_tag2metric_list.append(metric_to_list(Tag2Metrics))
		valid_tag2tag_fact_list.append(tag2tag_to_list(tag2tag_fact))
		valid_tag2tag_cond_list.append(tag2tag_to_list(tag2tag_cond))

		loss, test_tag_level, test_Metrics, test_tuple_level= _single_evaluation(fact_model, cond_model, dataCenter, TEST_DATA, threshold_fact, threshold_cond)
		test_tag_levels.append(test_tag_level)
		test_tuple_levels.append(test_tuple_level)
		Tag2Metrics, tag2tag_fact, tag2tag_cond = test_Metrics
		test_tag2metric_list.append(metric_to_list(Tag2Metrics))
		test_tag2tag_fact_list.append(tag2tag_to_list(tag2tag_fact))
		test_tag2tag_cond_list.append(tag2tag_to_list(tag2tag_cond))

	valid_tag_levels_mean = np.asarray(valid_tag_levels).mean(0)
	valid_tag_levels_std = np.asarray(valid_tag_levels).std(0)

	valid_tag2metric_mean = np.asarray(valid_tag2metric_list).mean(0)
	valid_tag2metric_std = np.asarray(valid_tag2metric_list).std(0)
	valid_tag2tag_fact_mean = np.asarray(valid_tag2tag_fact_list).mean(0)
	valid_tag2tag_fact_std = np.asarray(valid_tag2tag_fact_list).std(0)
	valid_tag2tag_cond_mean = np.asarray(valid_tag2tag_cond_list).mean(0)
	valid_tag2tag_cond_std = np.asarray(valid_tag2tag_cond_list).std(0)

	valid_tuple_levels_mean = np.asarray(valid_tuple_levels).mean(0)
	valid_tuple_levels_std = np.asarray(valid_tuple_levels).std(0)

	test_tag_levels_mean = np.asarray(test_tag_levels).mean(0)
	test_tag_levels_std = np.asarray(test_tag_levels).std(0)

	test_tag2metric_mean = np.asarray(test_tag2metric_list).mean(0)
	test_tag2metric_std = np.asarray(test_tag2metric_list).std(0)
	test_tag2tag_fact_mean = np.asarray(test_tag2tag_fact_list).mean(0)
	test_tag2tag_fact_std = np.asarray(test_tag2tag_fact_list).std(0)
	test_tag2tag_cond_mean = np.asarray(test_tag2tag_cond_list).mean(0)
	test_tag2tag_cond_std = np.asarray(test_tag2tag_cond_list).std(0)

	test_tuple_levels_mean = np.asarray(test_tuple_levels).mean(0)
	test_tuple_levels_std = np.asarray(test_tuple_levels).std(0)

	# print valid_tag_levels_mean, valid_tag_levels_std, valid_tuple_levels_mean, valid_tuple_levels_std
	# print test_tag_levels_mean, test_tag_levels_std, test_tuple_levels_mean, test_tuple_levels_std
	macro_F1 = valid_tag_levels_mean[-1][-1]
	
	if macro_F1 > max_f1:
		max_f1 = macro_F1
		print(max_f1)
		print('saving model ...')
		torch.save(fact_model.state_dict(), out_model_name+'_fact.torch')
		torch.save(cond_model.state_dict(), out_model_name+'_cond.torch')
		print('saving done.')

		fo = open(file_name, 'w')
		for i in range(len(test_tag_levels_mean)):
			for j in range(len(test_tag_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (test_tag_levels_mean[i][j], test_tag_levels_std[i][j]))
		for i in range(len(test_tuple_levels_mean)):
			for j in range(len(test_tuple_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (test_tuple_levels_mean[i][j], test_tuple_levels_std[i][j]))

		for i in range(len(valid_tag_levels_mean)):
			for j in range(len(valid_tag_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (valid_tag_levels_mean[i][j], valid_tag_levels_std[i][j]))
		for i in range(len(valid_tuple_levels_mean)):
			for j in range(len(valid_tuple_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (valid_tuple_levels_mean[i][j], valid_tuple_levels_std[i][j]))

		fo.write('\n')

		i = 0
		assert len(Tag2Metrics) == len(valid_tag2metric_mean)
		for tag in sorted(Tag2Metrics):
			assert len(valid_tag2metric_mean[i]) == 3
			# print '%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f' % (tag, valid_tag2metric_mean[i][0], valid_tag2metric_std[i][0], valid_tag2metric_mean[i][1], valid_tag2metric_std[i][1], valid_tag2metric_mean[i][2], valid_tag2metric_std[i][2])
			fo.write('%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f\n' % (tag, valid_tag2metric_mean[i][0], valid_tag2metric_std[i][0], valid_tag2metric_mean[i][1], valid_tag2metric_std[i][1], valid_tag2metric_mean[i][2], valid_tag2metric_std[i][2]))
			i += 1


		string = '-'
		for vtag in sorted(tag2tag_fact['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')

		i = 0
		assert len(tag2tag_fact) == len(valid_tag2tag_fact_mean) == len(valid_tag2tag_fact_std)
		for tag in sorted(tag2tag_fact):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_fact[tag]):
				mean = valid_tag2tag_fact_mean[i][j]
				std = valid_tag2tag_fact_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		string = '-'
		for vtag in sorted(tag2tag_cond['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')
		i = 0
		assert len(tag2tag_cond) == len(valid_tag2tag_cond_mean) == len(valid_tag2tag_cond_std)
		for tag in sorted(tag2tag_cond):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_cond[tag]):
				mean = valid_tag2tag_cond_mean[i][j]
				std = valid_tag2tag_cond_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		i = 0
		assert len(Tag2Metrics) == len(test_tag2metric_mean)
		for tag in sorted(Tag2Metrics):
			assert len(test_tag2metric_mean[i]) == 3
			# print '%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f' % (tag, test_tag2metric_mean[i][0], test_tag2metric_std[i][0], test_tag2metric_mean[i][1], test_tag2metric_std[i][1], test_tag2metric_mean[i][2], test_tag2metric_std[i][2])
			fo.write('%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f\n' % (tag, test_tag2metric_mean[i][0], test_tag2metric_std[i][0], test_tag2metric_mean[i][1], test_tag2metric_std[i][1], test_tag2metric_mean[i][2], test_tag2metric_std[i][2]))
			i += 1


		string = '-'
		for vtag in sorted(tag2tag_fact['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')

		i = 0
		assert len(tag2tag_fact) == len(test_tag2tag_fact_mean) == len(test_tag2tag_fact_std)
		for tag in sorted(tag2tag_fact):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_fact[tag]):
				mean = test_tag2tag_fact_mean[i][j]
				std = test_tag2tag_fact_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		string = '-'
		for vtag in sorted(tag2tag_cond['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')
		i = 0
		assert len(tag2tag_cond) == len(test_tag2tag_cond_mean) == len(test_tag2tag_cond_std)
		for tag in sorted(tag2tag_cond):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_cond[tag]):
				mean = test_tag2tag_cond_mean[i][j]
				std = test_tag2tag_cond_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		fo.close()

	return max_f1		

def _core_evaluation(OUTs, SENTENCEs, instance_list, predict_fact_batch, predict_condition_batch, dataCenter, threshold_fact, threshold_cond, write_prediction=False, out_file=None, is_filter=False):
	assert len(OUTs) == len(predict_fact_batch) == len(predict_condition_batch)
	loss = 0

	# for i in range(len(OUTs)):
	# 	for j in range(len(OUTs[i][0])):
	# 		tagID = OUTs[i][0][j]
	# 		_loss = (-weight_classes_fact[tagID].item() * predict_fact_batch[i][j][tagID].item())
	# 		loss += _loss
	# 	for j in range(len(OUTs[i][1])):
	# 		tagID = OUTs[i][1][j]
	# 		_loss = (-weight_classes_condition[tagID].item() * predict_condition_batch[i][j][tagID].item())
	# 		loss += _loss

	# loss /= len(OUTs)

	Tag2Metrics = dict()
	Tag2Metrics_fact = dict()
	Tag2Metrics_condition = dict()

	for tag in dataCenter.Tag2ID_fact:
		Tag2Metrics[tag] = Metrics()
		Tag2Metrics_fact[tag] = Metrics()
	for tag in dataCenter.Tag2ID_condition:
		Tag2Metrics[tag] = Metrics()
		Tag2Metrics_condition[tag] = Metrics()

	tag2tag_fact = dict()
	tag2tag_cond = dict()

	precision_sum_f = 0
	recall_sum_f = 0
	precision_sum_c = 0
	recall_sum_c = 0

	precisions_f = []
	recalls_f = []
	precisions_c = []
	recalls_c = []

	for tag in dataCenter.Tag2ID_fact:
		if tag not in tag2tag_fact:
			tag2tag_fact[tag] = dict()
			for _tag in dataCenter.Tag2ID_fact:
				tag2tag_fact[tag][_tag] = 0

	for tag in dataCenter.Tag2ID_condition:
		if tag not in tag2tag_cond:
			tag2tag_cond[tag] = dict()
			for _tag in dataCenter.Tag2ID_condition:
				tag2tag_cond[tag][_tag] = 0
	if write_prediction:
		fo = open(out_file, 'w')
		should_miss = []
		but_miss = []
		total_cap = []

		Tag2Metrics_concept = Metrics()
		Tag2Metrics_attribute = Metrics()
		Tag2Metrics_predicate = Metrics()

	filter_nu = 0
	for i in range(len(predict_fact_batch)):
		predicted_fact_tag_distribs, predicted_fact_tags = torch.max(predict_fact_batch[i], 1)
		predicted_conditions_tag_distribs, predicted_conditions_tags = torch.max(predict_condition_batch[i], 1)
		fact_tags = [dataCenter.ID2Tag_fact[tag_id.item()] for tag_id in predicted_fact_tags]
		cond_tags = [dataCenter.ID2Tag_condition[tag_id.item()] for tag_id in predicted_conditions_tags]

		if is_filter:
			is_discarded_fact, fact_predicate_set = is_discarded(fact_tags)
			is_discarded_cond, cond_predicate_set = is_discarded(cond_tags)

			if is_discarded_fact or is_discarded_cond:
				filter_nu += 1
				continue
			if fact_predicate_set & cond_predicate_set != set():
				filter_nu += 1
				continue

			fact_tags, corrected_fact = smooth_tag_sequence(fact_tags)
			cond_tags, corrected_cond = smooth_tag_sequence(cond_tags)
			if corrected_fact or corrected_cond:
				filter_nu += 1
				continue
			
		for j in range(len(OUTs[i][0])):
			y_true = dataCenter.ID2Tag_fact[OUTs[i][0][j]]
			# y_predict = fact_tags[j]
			if predicted_fact_tag_distribs[j].item() >= torch.log(torch.Tensor([threshold_fact])).item():
			 	y_predict = fact_tags[j]
			else:
			 	y_predict = 'O'
			 	fact_tags[j] = 'O'


			if y_true == y_predict:
				tag = y_predict
				Tag2Metrics[tag].TP += 1
				Tag2Metrics_fact[tag].TP += 1
				tag2tag_fact[tag][tag] += 1

			elif y_true != y_predict:
				tag_true = y_true
				tag_predict = y_predict
				Tag2Metrics[tag_true].FN += 1
				Tag2Metrics[tag_predict].FP += 1
				Tag2Metrics_fact[tag_true].FN += 1
				Tag2Metrics_fact[tag_predict].FP += 1
				tag2tag_fact[tag_true][tag_predict] += 1
		for j in range(len(OUTs[i][1])):
			y_true = dataCenter.ID2Tag_condition[OUTs[i][1][j]]
			# y_predict = cond_tags[j]
			if predicted_conditions_tag_distribs[j].item() >= torch.log(torch.Tensor([threshold_cond])).item():
			 	y_predict = cond_tags[j]
			else:
			 	y_predict = 'O'
			 	cond_tags[j] = 'O'

			if y_true == y_predict:
				tag = y_predict
				Tag2Metrics[tag].TP += 1
				Tag2Metrics_condition[tag].TP += 1
				tag2tag_cond[tag][tag] += 1

			elif y_true != y_predict:
				tag_true = y_true
				tag_predict = y_predict
				Tag2Metrics[tag_true].FN += 1
				Tag2Metrics[tag_predict].FP += 1
				Tag2Metrics_condition[tag_true].FN += 1
				Tag2Metrics_condition[tag_predict].FP += 1
				tag2tag_cond[tag_true][tag_predict] += 1

		instance = instance_list[i]
		assert len(instance.SENTENCE) == len(SENTENCEs[i])
		_facts = []
		_conditions = []
		for out in instance.multi_output:
			if out[0].startswith('f'):
				__facts = post_decoder(instance.multi_input[0][1], out[1], dataCenter.ID2Tag_fact)
				_facts.extend(__facts)
			else:
				__conditions = post_decoder(instance.multi_input[0][1], out[1], dataCenter.ID2Tag_condition)
				_conditions.extend(__conditions)

		# predicted_fact_tags = predicted_fact_tags.to(torch.device("cpu")).detach().numpy().tolist()
		# predicted_conditions_tags = predicted_conditions_tags.to(torch.device("cpu")).detach().numpy().tolist()
		# revise_max(predicted_fact_tag_distribs, predicted_fact_tags, threshold_fact)
		# revise_max(predicted_conditions_tag_distribs, predicted_conditions_tags, threshold_cond)

		facts = post_decoder(instance.multi_input[0][1], fact_tags, dataCenter.ID2Tag_fact)
		conditions = post_decoder(instance.multi_input[0][1], cond_tags, dataCenter.ID2Tag_condition)

		p_f, r_f, ps_f, rs_f = match_score(_facts, facts)
		p_c, r_c, ps_c, rs_c = match_score(_conditions, conditions)

		precision_sum_f += p_f
		recall_sum_f += r_f
		precision_sum_c += p_c
		recall_sum_c += r_c

		precisions_f.extend(ps_f)
		recalls_f.extend(rs_f)
		precisions_c.extend(ps_c)
		recalls_c.extend(rs_c)

		if write_prediction:
			fo.write('===== %s stmt%s =====\n' % (str(instance_list[i].paper_id), str(instance_list[i].stmt_id)))
			fo.write('WORD\t'+'\t'.join(instance_list[i].SENTENCE)+'\n')
			fo.write('POSTAG\t'+'\t'.join(instance_list[i].POSTAG)+'\n')
			fo.write('CAP\t'+'\t'.join(instance_list[i].CAP)+'\n')
			# fact_tags = [dataCenter.ID2Tag_fact[tag_id] for tag_id in predicted_fact_tags]
			fo.write('f\t'+'\t'.join(fact_tags)+'\n')
			# cond_tags = [dataCenter.ID2Tag_condition[tag_id] for tag_id in predicted_conditions_tags]
			fo.write('c\t'+'\t'.join(cond_tags)+'\n')
			fact_g_tags = [dataCenter.ID2Tag_fact[tag_id] for tag_id in instance_list[i].OUT[0]]
			fo.write('f_g\t'+'\t'.join(fact_g_tags)+'\n')
			cond_g_tags = [dataCenter.ID2Tag_condition[tag_id] for tag_id in instance_list[i].OUT[1]]
			fo.write('c_g\t'+'\t'.join(cond_g_tags)+'\n')

			for j in range(len(fact_g_tags)):
				cap = instance_list[i].CAP[j]
				f = fact_tags[j]
				c = cond_tags[j]
				f_g = fact_g_tags[j]
				c_g = cond_g_tags[j]
				if cap != 'O':
					total_cap.append(cap)
				if cap != 'O' and (f == 'O' and c == 'O'):
					if (f_g != 'O' and 'P' not in f_g) or (c_g != 'O' and 'P' not in c_g):
						but_miss.append(cap)
				if cap != 'O' and ((f_g == 'O' or 'P' in f_g) and (c_g == 'O' or 'P' in c_g)):
					should_miss.append(cap)

				m_tag = f+c
				m_tag_g = f_g+c_g

				if 'C' in m_tag and 'C' in m_tag_g:
					Tag2Metrics_concept.TP += 1
				elif 'C' not in m_tag and 'C' in m_tag_g:
					Tag2Metrics_concept.FN += 1
				elif 'C' in m_tag and 'C' not in m_tag_g:
					Tag2Metrics_concept.FP += 1

				if 'A' in m_tag and 'A' in m_tag_g:
					Tag2Metrics_attribute.TP += 1
				elif 'A' not in m_tag and 'A' in m_tag_g:
					Tag2Metrics_attribute.FN += 1
				elif 'A' in m_tag and 'A' not in m_tag_g:
					Tag2Metrics_attribute.FP += 1

				if 'P' in m_tag and 'P' in m_tag_g:
					Tag2Metrics_predicate.TP += 1
				elif 'P' not in m_tag and 'P' in m_tag_g:
					Tag2Metrics_predicate.FN += 1
				elif 'P' in m_tag and 'P' not in m_tag_g:
					Tag2Metrics_predicate.FP += 1

	
	if write_prediction:
		cap_set = set(total_cap)
		fo.write('total:\n')
		for cap in cap_set:
			fo.write('%s\t%d\n' % (cap, total_cap.count(cap)))

		fo.write('should_miss:\n')
		for cap in cap_set:
			fo.write('%s\t%d\n' % (cap, should_miss.count(cap)))

		fo.write('but_miss:\n')
		for cap in cap_set:
			fo.write('%s\t%d\n' % (cap, but_miss.count(cap)))

		fo.write('%d\t%d\t%d\t%.2f\n' % (len(but_miss), len(should_miss), len(total_cap), float(len(total_cap) - len(should_miss) - len(but_miss))/(len(total_cap) - len(should_miss))*100))

		fo.write('concept\t%.2f\t%.2f\t%.2f\n' % (Tag2Metrics_concept.Precision()*100, Tag2Metrics_concept.Recall()*100, Tag2Metrics_concept.F_1()*100))
		fo.write('attribute\t%.2f\t%.2f\t%.2f\n' % (Tag2Metrics_attribute.Precision()*100, Tag2Metrics_attribute.Recall()*100, Tag2Metrics_attribute.F_1()*100))
		fo.write('predicate\t%.2f\t%.2f\t%.2f\n' % (Tag2Metrics_predicate.Precision()*100, Tag2Metrics_predicate.Recall()*100, Tag2Metrics_predicate.F_1()*100))
		fo.close()

	microEval = Metrics()
	microEval_fact = Metrics()
	microEval_condition = Metrics()
	macro_P = 0
	macro_R = 0
	macro_P_fact = 0
	macro_R_fact = 0
	macro_P_condition = 0
	macro_R_condition = 0
	# print threshold_fact, threshold_cond
	for tag in Tag2Metrics:
		if tag == 'O':
			continue

		macro_P += Tag2Metrics[tag].Precision()*100
		macro_R += Tag2Metrics[tag].Recall()*100
		microEval.TP += Tag2Metrics[tag].TP
		microEval.FP += Tag2Metrics[tag].FP
		microEval.FN += Tag2Metrics[tag].FN

		if tag in dataCenter.Tag2ID_fact:
			macro_P_fact += Tag2Metrics_fact[tag].Precision()*100
			macro_R_fact += Tag2Metrics_fact[tag].Recall()*100
			microEval_fact.TP += Tag2Metrics_fact[tag].TP
			microEval_fact.FP += Tag2Metrics_fact[tag].FP
			microEval_fact.FN += Tag2Metrics_fact[tag].FN
		elif tag in dataCenter.Tag2ID_condition:
			macro_P_condition += Tag2Metrics_condition[tag].Precision()*100
			macro_R_condition += Tag2Metrics_condition[tag].Recall()*100
			microEval_condition.TP += Tag2Metrics_condition[tag].TP
			microEval_condition.FP += Tag2Metrics_condition[tag].FP
			microEval_condition.FN += Tag2Metrics_condition[tag].FN
		else:
			print('error')
			sys.exit(1)

	macro_P /= (len(Tag2Metrics) - 1)
	macro_R /= (len(Tag2Metrics) - 1)
	macro_P_fact /= (len(Tag2Metrics_fact) - 1)
	macro_R_fact /= (len(Tag2Metrics_fact) - 1)
	macro_P_condition /= (len(Tag2Metrics_condition) - 1)
	macro_R_condition /= (len(Tag2Metrics_condition) - 1)

	if (macro_P == macro_R == 0):
		macro_F1 = 0
	else:
		macro_F1 = 2 * (macro_P * macro_R) / (macro_P + macro_R)

	if (macro_P_fact == macro_R_fact == 0):
		macro_F1_fact = 0
	else:
		macro_F1_fact = 2 * (macro_P_fact * macro_R_fact) / (macro_P_fact + macro_R_fact)

	if (macro_P_condition == macro_R_condition == 0):
		macro_F1_condition = 0
	else:
		macro_F1_condition = 2 * (macro_P_condition * macro_R_condition) / (macro_P_condition + macro_R_condition)

	tag_level_fact = [microEval_fact.Precision()*100, microEval_fact.Recall()*100, microEval_fact.F_1()*100, macro_P_fact, macro_R_fact, macro_F1_fact]
	tag_level_cond = [microEval_condition.Precision()*100, microEval_condition.Recall()*100, microEval_condition.F_1()*100, macro_P_condition, macro_R_condition, macro_F1_condition]
	tag_level = [microEval.Precision()*100, microEval.Recall()*100, microEval.F_1()*100, macro_P, macro_R, macro_F1]

	precisions_f = np.asarray(precisions_f)
	precisions_c = np.asarray(precisions_c)
	recalls_f = np.asarray(recalls_f)
	recalls_c = np.asarray(recalls_c)

	micro_precision_f = precisions_f.sum()/float(len(precisions_f)*5)
	micro_recall_f = recalls_f.sum()/float(len(recalls_f)*5)
	micro_f1_f = get_f1(micro_precision_f, micro_recall_f)

	micro_precision_c = precisions_c.sum()/float(len(precisions_c)*5)
	micro_recall_c = recalls_c.sum()/float(len(recalls_c)*5)
	micro_f1_c = get_f1(micro_precision_c, micro_recall_c)
	macro_precision_f = precision_sum_f/(len(OUTs)-filter_nu)
	macro_recall_f = recall_sum_f/(len(OUTs)-filter_nu)
	macro_f1_f = get_f1(macro_precision_f, macro_recall_f)

	macro_precision_c = precision_sum_c/(len(OUTs)-filter_nu)
	macro_recall_c = recall_sum_c/(len(OUTs)-filter_nu)
	macro_f1_c = get_f1(macro_precision_c, macro_recall_c)

	precisions = np.concatenate((precisions_f, precisions_c))
	recalls = np.concatenate((recalls_f, recalls_c))
	micro_precision = precisions.sum()/float(len(precisions)*5)
	micro_recall = recalls.sum()/float(len(recalls)*5)
	micro_f1 = get_f1(micro_precision, micro_recall)

	macro_precision = (macro_precision_f+macro_precision_c)/2
	macro_recall = (macro_recall_f+macro_recall_c)/2
	macro_f1 = get_f1(macro_precision, macro_recall)

	tuple_level_fact = [micro_precision_f*100, micro_recall_f*100, micro_f1_f*100, macro_precision_f*100, macro_recall_f*100, macro_f1_f*100]
	tuple_level_cond = [micro_precision_c*100, micro_recall_c*100, micro_f1_c*100, macro_precision_c*100, macro_recall_c*100, macro_f1_c*100]
	tuple_level = [micro_precision*100, micro_recall*100, micro_f1*100, macro_precision*100, macro_recall*100, macro_f1*100]

	return loss, [tag_level_fact, tag_level_cond, tag_level], [Tag2Metrics, tag2tag_fact, tag2tag_cond], [tuple_level_fact, tuple_level_cond, tuple_level]

def _evaluation_two_ensemble(models, dataCenter, data, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition, ensemble_model, device):

	raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_POSCAPs, raw_LM_SENTENCEs, raw_OUTs, raw_instance_list = data
	#SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	MICS = zip(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs, raw_instance_list)
	MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs, instance_list = zip(*MICS)
	SENTENCEs = list(SENTENCEs)
	POSTAGs = list(POSTAGs)
	CAPs = list(CAPs)
	POSCAPs = list(POSCAPs)
	LM_SENTENCEs = list(LM_SENTENCEs)
	OUTs = list(OUTs)
	instance_list = list(instance_list)

	lm_input = single_model_predict(models[0], (SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs), len(SENTENCEs))
	pos_input = single_model_predict(models[1], (SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs), len(SENTENCEs))
	cap_input = single_model_predict(models[2], (SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs), len(SENTENCEs))

	first_input_batch = ensemble_model_predict(models[3], (lm_input, pos_input, cap_input))
	second_input_batch = ensemble_model_predict(models[4], (lm_input, pos_input, cap_input))

	predict_fact_batch, predict_condition_batch = ensemble_model((first_input_batch, second_input_batch))

	return _core_evaluation(OUTs, SENTENCEs, instance_list, predict_fact_batch, predict_condition_batch, dataCenter, threshold_fact, threshold_cond)

def _evaluation_ensemble(models, dataCenter, data, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition, ensemble_model, device, is_filter=False, write_prediction=False, out_file=None):

	raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_POSCAPs, raw_LM_SENTENCEs, raw_OUTs, raw_instance_list = data
	#SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	MICS = zip(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs, raw_instance_list)
	MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs, instance_list = zip(*MICS)
	SENTENCEs = list(SENTENCEs)
	POSTAGs = list(POSTAGs)
	CAPs = list(CAPs)
	POSCAPs = list(POSCAPs)
	LM_SENTENCEs = list(LM_SENTENCEs)
	OUTs = list(OUTs)
	instance_list = list(instance_list)

	lm_input = single_model_predict(models[0], (SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs), len(SENTENCEs))
	pos_input = single_model_predict(models[1], (SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs), len(SENTENCEs))
	cap_input = single_model_predict(models[2], (SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs), len(SENTENCEs))

	predict_fact_batch, predict_condition_batch = ensemble_model((lm_input, pos_input, cap_input))

	return _core_evaluation(OUTs, SENTENCEs, instance_list, predict_fact_batch, predict_condition_batch, dataCenter, threshold_fact, threshold_cond, write_prediction, out_file, is_filter)
	

def _evaluation(model, dataCenter, data, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition, write_prediction=False, out_file=None):

	raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_POSCAPs, raw_LM_SENTENCEs, raw_OUTs, raw_instance_list = data
	#SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	MICS = zip(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs, raw_instance_list)
	MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs, instance_list = zip(*MICS)
	SENTENCEs = list(SENTENCEs)
	POSTAGs = list(POSTAGs)
	CAPs = list(CAPs)
	POSCAPs = list(POSCAPs)
	LM_SENTENCEs = list(LM_SENTENCEs)
	OUTs = list(OUTs)
	instance_list = list(instance_list)

	predict_fact_batch, predict_condition_batch = model.predict((SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs))

	return _core_evaluation(OUTs, SENTENCEs, instance_list, predict_fact_batch, predict_condition_batch, dataCenter, threshold_fact, threshold_cond, write_prediction, out_file)

def _single_evaluation(fact_model, cond_model, dataCenter, data, threshold_fact, threshold_cond):

	raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_POSCAPs, raw_LM_SENTENCEs, raw_OUTs, raw_instance_list = data
	#SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	MICS = zip(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs, raw_instance_list)
	MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs, instance_list = zip(*MICS)
	SENTENCEs = list(SENTENCEs)
	POSTAGs = list(POSTAGs)
	CAPs = list(CAPs)
	POSCAPs = list(POSCAPs)
	LM_SENTENCEs = list(LM_SENTENCEs)
	OUTs = list(OUTs)
	instance_list = list(instance_list)

	predict_fact_batch = fact_model.predict((SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs))
	predict_condition_batch = cond_model.predict((SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs))

	return _core_evaluation(OUTs, SENTENCEs, instance_list, predict_fact_batch, predict_condition_batch, dataCenter, threshold_fact, threshold_cond)

def single_model_load(model_file, device, dataCenter, seed, use_gate, enhance):
	dim = 50
	input_size = dim
	hidden_dim = 300

	str_config = model_file.split('_')[-1].split('.')[0]
	config = [bool(int(i)) for i in str_config]
	assert len(config) == 9
	lm_config = config[:3]
	postag_config = config[3:6]
	cap_config = config[6:9]
	poscap_config = [False, False, False] # ingore POSCAP, which was used but not now

	print('single_model config:', str_config)

	stmt_extraction_net = Stmt_Extraction_Net(dataCenter.WordEmbedding, dataCenter.Word2ID, dataCenter.POS2ID, dataCenter.CAP2ID, dataCenter.POSCAP2ID, dim, input_size, hidden_dim, len(dataCenter.Tag2ID_fact), len(dataCenter.Tag2ID_condition), lm_config, postag_config, cap_config, poscap_config, device, seed, use_gate, enhance)
	stmt_extraction_net.to(device)
	stmt_extraction_net.load_state_dict(torch.load(model_file))

	return stmt_extraction_net

def single_model_predict(model, input_tuple, batch_size, get_hidden=False):
	if get_hidden:
		predict_fact_batch, predict_condition_batch = model.predict_hidden(input_tuple, batch_size)
	else:
		predict_fact_batch, predict_condition_batch = model.predict_distrib(input_tuple, batch_size)
	return [predict_fact_batch, predict_condition_batch]

def ensemble_model_load(model_file, device, dataCenter, seed):
	str_config = model_file.split('_')[-1].split('.')[0]
	config = [bool(int(i)) for i in str_config]
	assert len(config) == 3
	use_lm = config[0]
	use_postag = config[1]
	use_cap = config[2]

	print('single_model config:', str_config)

	ensemble_model = Ensemble_Net(use_lm, use_postag, use_cap, len(dataCenter.Tag2ID_fact), device, seed)
	ensemble_model.to(device)
	ensemble_model.load_state_dict(torch.load(model_file))

	return ensemble_model

def ensemble_model_predict(model, input_tuple):
	predict_fact_batch, predict_condition_batch = model(input_tuple)
	return [predict_fact_batch, predict_condition_batch]

def apply_model_two_ensemble(models, batch_size, apply_type, dataCenter, device, weight_classes_fact, weight_classes_condition, ensemble_model):
	raw_SENTENCEs = getattr(dataCenter, apply_type+'_SENTENCEs')
	raw_POSTAGs = getattr(dataCenter, apply_type+'_POSTAGs')
	raw_CAPs = getattr(dataCenter, apply_type+'_CAPs')
	raw_LM_SENTENCEs = getattr(dataCenter, apply_type+'_LM_SENTENCEs')
	raw_POSCAPs = getattr(dataCenter, apply_type+'_POSCAPs')
	raw_OUTs = getattr(dataCenter, apply_type+'_OUTs')
		
	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	batches = len(SENTENCEs) // batch_size
	Tag2Metrics = dict()

	if apply_type == 'TRAIN':
		params = []
		for param in ensemble_model.parameters():
			if param.requires_grad:
				params.append(param)
		optimizer = optim.SGD(params, lr = 0.01, weight_decay=0.0005, momentum=0.9)

	loss_sum = 0
	for index in range(batches):
		SENTENCEs_batch = SENTENCEs[index*batch_size: (index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size: (index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size: (index+1)*batch_size]
		POSCAPs_batch = POSCAPs[index*batch_size: (index+1)*batch_size]
		LM_SENTENCEs_batch = LM_SENTENCEs[index*batch_size: (index+1)*batch_size]
		OUTs_batch = OUTs[index*batch_size: (index+1)*batch_size]
		
		MICS = zip(SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch)
		MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
		SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch = zip(*MICS)
		SENTENCEs_batch = list(SENTENCEs_batch)
		POSTAGs_batch = list(POSTAGs_batch)
		CAPs_batch = list(CAPs_batch)
		POSCAPs_batch = list(POSCAPs_batch)
		LM_SENTENCEs_batch = list(LM_SENTENCEs_batch)
		OUTs_batch = list(OUTs_batch)
		
		assert len(SENTENCEs_batch) == len(OUTs_batch)

		lm_input_batch = single_model_predict(models[0], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), batch_size)
		pos_input_batch = single_model_predict(models[1], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), batch_size)
		cap_input_batch = single_model_predict(models[2], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), batch_size)

		first_input_batch = ensemble_model_predict(models[3], (lm_input_batch, pos_input_batch, cap_input_batch))
		second_input_batch = ensemble_model_predict(models[4], (lm_input_batch, pos_input_batch, cap_input_batch))

		optimizer.zero_grad()
		ensemble_model.zero_grad()

		predict_fact_batch, predict_condition_batch = ensemble_model((first_input_batch, second_input_batch))

		assert len(OUTs_batch) == len(predict_fact_batch) == len(predict_condition_batch)

		loss = autograd.Variable(torch.FloatTensor([0])).to(device)

		for i in range(len(OUTs_batch)):
			for j in range(len(OUTs_batch[i][0])):
				tagID = OUTs_batch[i][0][j]
				_loss = (-weight_classes_fact[tagID] * predict_fact_batch[i][j][tagID])
				loss += _loss
			for j in range(len(OUTs_batch[i][1])):
				tagID = OUTs_batch[i][1][j]
				_loss = (-weight_classes_condition[tagID] * predict_condition_batch[i][j][tagID])
				loss += _loss

		loss /= len(OUTs_batch)
		loss_sum += loss

		print('batch-%d-aver_loss = %.6f(%d)' % (index, loss.data[0], len(SENTENCEs_batch[0])))

		loss.backward()
		nn.utils.clip_grad_norm_(ensemble_model.parameters(), 5)
		#print grad_norm
		optimizer.step()
	loss_sum /= batches
	print('loss_mean =', loss_sum)
	if loss_sum < 0.01:
		sys.exit(1)

def apply_model_ensemble(models, batch_size, apply_type, dataCenter, device, weight_classes_fact, weight_classes_condition, ensemble_model):
	raw_SENTENCEs = getattr(dataCenter, apply_type+'_SENTENCEs')
	raw_POSTAGs = getattr(dataCenter, apply_type+'_POSTAGs')
	raw_CAPs = getattr(dataCenter, apply_type+'_CAPs')
	raw_LM_SENTENCEs = getattr(dataCenter, apply_type+'_LM_SENTENCEs')
	raw_POSCAPs = getattr(dataCenter, apply_type+'_POSCAPs')
	raw_OUTs = getattr(dataCenter, apply_type+'_OUTs')
		
	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	batches = len(SENTENCEs) // batch_size
	Tag2Metrics = dict()

	if apply_type == 'TRAIN':
		params = []
		for param in ensemble_model.parameters():
			if param.requires_grad:
				params.append(param)
		optimizer = optim.SGD(params, lr = 0.01, weight_decay=0.0005, momentum=0.9)

	loss_sum = 0
	for index in range(batches):
		SENTENCEs_batch = SENTENCEs[index*batch_size: (index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size: (index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size: (index+1)*batch_size]
		POSCAPs_batch = POSCAPs[index*batch_size: (index+1)*batch_size]
		LM_SENTENCEs_batch = LM_SENTENCEs[index*batch_size: (index+1)*batch_size]
		OUTs_batch = OUTs[index*batch_size: (index+1)*batch_size]
		
		MICS = zip(SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch)
		MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
		SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch = zip(*MICS)
		SENTENCEs_batch = list(SENTENCEs_batch)
		POSTAGs_batch = list(POSTAGs_batch)
		CAPs_batch = list(CAPs_batch)
		POSCAPs_batch = list(POSCAPs_batch)
		LM_SENTENCEs_batch = list(LM_SENTENCEs_batch)
		OUTs_batch = list(OUTs_batch)
		
		assert len(SENTENCEs_batch) == len(OUTs_batch)

		lm_input_batch = single_model_predict(models[0], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), batch_size)

		pos_input_batch = single_model_predict(models[1], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), batch_size)

		cap_input_batch = single_model_predict(models[2], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), batch_size)

		optimizer.zero_grad()
		ensemble_model.zero_grad()

		predict_fact_batch, predict_condition_batch = ensemble_model((lm_input_batch, pos_input_batch, cap_input_batch))

		assert len(OUTs_batch) == len(predict_fact_batch) == len(predict_condition_batch)

		loss = autograd.Variable(torch.FloatTensor([0])).to(device)

		for i in range(len(OUTs_batch)):
			for j in range(len(OUTs_batch[i][0])):
				tagID = OUTs_batch[i][0][j]
				_loss = (-weight_classes_fact[tagID] * predict_fact_batch[i][j][tagID])
				loss += _loss
			for j in range(len(OUTs_batch[i][1])):
				tagID = OUTs_batch[i][1][j]
				_loss = (-weight_classes_condition[tagID] * predict_condition_batch[i][j][tagID])
				loss += _loss

		loss /= len(OUTs_batch)
		loss_sum += loss

		print('batch-%d-aver_loss = %.6f(%d)' % (index, loss.data[0], len(SENTENCEs_batch[0])))

		loss.backward()
		nn.utils.clip_grad_norm_(ensemble_model.parameters(), 5)
		#print grad_norm
		optimizer.step()
	loss_sum /= batches
	print('loss_mean =', loss_sum)
	if loss_sum < 0.9:
		sys.exit(1)

def apply_model(model, batch_size, apply_type, dataCenter, device, weight_classes_fact=None, weight_classes_condition=None):
	raw_SENTENCEs = getattr(dataCenter, apply_type+'_SENTENCEs')
	raw_POSTAGs = getattr(dataCenter, apply_type+'_POSTAGs')
	raw_CAPs = getattr(dataCenter, apply_type+'_CAPs')
	raw_LM_SENTENCEs = getattr(dataCenter, apply_type+'_LM_SENTENCEs')
	raw_POSCAPs = getattr(dataCenter, apply_type+'_POSCAPs')
	raw_OUTs = getattr(dataCenter, apply_type+'_OUTs')
		

	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	batches = len(SENTENCEs) // batch_size
	Tag2Metrics = dict()

	if apply_type == 'TRAIN':
		params = []
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)
		optimizer = optim.SGD(params, lr = 0.01, weight_decay=0.0005, momentum=0.9)

	for index in range(batches):
		SENTENCEs_batch = SENTENCEs[index*batch_size: (index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size: (index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size: (index+1)*batch_size]
		POSCAPs_batch = POSCAPs[index*batch_size: (index+1)*batch_size]
		LM_SENTENCEs_batch = LM_SENTENCEs[index*batch_size: (index+1)*batch_size]
		OUTs_batch = OUTs[index*batch_size: (index+1)*batch_size]
		
		MICS = zip(SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch)
		MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
		SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch = zip(*MICS)
		SENTENCEs_batch = list(SENTENCEs_batch)
		POSTAGs_batch = list(POSTAGs_batch)
		CAPs_batch = list(CAPs_batch)
		POSCAPs_batch = list(POSCAPs_batch)
		LM_SENTENCEs_batch = list(LM_SENTENCEs_batch)
		OUTs_batch = list(OUTs_batch)
		
		assert len(SENTENCEs_batch) == len(OUTs_batch)
		if apply_type == 'TRAIN':
			optimizer.zero_grad()
			model.zero_grad()

			predict_fact_batch, predict_condition_batch = model((SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), batch_size)

			assert len(OUTs_batch) == len(predict_fact_batch) == len(predict_condition_batch)

			loss = autograd.Variable(torch.FloatTensor([0])).to(device)

			for i in range(len(OUTs_batch)):
				for j in range(len(OUTs_batch[i][0])):
					tagID = OUTs_batch[i][0][j]
					_loss = (-weight_classes_fact[tagID] * predict_fact_batch[i][j][tagID])
					loss += _loss
				for j in range(len(OUTs_batch[i][1])):
					tagID = OUTs_batch[i][1][j]
					_loss = (-weight_classes_condition[tagID] * predict_condition_batch[i][j][tagID])
					loss += _loss

			loss /= len(OUTs_batch)

			print('batch-%d-aver_loss = %.6f(%d)' % (index, loss.data[0], len(SENTENCEs_batch[0])))

			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 5)
			#print grad_norm
			optimizer.step()
		else:
			predict_fact_batch, predict_condition_batch = model.predict((SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch))

			assert len(OUTs_batch) == len(predict_fact_batch) == len(predict_condition_batch)
			print(len(OUTs_batch), len(predict_fact_batch), len(predict_condition_batch))

			for i in range(len(predict_fact_batch)):
				_, predicted_fact_tags = torch.max(predict_fact_batch[i], 1)
				_, predicted_conditions_tags = torch.max(predict_condition_batch[i], 1)
				for j in range(len(OUTs_batch[i][0])):
					y_true = OUTs_batch[i][0][j]
					y_predict = predicted_fact_tags[j].item()

					if y_true == y_predict:
						tag = dataCenter.ID2Tag_fact[y_predict]
						if tag not in Tag2Metrics:
							Tag2Metrics[tag] = Metrics()
						Tag2Metrics[tag].TP += 1

					elif y_true != y_predict:
						tag_true = dataCenter.ID2Tag_fact[y_true]
						tag_predict = dataCenter.ID2Tag_fact[y_predict]
						if tag_true not in Tag2Metrics:
							Tag2Metrics[tag_true] = Metrics()
						if tag_predict not in Tag2Metrics:
							Tag2Metrics[tag_predict] = Metrics()
						Tag2Metrics[tag_true].FN += 1
						Tag2Metrics[tag_predict].FP += 1

				for j in range(len(OUTs_batch[i][1])):
					y_true = OUTs_batch[i][1][j]
					y_predict = predicted_conditions_tags[j].item()
					if y_true == y_predict:
						tag = dataCenter.ID2Tag_condition[y_predict]
						if tag not in Tag2Metrics:
							Tag2Metrics[tag] = Metrics()
						Tag2Metrics[tag].TP += 1

					elif y_true != y_predict:
						tag_true = dataCenter.ID2Tag_condition[y_true]
						tag_predict = dataCenter.ID2Tag_condition[y_predict]
						if tag_true not in Tag2Metrics:
							Tag2Metrics[tag_true] = Metrics()
						if tag_predict not in Tag2Metrics:
							Tag2Metrics[tag_predict] = Metrics()
						Tag2Metrics[tag_true].FN += 1
						Tag2Metrics[tag_predict].FP += 1

	return Tag2Metrics

def single_apply_model(fact_model, cond_model, batch_size, apply_type, dataCenter, device, weight_classes_fact=None, weight_classes_condition=None):
	raw_SENTENCEs = getattr(dataCenter, apply_type+'_SENTENCEs')
	raw_POSTAGs = getattr(dataCenter, apply_type+'_POSTAGs')
	raw_CAPs = getattr(dataCenter, apply_type+'_CAPs')
	raw_LM_SENTENCEs = getattr(dataCenter, apply_type+'_LM_SENTENCEs')
	raw_POSCAPs = getattr(dataCenter, apply_type+'_POSCAPs')
	raw_OUTs = getattr(dataCenter, apply_type+'_OUTs')
		

	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	batches = len(SENTENCEs) // batch_size
	Tag2Metrics = dict()

	if apply_type == 'TRAIN':
		params = []
		for param in fact_model.parameters():
			if param.requires_grad:
				params.append(param)
		fact_optimizer = optim.SGD(params, lr = 0.01, weight_decay=0.0005, momentum=0.9)
		params = []
		for param in cond_model.parameters():
			if param.requires_grad:
				params.append(param)
		cond_optimizer = optim.SGD(params, lr = 0.01, weight_decay=0.0005, momentum=0.9)

	for index in range(batches):
		SENTENCEs_batch = SENTENCEs[index*batch_size: (index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size: (index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size: (index+1)*batch_size]
		POSCAPs_batch = POSCAPs[index*batch_size: (index+1)*batch_size]
		LM_SENTENCEs_batch = LM_SENTENCEs[index*batch_size: (index+1)*batch_size]
		OUTs_batch = OUTs[index*batch_size: (index+1)*batch_size]
		
		MICS = zip(SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch)
		MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
		SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch = zip(*MICS)
		SENTENCEs_batch = list(SENTENCEs_batch)
		POSTAGs_batch = list(POSTAGs_batch)
		CAPs_batch = list(CAPs_batch)
		POSCAPs_batch = list(POSCAPs_batch)
		LM_SENTENCEs_batch = list(LM_SENTENCEs_batch)
		OUTs_batch = list(OUTs_batch)
		
		assert len(SENTENCEs_batch) == len(OUTs_batch)

		fact_optimizer.zero_grad()
		fact_model.zero_grad()
		cond_optimizer.zero_grad()
		cond_model.zero_grad()

		predict_fact_batch = fact_model((SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch))

		predict_condition_batch = cond_model((SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch))

		assert len(OUTs_batch) == len(predict_fact_batch) == len(predict_condition_batch)

		fact_loss = autograd.Variable(torch.FloatTensor([0])).to(device)
		cond_loss = autograd.Variable(torch.FloatTensor([0])).to(device)

		for i in range(len(OUTs_batch)):
			for j in range(len(OUTs_batch[i][0])):
				tagID = OUTs_batch[i][0][j]
				_loss = (-weight_classes_fact[tagID] * predict_fact_batch[i][j][tagID])
				fact_loss += _loss
			for j in range(len(OUTs_batch[i][1])):
				tagID = OUTs_batch[i][1][j]
				_loss = (-weight_classes_condition[tagID] * predict_condition_batch[i][j][tagID])
				cond_loss += _loss

		fact_loss /= len(OUTs_batch)
		cond_loss /= len(OUTs_batch)

		print('batch-%d-aver_fact_loss = %.6f, aver_cond_loss = %.6f(%d)' % (index, fact_loss.data[0], cond_loss.data[0], len(SENTENCEs_batch[0])))

		fact_loss.backward()
		cond_loss.backward()
		nn.utils.clip_grad_norm_(fact_model.parameters(), 5)
		nn.utils.clip_grad_norm_(cond_model.parameters(), 5)
		#print grad_norm
		fact_optimizer.step()
		cond_optimizer.step()

def retrain_evaluation(model, file_name, dataCenter, threshold_fact, threshold_cond, max_f1, max_std, out_model_name, num_pass, weight_classes_fact, weight_classes_condition, better=True):

	valid_tag_levels = []
	valid_tuple_levels = []
	valid_tag2metric_list = []
	valid_tag2tag_fact_list = []
	valid_tag2tag_cond_list = []

	test_tag_levels = []
	test_tuple_levels = []
	test_tag2metric_list = []
	test_tag2tag_fact_list = []
	test_tag2tag_cond_list = []

	for i in range(num_pass):
		VALID_DATA, TEST_DATA = dataCenter.get_evaluation(1.0/num_pass)
		print(len(VALID_DATA[0]), len(TEST_DATA[0]))
		loss, valid_tag_level, valid_Metrics, valid_tuple_level= _evaluation(model, dataCenter, VALID_DATA, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition)
		valid_tag_levels.append(valid_tag_level)
		valid_tuple_levels.append(valid_tuple_level)
		Tag2Metrics, tag2tag_fact, tag2tag_cond = valid_Metrics
		valid_tag2metric_list.append(metric_to_list(Tag2Metrics))
		valid_tag2tag_fact_list.append(tag2tag_to_list(tag2tag_fact))
		valid_tag2tag_cond_list.append(tag2tag_to_list(tag2tag_cond))

		loss, test_tag_level, test_Metrics, test_tuple_level= _evaluation(model, dataCenter, TEST_DATA, threshold_fact, threshold_cond, weight_classes_fact, weight_classes_condition)
		test_tag_levels.append(test_tag_level)
		test_tuple_levels.append(test_tuple_level)
		Tag2Metrics, tag2tag_fact, tag2tag_cond = test_Metrics
		test_tag2metric_list.append(metric_to_list(Tag2Metrics))
		test_tag2tag_fact_list.append(tag2tag_to_list(tag2tag_fact))
		test_tag2tag_cond_list.append(tag2tag_to_list(tag2tag_cond))

	valid_tag_levels_mean = np.asarray(valid_tag_levels).mean(0)
	valid_tag_levels_std = np.asarray(valid_tag_levels).std(0)

	valid_tag2metric_mean = np.asarray(valid_tag2metric_list).mean(0)
	valid_tag2metric_std = np.asarray(valid_tag2metric_list).std(0)
	valid_tag2tag_fact_mean = np.asarray(valid_tag2tag_fact_list).mean(0)
	valid_tag2tag_fact_std = np.asarray(valid_tag2tag_fact_list).std(0)
	valid_tag2tag_cond_mean = np.asarray(valid_tag2tag_cond_list).mean(0)
	valid_tag2tag_cond_std = np.asarray(valid_tag2tag_cond_list).std(0)

	valid_tuple_levels_mean = np.asarray(valid_tuple_levels).mean(0)
	valid_tuple_levels_std = np.asarray(valid_tuple_levels).std(0)

	test_tag_levels_mean = np.asarray(test_tag_levels).mean(0)
	test_tag_levels_std = np.asarray(test_tag_levels).std(0)

	test_tag2metric_mean = np.asarray(test_tag2metric_list).mean(0)
	test_tag2metric_std = np.asarray(test_tag2metric_list).std(0)
	test_tag2tag_fact_mean = np.asarray(test_tag2tag_fact_list).mean(0)
	test_tag2tag_fact_std = np.asarray(test_tag2tag_fact_list).std(0)
	test_tag2tag_cond_mean = np.asarray(test_tag2tag_cond_list).mean(0)
	test_tag2tag_cond_std = np.asarray(test_tag2tag_cond_list).std(0)

	test_tuple_levels_mean = np.asarray(test_tuple_levels).mean(0)
	test_tuple_levels_std = np.asarray(test_tuple_levels).std(0)

	micro_F1 = valid_tag_levels_mean[-1][2]
	micro_std = valid_tag_levels_std[-1][2]
	macro_F1 = valid_tag_levels_mean[-1][-1]
	macro_std = valid_tag_levels_std[-1][-1]
	
	print(test_tag_levels_mean[-1][-1], test_tag_levels_std[-1][-1])
	print(micro_F1, micro_std, macro_F1, macro_std)
	# if (macro_F1-macro_std) > (max_f1-max_std) and (macro_F1+macro_std) > (max_f1+max_std):
	# if (macro_F1-macro_std) > (max_f1-max_std) and (macro_F1+macro_std > max_f1+max_std) and (macro_std < 3.0):
	# if better and (macro_F1-macro_std > max_f1-max_std) and (macro_F1+macro_std > max_f1+max_std):
	if macro_F1 > max_f1:
	# if True:
		max_f1 = macro_F1
		max_std = macro_std
		print('saving model ...')
		torch.save(model.state_dict(), out_model_name)
		print('saving done.')

		fo = open(file_name, 'w')
		for i in range(len(test_tag_levels_mean)):
			for j in range(len(test_tag_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (test_tag_levels_mean[i][j], test_tag_levels_std[i][j]))
		for i in range(len(test_tuple_levels_mean)):
			for j in range(len(test_tuple_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (test_tuple_levels_mean[i][j], test_tuple_levels_std[i][j]))

		for i in range(len(valid_tag_levels_mean)):
			for j in range(len(valid_tag_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (valid_tag_levels_mean[i][j], valid_tag_levels_std[i][j]))
		for i in range(len(valid_tuple_levels_mean)):
			for j in range(len(valid_tuple_levels_mean[i])):
				fo.write('%.2f+/-%.2f\t' % (valid_tuple_levels_mean[i][j], valid_tuple_levels_std[i][j]))

		fo.write('\n')

		i = 0
		assert len(Tag2Metrics) == len(valid_tag2metric_mean)
		for tag in sorted(Tag2Metrics):
			assert len(valid_tag2metric_mean[i]) == 3
			# print '%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f' % (tag, valid_tag2metric_mean[i][0], valid_tag2metric_std[i][0], valid_tag2metric_mean[i][1], valid_tag2metric_std[i][1], valid_tag2metric_mean[i][2], valid_tag2metric_std[i][2])
			fo.write('%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f\n' % (tag, valid_tag2metric_mean[i][0], valid_tag2metric_std[i][0], valid_tag2metric_mean[i][1], valid_tag2metric_std[i][1], valid_tag2metric_mean[i][2], valid_tag2metric_std[i][2]))
			i += 1


		string = '-'
		for vtag in sorted(tag2tag_fact['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')

		i = 0
		assert len(tag2tag_fact) == len(valid_tag2tag_fact_mean) == len(valid_tag2tag_fact_std)
		for tag in sorted(tag2tag_fact):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_fact[tag]):
				mean = valid_tag2tag_fact_mean[i][j]
				std = valid_tag2tag_fact_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		string = '-'
		for vtag in sorted(tag2tag_cond['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')
		i = 0
		assert len(tag2tag_cond) == len(valid_tag2tag_cond_mean) == len(valid_tag2tag_cond_std)
		for tag in sorted(tag2tag_cond):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_cond[tag]):
				mean = valid_tag2tag_cond_mean[i][j]
				std = valid_tag2tag_cond_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		i = 0
		assert len(Tag2Metrics) == len(test_tag2metric_mean)
		for tag in sorted(Tag2Metrics):
			assert len(test_tag2metric_mean[i]) == 3
			# print '%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f' % (tag, test_tag2metric_mean[i][0], test_tag2metric_std[i][0], test_tag2metric_mean[i][1], test_tag2metric_std[i][1], test_tag2metric_mean[i][2], test_tag2metric_std[i][2])
			fo.write('%s\t%.2f+/-%.2f\t%.2f+/-%.2f\t%.2f+/-%.2f\n' % (tag, test_tag2metric_mean[i][0], test_tag2metric_std[i][0], test_tag2metric_mean[i][1], test_tag2metric_std[i][1], test_tag2metric_mean[i][2], test_tag2metric_std[i][2]))
			i += 1


		string = '-'
		for vtag in sorted(tag2tag_fact['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')

		i = 0
		assert len(tag2tag_fact) == len(test_tag2tag_fact_mean) == len(test_tag2tag_fact_std)
		for tag in sorted(tag2tag_fact):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_fact[tag]):
				mean = test_tag2tag_fact_mean[i][j]
				std = test_tag2tag_fact_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		string = '-'
		for vtag in sorted(tag2tag_cond['O']):
			string += ('\t'+vtag+'\t_std_')
		# print string
		fo.write(string+'\n')
		i = 0
		assert len(tag2tag_cond) == len(test_tag2tag_cond_mean) == len(test_tag2tag_cond_std)
		for tag in sorted(tag2tag_cond):
			string = tag
			j = 0
			for vtag in sorted(tag2tag_cond[tag]):
				mean = test_tag2tag_cond_mean[i][j]
				std = test_tag2tag_cond_std[i][j]
				string += ('\t%.0f\t%.0f' % (mean, std))
				j += 1
			# print string
			fo.write(string+'\n')
			i += 1

		fo.close()

	return max_f1, max_std

def retrain_model(model, out_file, batch_size, dataCenter, device, weight_classes_fact, weight_classes_condition, tuple_sequence, out_model_name, in_model_name, max_f1, max_std, num_pass):
	raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs = tuple_sequence

	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	batches = len(SENTENCEs) // batch_size

	params = []
	for param in model.parameters():
		if param.requires_grad:
			params.append(param)
	optimizer = optim.SGD(params, lr = 0.01, weight_decay=0.0005, momentum=0.9)

	for index in range(batches):
		SENTENCEs_batch = SENTENCEs[index*batch_size: (index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size: (index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size: (index+1)*batch_size]
		POSCAPs_batch = POSCAPs[index*batch_size: (index+1)*batch_size]
		LM_SENTENCEs_batch = LM_SENTENCEs[index*batch_size: (index+1)*batch_size]
		OUTs_batch = OUTs[index*batch_size: (index+1)*batch_size]
		
		MICS = zip(SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch)
		MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
		SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch = zip(*MICS)
		SENTENCEs_batch = list(SENTENCEs_batch)
		POSTAGs_batch = list(POSTAGs_batch)
		CAPs_batch = list(CAPs_batch)
		POSCAPs_batch = list(POSCAPs_batch)
		LM_SENTENCEs_batch = list(LM_SENTENCEs_batch)
		OUTs_batch = list(OUTs_batch)
		
		assert len(SENTENCEs_batch) == len(OUTs_batch)
		optimizer.zero_grad()
		model.zero_grad()

		predict_fact_batch, predict_condition_batch = model((SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), batch_size)

		assert len(OUTs_batch) == len(predict_fact_batch) == len(predict_condition_batch)

		loss = autograd.Variable(torch.FloatTensor([0])).to(device)

		for i in range(len(OUTs_batch)):
			for j in range(len(OUTs_batch[i][0])):
				tagID = OUTs_batch[i][0][j]
				_loss = (-weight_classes_fact[tagID] * predict_fact_batch[i][j][tagID])
				loss += _loss
			for j in range(len(OUTs_batch[i][1])):
				tagID = OUTs_batch[i][1][j]
				_loss = (-weight_classes_condition[tagID] * predict_condition_batch[i][j][tagID])
				loss += _loss

		loss /= len(OUTs_batch)

		print('batch-%d-aver_loss = %.6f(%d)' % (index, loss.data[0], len(SENTENCEs_batch[0])))


		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), 5)
		#print grad_norm
		optimizer.step()

		max_f1, max_std = retrain_evaluation(model, out_file, dataCenter, 0, 0, max_f1, max_std, out_model_name, num_pass, weight_classes_fact, weight_classes_condition)
		print('MAX:', max_f1, max_std)

	return max_f1, max_std

def retrain_ensemble_model(models, ensemble_model, out_file, batch_size, dataCenter, device, weight_classes_fact, weight_classes_condition, tuple_sequence, out_model_name, in_model_name, max_f1, max_std, num_pass):
	raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs = tuple_sequence

	SENTENCEs, POSTAGs, CAPs, LM_SENTENCEs, POSCAPs, OUTs = shuffle(raw_SENTENCEs, raw_POSTAGs, raw_CAPs, raw_LM_SENTENCEs, raw_POSCAPs, raw_OUTs)

	batches = len(SENTENCEs) // batch_size

	params = []
	for param in ensemble_model.parameters():
		if param.requires_grad:
			params.append(param)
	optimizer = optim.SGD(params, lr = 0.01, weight_decay=0.0005, momentum=0.9)

	for index in range(batches):
		SENTENCEs_batch = SENTENCEs[index*batch_size: (index+1)*batch_size]
		POSTAGs_batch = POSTAGs[index*batch_size: (index+1)*batch_size]
		CAPs_batch = CAPs[index*batch_size: (index+1)*batch_size]
		POSCAPs_batch = POSCAPs[index*batch_size: (index+1)*batch_size]
		LM_SENTENCEs_batch = LM_SENTENCEs[index*batch_size: (index+1)*batch_size]
		OUTs_batch = OUTs[index*batch_size: (index+1)*batch_size]
		
		MICS = zip(SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch)
		MICS = sorted(MICS, key = lambda s: len(s[0]), reverse =True)
		SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch, OUTs_batch = zip(*MICS)
		SENTENCEs_batch = list(SENTENCEs_batch)
		POSTAGs_batch = list(POSTAGs_batch)
		CAPs_batch = list(CAPs_batch)
		POSCAPs_batch = list(POSCAPs_batch)
		LM_SENTENCEs_batch = list(LM_SENTENCEs_batch)
		OUTs_batch = list(OUTs_batch)
		
		assert len(SENTENCEs_batch) == len(OUTs_batch)
		optimizer.zero_grad()
		ensemble_model.zero_grad()

		lm_input = single_model_predict(models[0], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), len(SENTENCEs_batch))
		pos_input = single_model_predict(models[1], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), len(SENTENCEs_batch))
		cap_input = single_model_predict(models[2], (SENTENCEs_batch, POSTAGs_batch, CAPs_batch, LM_SENTENCEs_batch, POSCAPs_batch), len(SENTENCEs_batch))

		predict_fact_batch, predict_condition_batch = ensemble_model((lm_input, pos_input, cap_input))

		assert len(OUTs_batch) == len(predict_fact_batch) == len(predict_condition_batch)

		loss = autograd.Variable(torch.FloatTensor([0])).to(device)

		for i in range(len(OUTs_batch)):
			for j in range(len(OUTs_batch[i][0])):
				tagID = OUTs_batch[i][0][j]
				_loss = (-weight_classes_fact[tagID] * predict_fact_batch[i][j][tagID])
				loss += _loss
			for j in range(len(OUTs_batch[i][1])):
				tagID = OUTs_batch[i][1][j]
				_loss = (-weight_classes_condition[tagID] * predict_condition_batch[i][j][tagID])
				loss += _loss

		loss /= len(OUTs_batch)

		print('batch-%d-aver_loss = %.6f(%d)' % (index, loss.data[0], len(SENTENCEs_batch[0])))


		loss.backward()
		nn.utils.clip_grad_norm_(ensemble_model.parameters(), 5)
		#print grad_norm
		optimizer.step()

		max_f1, max_std = evaluation_ensemble(models, out_file, dataCenter, 0, 0, max_f1, max_std, out_model_name, num_pass, False, weight_classes_fact, weight_classes_condition, ensemble_model, device)
		print('MAX:', max_f1, max_std)

	return max_f1, max_std