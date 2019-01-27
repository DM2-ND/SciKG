import sys, os, io
import random
import json
import pickle
import logging
import argparse
import struct
import math
import itertools
import copy

import numpy as np

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

def is_blocked(start, end, predicate_set):
	if start > end:
		return True
	for predicate in predicate_set:
		if predicate[1] > start and predicate[1] < end:
			return True

def new_post_decoder(words, predicted_fact_tags, ID2Tag=None):
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
	def __init__(self):
		super(DataCenter, self).__init__()

		self.oov_set = set()

		self.Tag2Num = dict()

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
				assert len(seq) == senLen
				SENTENCEs.append(seq)
				instance.SENTENCE = seq
			elif seq_name == 'POSTAG':
				assert len(seq) == senLen
				POSTAGs.append(seq)
				instance.POSTAG = seq
			else:
				assert len(seq) == senLen
				CAPs.append(seq)
				instance.CAP = seq

		# print('OUT:'
		facts_out = ['O'] * senLen
		conditions_out = ['O'] * senLen

		assert len(instance.multi_output) == 2
		outs = []
		for _output in instance.multi_output:
			key = _output[0]
			seq = _output[1]
			for index in range(len(seq)):
				tag = seq[index]
				# print(tag,
				if key.startswith('f'):
					if tag != 'O':
						facts_out[index] = tag
				else:
					if tag != 'O':
						conditions_out[index] = tag
				
		assert len(facts_out) == len(conditions_out) == senLen

		outs = [facts_out, conditions_out]
		OUTs.append(outs)
		instance.OUT = outs
		try:
			assert len(SENTENCEs) == len(OUTs)
		except:
			print(paper_id, stmt_id, len(SENTENCEs), len(OUTs))
			print(multi_input)
			print(multi_output)
			sys.exit(1)
		if len(SENTENCEs) % 10000 == 0:
			print(len(SENTENCEs))

		instance_list = getattr(self, 'instance_'+dataset_type)
		instance_list.append(instance)

	def count_tag(self, tag):
		if tag not in self.Tag2Num:
			self.Tag2Num[tag] = 0
		self.Tag2Num[tag] += 1

	def _loading_dataset(self, dataset_type, dataFile):

		SENTENCEs = getattr(self, dataset_type+'_SENTENCEs')
		POSTAGs = getattr(self, dataset_type+'_POSTAGs')
		CAPs = getattr(self, dataset_type+'_CAPs')
		POSCAPs = getattr(self, dataset_type+'_POSCAPs')
		LM_SENTENCEs = getattr(self, dataset_type+'_LM_SENTENCEs')
		OUTs = getattr(self, dataset_type+'_OUTs')

		attr_tuple = (SENTENCEs, POSTAGs, CAPs, OUTs)

		print('loading '+dataset_type+' data from '+dataFile)

		paper_id_set = getattr(self, 'paper_id_set_'+dataset_type)
		paper_id = 'none'
		stmt_id = '0'
		multi_input = []
		multi_output = []
		previous = False

		line_nu = 0
		with open(dataFile, 'r') as fd:
			for line in fd:
				if line.startswith('=====') or len(line.strip('\t').split('\t')) < 2:
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
					continue
				line_list = line.strip('\n').split('\t')
				seq_name = line_list[0]
				seq = line_list[1:]
				if seq_name in ['WORD', 'POSTAG', 'CAP']:
					multi_input.append((seq_name, seq))
				else:
					multi_output.append((seq_name, seq))

		# instance_list = getattr(self, 'instance_'+dataset_type)
		# for i in range(len(CAPs)):
		# 	CAP = CAPs[i]
		# 	POSTAG = POSTAGs[i]
		# 	POSCAP = []
		# 	assert len(CAP) == len(POSTAG)
		# 	for j in range(len(CAP)):
		# 		if CAP[j] == 'O':
		# 			POSCAP.append(POSTAG[j])
		# 		else:
		# 			POSCAP.append(CAP[j])
		# 	assert len(CAP) == len(POSTAG) == len(POSCAP)
		# 	POSCAPs.append(POSCAP)
		# 	assert instance_list[i].CAP == CAP
		# 	instance_list[i].POSCAP = POSCAP

		print(len(SENTENCEs), len(POSTAGs), len(CAPs))

		print('done.')

	# def loading_dataset(self, trainFile, validFile, testFile):
	def loading_dataset(self, trainFile, evalFile):

		if trainFile != None:
			self._loading_dataset('TRAIN', trainFile)

		if evalFile != None:
			self._loading_dataset('EVAL', evalFile)


