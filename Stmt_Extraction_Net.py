# -*- coding: utf-8 -*-

import torch
import math
import sys

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM_encoder(nn.Module):
	"""docstring for LSTM_encoder"""
	def __init__(self, wordEmbedding, word2ID, pos2ID, cap2ID, poscap2ID, word_dim, input_size, hidden_size, num_layers, bidirectional, lm_config, postag_config, cap_config, poscap_config, device, use_gate, lm_type='normal'):
		super(LSTM_encoder, self).__init__()

		self.device = device
		self.bidirectional = bidirectional

		self.use_gate = use_gate

		self.lm_config = lm_config
		self.postag_config = postag_config
		self.cap_config = cap_config
		self.poscap_config = poscap_config

		self.WordEmbedding = nn.Embedding(len(word2ID), word_dim)
		self.POSEmbedding = nn.Embedding(len(pos2ID), int(math.ceil(math.log(len(pos2ID),2))))
		self.CAPEmbedding = nn.Embedding(len(cap2ID), int(math.ceil(math.log(len(cap2ID),2))))
		self.POSCAPEmbedding = nn.Embedding(len(poscap2ID), int(math.ceil(math.log(len(poscap2ID),2))))

		self.lg_lm = nn.Parameter(torch.randn(word_dim, word_dim))
		self.lg_pos = nn.Parameter(torch.randn(word_dim, word_dim))
		self.lg_cap = nn.Parameter(torch.randn(word_dim, word_dim))
		self.lg_poscap = nn.Parameter(torch.randn(word_dim, word_dim))

		self.lg_lm_b = nn.Parameter(torch.randn(word_dim))
		self.lg_pos_b = nn.Parameter(torch.randn(word_dim))
		self.lg_cap_b = nn.Parameter(torch.randn(word_dim))
		self.lg_poscap_b = nn.Parameter(torch.randn(word_dim))

		if lm_type == 'normal':
			self.w_lm = nn.Parameter(torch.randn(200, word_dim))
		elif lm_type == 'bert-base':
			self.w_lm = nn.Parameter(torch.randn(768, word_dim))
		else:
			assert lm_type == 'bert-large'
			self.w_lm = nn.Parameter(torch.randn(1024, word_dim))
		self.w_pos = nn.Parameter(torch.randn(int(math.ceil(math.log(len(pos2ID),2))), word_dim))
		self.w_cap = nn.Parameter(torch.randn(int(math.ceil(math.log(len(cap2ID),2))), word_dim))
		self.w_poscap = nn.Parameter(torch.randn(int(math.ceil(math.log(len(poscap2ID),2))), word_dim))

		self.WordEmbedding.weight = nn.Parameter(wordEmbedding, requires_grad=False)

		self.Word2ID = word2ID
		self.POS2ID = pos2ID
		self.CAP2ID = cap2ID
		self.POSCAP2ID = poscap2ID

		self.input_size = input_size
		# self.batch_size = batch_size
		self.hidden_size = hidden_size

		self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers, batch_first=True, bidirectional=self.bidirectional)
		self.hidden = None

		#for weight in self.parameters():
			#print type(weight), weight.size(), weight.requires_grad

	def forward(self, tuple_batch):
		sentences_batch, pos_batch, cap_batch, lm_batch, poscap_batch = tuple_batch

		length = []
		wordsIndex_batch = []
		posesIndex_batch = []
		capsIndex_batch = []
		lm_batch_new = []
		poscapsIndex_batch = []

		length_max = len(sentences_batch[0])

		for index in range(len(sentences_batch)):

			sentence = sentences_batch[index]
			poses = pos_batch[index]
			caps = cap_batch[index]
			lms = lm_batch[index]
			poscaps = poscap_batch[index]

			assert len(sentence) == len(poses) == len(caps) == len(lms) == len(poscaps)

			length.append(len(sentence))

			wordsIndex = []
			posesIndex = []
			capsIndex = []
			poscapsIndex = []

			for word in sentence:
				wordsIndex.append(self.Word2ID[word])
			wordsIndex += [0]*(length_max-len(wordsIndex))
			wordsIndex_batch.append(wordsIndex)

			for pos in poses:
				if pos not in self.POS2ID:
					posesIndex.append(self.POS2ID['SYM'])
				else:
					posesIndex.append(self.POS2ID[pos])
			posesIndex += [0]*(length_max-len(posesIndex))
			posesIndex_batch.append(posesIndex)

			for cap in caps:
				capsIndex.append(self.CAP2ID[cap])
			capsIndex += [0]*(length_max-len(capsIndex))
			capsIndex_batch.append(capsIndex)

			for poscap in poscaps:
				poscapsIndex.append(self.POSCAP2ID[poscap])
			poscapsIndex += [0]*(length_max-len(poscapsIndex))
			poscapsIndex_batch.append(poscapsIndex)

			# print lms.size(), length_max
			lms_pad = torch.randn((length_max-len(lms), len(lms[0])), dtype=torch.float32)
			lms = torch.cat([lms.to(self.device), lms_pad.to(self.device)])
			lm_batch_new.append(lms.view(1, lms.size(0), lms.size(1)))

		wordsIndex_batch = autograd.Variable(torch.LongTensor(wordsIndex_batch)).to(self.device)
		posesIndex_batch = autograd.Variable(torch.LongTensor(posesIndex_batch)).to(self.device)
		capsIndex_batch = autograd.Variable(torch.LongTensor(capsIndex_batch)).to(self.device)

		lm_batch_new = torch.cat(lm_batch_new, 0)
		lmsEmb = autograd.Variable(lm_batch_new).to(self.device)
		# print lm_batch_new.size()

		poscapsIndex_batch = autograd.Variable(torch.LongTensor(poscapsIndex_batch)).to(self.device)

		sentencesEmb = self.WordEmbedding(wordsIndex_batch)
		posesEmb = self.POSEmbedding(posesIndex_batch)
		capsEmb = self.CAPEmbedding(capsIndex_batch)
		poscapsEmb = self.POSCAPEmbedding(poscapsIndex_batch)

		emb = sentencesEmb + 0

		if self.use_gate:
			if self.lm_config[0]:
				emb += (torch.matmul(lmsEmb, self.w_lm)*torch.sigmoid(torch.matmul(sentencesEmb, self.lg_lm)+self.lg_lm_b))
			if self.postag_config[0]:
				emb += (torch.matmul(posesEmb, self.w_pos)*torch.sigmoid(torch.matmul(sentencesEmb, self.lg_pos)+self.lg_pos_b))
			if self.cap_config[0]:
				emb += (torch.matmul(capsEmb, self.w_cap)*torch.sigmoid(torch.matmul(sentencesEmb, self.lg_cap)+self.lg_cap_b))
			if self.poscap_config[0]:
				emb += (torch.matmul(poscapsEmb, self.w_poscap)*torch.sigmoid(torch.matmul(sentencesEmb, self.lg_poscap)+self.lg_poscap_b))
		else:
			if self.lm_config[0]:
				emb += torch.matmul(lmsEmb, self.w_lm)
			if self.postag_config[0]:
				emb += torch.matmul(posesEmb, self.w_pos)
			if self.cap_config[0]:
				emb += torch.matmul(capsEmb, self.w_cap)
			if self.poscap_config[0]:
				emb += torch.matmul(poscapsEmb, self.w_poscap)
		
		emb = emb.to(self.device)

		packed_sentencesEmb = pack_padded_sequence(emb, length, batch_first=True)
		
		packed_output, (ht, ct)  = self.lstm(packed_sentencesEmb, self.hidden)
		output, _ = pad_packed_sequence(packed_output, batch_first=True)

		return output.transpose(0,1).transpose(1,2), posesEmb.transpose(0,1).transpose(1,2), capsEmb.transpose(0,1).transpose(1,2), lmsEmb.transpose(0,1).transpose(1,2), poscapsEmb.transpose(0,1).transpose(1,2)
			
	def init_hidden(self, batch_size):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_size)
		return (autograd.Variable(torch.zeros(2, batch_size, self.hidden_size)).to(self.device),
				autograd.Variable(torch.zeros(2, batch_size, self.hidden_size)).to(self.device))

class LSTM_decoder(nn.Module):
	"""docstring for LSTM_decoder"""
	def __init__(self, input_size, hidden_size, tagset_size, pos2ID, cap2ID, poscap2ID, lm_config, postag_config, cap_config, poscap_config, device, use_gate, enhance, lm_type='normal'):
		super(LSTM_decoder, self).__init__()

		self.device = device
		self.tagset_size = tagset_size
		self.hidden_size = hidden_size
		#self.hidden = self.init_hidden()

		self.use_gate = use_gate
		self.enhance = enhance

		self.lm_config = lm_config
		self.postag_config = postag_config
		self.cap_config = cap_config
		self.poscap_config = poscap_config

		self.w_ii = nn.Parameter(torch.randn(4 * hidden_size, input_size))
		self.w_hi = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
		self.w_ti = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
		self.w_co = nn.Parameter(torch.randn(hidden_size, hidden_size))
		self.w_ht = nn.Parameter(torch.randn(hidden_size, hidden_size))
		self.b_i = nn.Parameter(torch.randn(5 * hidden_size))

		self.w_y_fact = nn.Parameter(torch.randn(tagset_size, hidden_size))
		self.b_y_fact = nn.Parameter(torch.randn(tagset_size))
		
		self.w_fact = nn.Parameter(torch.randn(tagset_size, tagset_size))
		self.w_y_cond = nn.Parameter(torch.randn(tagset_size, hidden_size))
		self.b_y_cond = nn.Parameter(torch.randn(tagset_size))

		self.mg_lm = nn.Parameter(torch.randn(input_size, input_size))
		self.mg_pos = nn.Parameter(torch.randn(input_size, input_size))
		self.mg_cap = nn.Parameter(torch.randn(input_size, input_size))
		self.mg_poscap = nn.Parameter(torch.randn(input_size, input_size))

		self.mg_lm_b = nn.Parameter(torch.randn(input_size))
		self.mg_pos_b = nn.Parameter(torch.randn(input_size))
		self.mg_cap_b = nn.Parameter(torch.randn(input_size))
		self.mg_poscap_b = nn.Parameter(torch.randn(input_size))

		self.tg_lm = nn.Parameter(torch.randn(hidden_size, hidden_size))
		self.tg_pos = nn.Parameter(torch.randn(hidden_size, hidden_size))
		self.tg_cap = nn.Parameter(torch.randn(hidden_size, hidden_size))
		self.tg_poscap = nn.Parameter(torch.randn(hidden_size, hidden_size))

		self.tg_lm_b = nn.Parameter(torch.randn(hidden_size))
		self.tg_pos_b = nn.Parameter(torch.randn(hidden_size))
		self.tg_cap_b = nn.Parameter(torch.randn(hidden_size))
		self.tg_poscap_b = nn.Parameter(torch.randn(hidden_size))

		if lm_type == 'normal':
			self.w_lmw = nn.Parameter(torch.randn(200, input_size))
		elif lm_type == 'bert-base':
			self.w_lmw = nn.Parameter(torch.randn(768, input_size))
		else:
			assert lm_type == 'bert-large'
			self.w_lmw = nn.Parameter(torch.randn(1024, input_size))
		self.w_posw = nn.Parameter(torch.randn(int(math.ceil(math.log(len(pos2ID),2))), input_size))
		self.w_capw = nn.Parameter(torch.randn(int(math.ceil(math.log(len(cap2ID),2))), input_size))
		self.w_poscapw = nn.Parameter(torch.randn(int(math.ceil(math.log(len(poscap2ID),2))), input_size))

		if lm_type == 'normal':
			self.w_lmt = nn.Parameter(torch.randn(hidden_size, 200))
		elif lm_type == 'bert-base':
			self.w_lmt = nn.Parameter(torch.randn(hidden_size, 768))
		else:
			assert lm_type == 'bert-large'
			self.w_lmt = nn.Parameter(torch.randn(hidden_size, 1024))
		self.w_post = nn.Parameter(torch.randn(hidden_size, int(math.ceil(math.log(len(pos2ID),2)))))
		self.w_capt = nn.Parameter(torch.randn(hidden_size, int(math.ceil(math.log(len(cap2ID),2)))))
		self.w_poscapt = nn.Parameter(torch.randn(hidden_size, int(math.ceil(math.log(len(poscap2ID),2)))))

		self.reset_parameters()

		self.hidden = None

		#for weight in self.parameters():
			#print type(weight), weight.size(), weight.requires_grad

	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			# print type(weight), weight.size()
			weight.data.uniform_(-stdv, stdv)

	def forward(self, inputs, lmsEmb, posesEmb, capsEmb, poscapsEmb):
		border = self.hidden_size
		hs = [self.hidden[0][0].transpose(0, 1)]
		cs = [self.hidden[1][0].transpose(0, 1)]
		ts = [self.hidden[2][0].transpose(0, 1)]

		new_inputs = inputs + 0
		if self.use_gate:
			if self.lm_config[1]:
				new_inputs += (torch.matmul(lmsEmb.transpose(1, 2), self.w_lmw).transpose(1, 2)*torch.sigmoid(torch.matmul(inputs.transpose(1, 2), self.mg_lm).transpose(1, 2)+self.mg_lm_b.view(-1,1)))
			if self.postag_config[1]:
				new_inputs += (torch.matmul(posesEmb.transpose(1, 2), self.w_posw).transpose(1, 2)*torch.sigmoid(torch.matmul(inputs.transpose(1, 2), self.mg_pos).transpose(1, 2)+self.mg_pos_b.view(-1,1)))
			if self.cap_config[1]:
				new_inputs += (torch.matmul(capsEmb.transpose(1, 2), self.w_capw).transpose(1, 2)*torch.sigmoid(torch.matmul(inputs.transpose(1, 2), self.mg_cap).transpose(1, 2)+self.mg_cap_b.view(-1,1)))
			if self.poscap_config[1]:
				new_inputs += (torch.matmul(poscapsEmb.transpose(1, 2), self.w_poscapw).transpose(1, 2)*torch.sigmoid(torch.matmul(inputs.transpose(1, 2), self.mg_poscap).transpose(1, 2)+self.mg_poscap_b.view(-1,1)))
		else:
			if self.lm_config[1]:
				new_inputs += torch.matmul(lmsEmb.transpose(1, 2), self.w_lmw).transpose(1, 2)
			if self.postag_config[1]:
				new_inputs += torch.matmul(posesEmb.transpose(1, 2), self.w_posw).transpose(1, 2)
			if self.cap_config[1]:
				new_inputs += torch.matmul(capsEmb.transpose(1, 2), self.w_capw).transpose(1, 2)
			if self.poscap_config[1]:
				new_inputs += torch.matmul(poscapsEmb.transpose(1, 2), self.w_poscapw).transpose(1, 2)

		hidden_out = []
		outputs_fact = []
		outputs_distrib_fact = []
		outputs_condition = []
		outputs_distrib_condition = []

		for index in range(len(new_inputs)):
			_input = new_inputs[index]
			posEmb = posesEmb[index]
			capEmb = capsEmb[index]
			lmEmb = lmsEmb[index]
			poscapEmb = poscapsEmb[index]

			ii = torch.mm(self.w_ii, _input)
			hi = torch.mm(self.w_hi, hs[-1])
			ti = torch.mm(self.w_ti, ts[-1])
			i = torch.sigmoid(ii[:border] + hi[:border] + ti[:border] + self.b_i[:border].view(-1,1))
			f = torch.sigmoid(ii[border:2*border] + hi[border:2*border] + ti[border:2*border] + self.b_i[border:2*border].view(-1,1))
			z = torch.tanh(ii[2*border:3*border] + hi[2*border:3*border] + ti[2*border:3*border] + self.b_i[2*border:3*border].view(-1,1))
			c = f * cs[-1] + i * z
			o = torch.sigmoid(ii[3*border:4*border] + hi[3*border:4*border] + torch.mm(self.w_co, c) + self.b_i[3*border:4*border].view(-1,1))
			h = o * torch.tanh(c)
			_T = torch.mm(self.w_ht, h) + self.b_i[4*border:].view(-1,1)
			T = _T + 0

			if self.use_gate:
				if self.lm_config[-1]:
					T += (torch.mm(self.w_lmt, lmEmb)*torch.sigmoid(torch.mm(self.tg_lm, _T)+self.tg_lm_b.view(-1,1)))
				if self.postag_config[-1]:
					T += (torch.mm(self.w_post, posEmb)*torch.sigmoid(torch.mm(self.tg_pos, _T)+self.tg_pos_b.view(-1,1)))
				if self.cap_config[-1]:
					T += (torch.mm(self.w_capt, capEmb)*torch.sigmoid(torch.mm(self.tg_cap, _T)+self.tg_cap_b.view(-1,1)))
				if self.poscap_config[-1]:
					T += (torch.mm(self.w_poscapt, poscapEmb)*torch.sigmoid(torch.mm(self.tg_poscap, _T)+self.tg_poscap_b.view(-1,1)))
			else:
				if self.lm_config[-1]:
					T += torch.mm(self.w_lmt, lmEmb)
				if self.postag_config[-1]:
					T += torch.mm(self.w_post, posEmb)
				if self.cap_config[-1]:
					T += torch.mm(self.w_capt, capEmb)
				if self.poscap_config[-1]:
					T += torch.mm(self.w_poscapt, poscapEmb)

			hs.append(h)
			cs.append(c)
			ts.append(T)

			y_fact = torch.mm(self.w_y_fact, T) + self.b_y_fact.view(-1,1)
			outputs_fact.append(F.log_softmax(y_fact, 0).view(1, self.tagset_size, -1))
			outputs_distrib_fact.append(F.softmax(y_fact, 0).view(1, self.tagset_size, -1))

			hidden_out.append(T.view(1, self.hidden_size, -1))

			if self.enhance:
				y_condition = torch.mm(self.w_y_cond, T) + torch.mm(self.w_fact, F.softmax(y_fact, 0)) + self.b_y_cond.view(-1,1)
			else:
				y_condition = torch.mm(self.w_y_cond, T) + self.b_y_cond.view(-1,1)

			outputs_condition.append(F.log_softmax(y_condition, 0).view(1, self.tagset_size, -1))
			outputs_distrib_condition.append(F.softmax(y_condition, 0).view(1, self.tagset_size, -1))

		outputs_fact = torch.cat(outputs_fact).transpose(0,2).transpose(1,2)
		outputs_distrib_fact = torch.cat(outputs_distrib_fact).transpose(0,2).transpose(1,2)

		outputs_condition = torch.cat(outputs_condition).transpose(0,2).transpose(1,2)
		outputs_distrib_condition = torch.cat(outputs_distrib_condition).transpose(0,2).transpose(1,2)

		hidden_out = torch.cat(hidden_out).transpose(0,2).transpose(1,2)

		# print outputs.size(), type(outputs)
		return outputs_fact, outputs_condition, outputs_distrib_fact, outputs_distrib_condition, hidden_out

	def init_hidden(self, batch_size):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_size)
		return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.device),
				autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.device),
				autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.device))

class Stmt_Extraction_Net(nn.Module):
	"""docstring for Tagger"""
	def __init__(self, wordEmbedding, word2ID, pos2ID, cap2ID, poscap2ID, embedding_dim, input_dim, hidden_dim, tagset_size_fact, tagset_size_condition, lm_config, postag_config, cap_config, poscap_config, device, seed, use_gate, enhance, lm_type='normal'):
		super(Stmt_Extraction_Net, self).__init__()

		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		self.hidden_dim = hidden_dim

		self.model_LSTM_encoder = LSTM_encoder(wordEmbedding, word2ID, pos2ID, cap2ID, poscap2ID, embedding_dim, input_dim, hidden_dim, num_layers = 1, bidirectional=True, lm_config=lm_config, postag_config=postag_config, cap_config=cap_config, poscap_config=poscap_config, device=device, use_gate=use_gate, lm_type=lm_type)

		self.model_LSTM_decoder = LSTM_decoder(hidden_dim * 2, hidden_dim * 2, tagset_size_fact, pos2ID, cap2ID, poscap2ID, lm_config, postag_config, cap_config, poscap_config, device, use_gate, enhance, lm_type=lm_type)
		# self.model_LSTM_decoder_condition = LSTM_decoder(hidden_dim * 2, hidden_dim * 2, tagset_size_condition, batch_size, pos2ID, cap2ID, poscap2ID, lm_config, postag_config, cap_config, poscap_config, device, use_gate)

	def forward(self, tuple_batch, batch_size):
		self.model_LSTM_encoder.hidden = self.model_LSTM_encoder.init_hidden(batch_size)
		encoder, posesEmb, capsEmb, lmsEmb, poscapsEmb = self.model_LSTM_encoder(tuple_batch)

		self.model_LSTM_decoder.hidden = self.model_LSTM_decoder.init_hidden(batch_size)
		fact_batch, condition_batch, _ , _, _ = self.model_LSTM_decoder(encoder, lmsEmb, posesEmb, capsEmb, poscapsEmb)

		return fact_batch, condition_batch
		
	def predict(self, tuple_batch):
		batch_size = len(tuple_batch[0])
		self.model_LSTM_encoder.hidden = self.model_LSTM_encoder.init_hidden(batch_size)
		encoder, posesEmb, capsEmb, lmsEmb, poscapsEmb = self.model_LSTM_encoder(tuple_batch)

		self.model_LSTM_decoder.hidden = self.model_LSTM_decoder.init_hidden(batch_size)
		fact_batch, condition_batch, _ , _, _ = self.model_LSTM_decoder(encoder, lmsEmb, posesEmb, capsEmb, poscapsEmb)

		return fact_batch, condition_batch

	def predict_distrib(self, tuple_batch, batch_size):
		self.model_LSTM_encoder.hidden = self.model_LSTM_encoder.init_hidden(batch_size)
		encoder, posesEmb, capsEmb, lmsEmb, poscapsEmb = self.model_LSTM_encoder(tuple_batch)

		self.model_LSTM_decoder.hidden = self.model_LSTM_decoder.init_hidden(batch_size)
		_, _, fact_batch, condition_batch, _ = self.model_LSTM_decoder(encoder, lmsEmb, posesEmb, capsEmb, poscapsEmb)

		return fact_batch, condition_batch

	def predict_hidden(self, tuple_batch, batch_size):
		self.model_LSTM_encoder.hidden = self.model_LSTM_encoder.init_hidden(batch_size)
		encoder, posesEmb, capsEmb, lmsEmb, poscapsEmb = self.model_LSTM_encoder(tuple_batch)

		self.model_LSTM_decoder.hidden = self.model_LSTM_decoder.init_hidden(batch_size)
		_, _, fact_batch, condition_batch, hidden_out = self.model_LSTM_decoder(encoder, lmsEmb, posesEmb, capsEmb, poscapsEmb)

		return hidden_out, hidden_out

class Ensemble_Net(nn.Module):
	"""docstring for Ensemble_Net"""
	def __init__(self, use_lm, use_postag, use_cap, tagset_size, device, seed):
		super(Ensemble_Net, self).__init__()

		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		self.use_lm = use_lm
		self.use_postag = use_postag
		self.use_cap = use_cap
		self.device = device

		self.w_lm_fact = nn.Parameter(torch.randn(tagset_size))
		self.w_pos_fact = nn.Parameter(torch.randn(tagset_size))
		self.w_cap_fact = nn.Parameter(torch.randn(tagset_size))
		self.b_fact = nn.Parameter(torch.randn(tagset_size))

		self.w_lm_cond = nn.Parameter(torch.randn(tagset_size))
		self.w_pos_cond = nn.Parameter(torch.randn(tagset_size))
		self.w_cap_cond = nn.Parameter(torch.randn(tagset_size))
		self.b_cond = nn.Parameter(torch.randn(tagset_size))

		for weight in self.parameters():
			print(type(weight), weight.size(), weight.requires_grad)

	def forward(self, tuple_batch):
		lm_input_batch, pos_input_batch, cap_input_batch = tuple_batch

		y_fact = 0
		y_cond = 0

		if self.use_lm:
			y_fact += lm_input_batch[0]*self.w_lm_fact
			y_cond += lm_input_batch[1]*self.w_lm_cond

		if self.use_postag:
			y_fact += pos_input_batch[0]*self.w_pos_fact
			y_cond += pos_input_batch[1]*self.w_pos_cond

		if self.use_cap:
			y_fact += cap_input_batch[0]*self.w_cap_fact
			y_cond += cap_input_batch[1]*self.w_cap_cond

		y_fact += self.b_fact
		y_cond += self.b_cond


		outputs_fact = F.log_softmax(y_fact, 2)
		outputs_cond = F.log_softmax(y_cond, 2)

		return outputs_fact, outputs_cond

class Ensemble_Net_new(nn.Module):
	"""docstring for Ensemble_Net_new"""
	def __init__(self, use_lm, use_postag, use_cap, hidden_size, tagset_size, device, seed):
		super(Ensemble_Net_new, self).__init__()

		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		self.use_lm = use_lm
		self.use_postag = use_postag
		self.use_cap = use_cap
		self.device = device

		self.w_lm_fact = nn.Parameter(torch.randn(hidden_size,tagset_size))
		self.w_pos_fact = nn.Parameter(torch.randn(hidden_size,tagset_size))
		self.w_cap_fact = nn.Parameter(torch.randn(hidden_size,tagset_size))
		self.b_fact = nn.Parameter(torch.randn(tagset_size))

		self.w_lm_cond = nn.Parameter(torch.randn(hidden_size,tagset_size))
		self.w_pos_cond = nn.Parameter(torch.randn(hidden_size,tagset_size))
		self.w_cap_cond = nn.Parameter(torch.randn(hidden_size,tagset_size))
		self.b_cond = nn.Parameter(torch.randn(tagset_size))

		for weight in self.parameters():
			print(type(weight), weight.size(), weight.requires_grad)

	def forward(self, tuple_batch):
		lm_input_batch, pos_input_batch, cap_input_batch = tuple_batch

		y_fact = 0
		y_cond = 0

		if self.use_lm:
			y_fact += torch.matmul(lm_input_batch[0], self.w_lm_fact)
			y_cond += torch.matmul(lm_input_batch[1], self.w_lm_cond)

		if self.use_postag:
			y_fact += torch.matmul(pos_input_batch[0], self.w_pos_fact)
			y_cond += torch.matmul(pos_input_batch[1], self.w_pos_cond)

		if self.use_cap:
			y_fact += torch.matmul(cap_input_batch[0], self.w_cap_fact)
			y_cond += torch.matmul(cap_input_batch[1], self.w_cap_cond)

		y_fact += self.b_fact
		y_cond += self.b_cond


		outputs_fact = F.log_softmax(y_fact, 2)
		outputs_cond = F.log_softmax(y_cond, 2)

		return outputs_fact, outputs_cond

class Two_Ensemble_Net(nn.Module):
	"""docstring for Two_Ensemble_Net"""
	def __init__(self, tagset_size, device, seed):
		super(Two_Ensemble_Net, self).__init__()

		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		self.device = device

		self.w_first_fact = nn.Parameter(torch.randn(tagset_size))
		self.w_second_fact = nn.Parameter(torch.randn(tagset_size))
		self.b_fact = nn.Parameter(torch.randn(tagset_size))

		self.w_first_cond = nn.Parameter(torch.randn(tagset_size))
		self.w_second_cond = nn.Parameter(torch.randn(tagset_size))
		self.b_cond = nn.Parameter(torch.randn(tagset_size))

		for weight in self.parameters():
			print(type(weight), weight.size(), weight.requires_grad)

	def forward(self, tuple_batch):
		first_input_batch, second_input_batch = tuple_batch

		y_fact = first_input_batch[0]*self.w_first_fact + second_input_batch[0]*self.w_second_fact + self.b_fact
		y_cond = first_input_batch[1]*self.w_first_cond + second_input_batch[1]*self.w_second_cond + self.b_cond

		outputs_fact = F.log_softmax(y_fact, 2)
		outputs_cond = F.log_softmax(y_cond, 2)

		return outputs_fact, outputs_cond
		
