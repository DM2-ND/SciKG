import sys, os, io
import numpy as np

from sklearn.utils import shuffle
from tools import *

parser = argparse.ArgumentParser(description='Conditional Statement Extraction')
parser.add_argument('--tuples', type=str, default='')
parser.add_argument('--out', type=str, default='')
args = parser.parse_args()

def write_meta_data(in_file, out_file):
	stmtid2tuples = dict()
	concept2num = dict()
	attribute2num = dict()
	fact_predicate2num = dict()
	cond_predicate2num = dict()
	cond2stmt = dict()
	fact2stmt = dict()
	head_fact2stmt = dict()
	tail_fact2stmt = dict()
	sent_nu = 0
	is_begin = True
	with open(in_file, 'r') as f:
		facts = []
		conditions = []
		sentence = ''
		stmt_id = ''
		for line in f:
			if line.startswith("===== "):
				if is_begin:
					stmt_id = line.strip().split('=====')[1].strip()
					sent_nu += 1
					is_begin = False
					continue
				assert sentence != ''
				# print(stmt_id, sentence, facts, conditions)
				if stmt_id in stmtid2tuples:
					print(stmt_id)
					stmt_id += '_0'
				stmtid2tuples[stmt_id] = dict()
				stmtid2tuples[stmt_id]['sentence'] = sentence
				stmtid2tuples[stmt_id]['facts'] = facts
				stmtid2tuples[stmt_id]['conditions'] = conditions
				sentence = ''
				facts = []
				conditions = []
				stmt_id = line.strip().split('=====')[1].strip()
				sent_nu += 1
				if sent_nu % 10000 == 0:
					print(sent_nu, 'done') 
			elif len(line.strip().split('\t')) == 1 and len(line.split(' ')) == 1:
				if not line.startswith('#'):
					is_begin = True
					continue
				assert sentence != ''
				if stmt_id in stmtid2tuples:
					print(stmt_id)
					stmt_id += '_0'
				stmtid2tuples[stmt_id] = dict()
				stmtid2tuples[stmt_id]['sentence'] = sentence
				stmtid2tuples[stmt_id]['facts'] = facts
				stmtid2tuples[stmt_id]['conditions'] = conditions
			elif line.startswith('f') and len(line.strip().split('\t')) == 4:
				tuple_str = line.strip()
				arr = tuple_str.split('\t')[1:]
				assert(len(arr) == 3)
				_tuple = ['']*5
				for i in [0,2]:
					if ':' in arr[i]:
						_arr = arr[i][1:-1].split(':')
						assert(len(_arr) == 2)
						_i = int(i*1.5)
						_tuple[_i] = '' if (_arr[0]=='NIL' or _arr[0].startswith('NIL#')) else _arr[0].split('#')[0].lower()
						_tuple[_i+1] = _arr[1].split('#')[0].lower()
						if _tuple[_i] not in concept2num:
							concept2num[_tuple[_i]] = 1
						else:
							concept2num[_tuple[_i]] += 1

						if _tuple[_i+1] not in attribute2num:
							attribute2num[_tuple[_i+1]] = 1
						else:
							attribute2num[_tuple[_i+1]] += 1
					else:
						_i = int(i*1.5)
						_tuple[_i] = '' if (arr[i]=='NIL' or arr[i].startswith('NIL#')) else arr[i].split('#')[0].lower()
						if _tuple[_i] not in concept2num:
							concept2num[_tuple[_i]] = 1
						else:
							concept2num[_tuple[_i]] += 1
				_tuple[2] = '' if (arr[1]=='NIL' or arr[1].startswith('NIL#')) else arr[1].split('#')[0].lower()
				if _tuple[2] not in fact_predicate2num:
					fact_predicate2num[_tuple[2]] = 1
				else:
					fact_predicate2num[_tuple[2]] += 1

				fact = '\n'+' '.join([':'.join(_tuple[:2]), _tuple[2], ':'.join(_tuple[3:])])+'\n'
				fact = fact.replace('\n: ', '').replace(' :\n', '').replace(':\n', '').replace(': ',' ').replace(' :', ' NIL:').replace('\n:', 'NIL:').replace('\n','')
				if fact not in fact2stmt:
					fact2stmt[fact] = set()
				fact2stmt[fact].add(stmt_id)

				fact = ' '.join(_tuple[:3]).strip()
				if fact not in head_fact2stmt:
					head_fact2stmt[fact] = set()
				head_fact2stmt[fact].add(stmt_id)

				fact = ' '.join(_tuple[2:]).strip()
				if fact not in tail_fact2stmt:
					tail_fact2stmt[fact] = set()
				tail_fact2stmt[fact].add(stmt_id)

				facts.append(_tuple)
			elif line.startswith('c') and len(line.strip().split('\t')) == 4:
				tuple_str = line.strip()
				arr = tuple_str.split('\t')[1:]
				assert(len(arr) == 3)
				_tuple = ['']*5
				for i in [0,2]:
					if ':' in arr[i]:
						_arr = arr[i][1:-1].split(':')
						assert(len(_arr) == 2)
						_i = int(i*1.5)
						_tuple[_i] = '' if (_arr[0]=='NIL' or _arr[0].startswith('NIL#')) else _arr[0].split('#')[0].lower()
						_tuple[_i+1] = _arr[1].split('#')[0].lower()
						if _tuple[_i] not in concept2num:
							concept2num[_tuple[_i]] = 1
						else:
							concept2num[_tuple[_i]] += 1

						if _tuple[_i+1] not in attribute2num:
							attribute2num[_tuple[_i+1]] = 1
						else:
							attribute2num[_tuple[_i+1]] += 1
					else:
						_i = int(i*1.5)
						_tuple[_i] = '' if (arr[i]=='NIL' or arr[i].startswith('NIL#')) else arr[i].split('#')[0].lower()
						if _tuple[_i] not in concept2num:
							concept2num[_tuple[_i]] = 1
						else:
							concept2num[_tuple[_i]] += 1
				_tuple[2] = '' if (arr[1]=='NIL' or arr[1].startswith('NIL#')) else arr[1].split('#')[0].lower()
				if _tuple[2] not in cond_predicate2num:
					cond_predicate2num[_tuple[2]] = 1
				else:
					cond_predicate2num[_tuple[2]] += 1

				cond = ' '.join(_tuple[2:]).strip()
				if cond not in cond2stmt:
					cond2stmt[cond] = set()
				cond2stmt[cond].add(stmt_id)

				conditions.append(_tuple)
			else:
				sentence = line.strip()
				assert len(sentence.split(' ')) > 1

	with open(out_file+'_stmtid2tuples.txt', 'w') as f:
		for stmt_id in stmtid2tuples:
			f.write(stmt_id+'\t')
			f.write(str(stmtid2tuples[stmt_id])+'\n')
		f.write('#'+str(sent_nu))
	print('writing done')

	with open(out_file+'_concept2num.txt', 'w') as f:
		tmp = sorted(concept2num.items(), key=lambda item:item[1], reverse=True)
		for e in tmp:
			if e[0].strip() == '':
				continue
			f.write(e[0]+'\t'+str(e[1])+'\n')
	print('writing done')

	with open(out_file+'_attribute2num.txt', 'w') as f:
		tmp = sorted(attribute2num.items(), key=lambda item:item[1], reverse=True)
		for e in tmp:
			if e[0].strip() == '':
				continue
			f.write(e[0]+'\t'+str(e[1])+'\n')
	print('writing done')

	with open(out_file+'_fact_predicate2num.txt', 'w') as f:
		tmp = sorted(fact_predicate2num.items(), key=lambda item:item[1], reverse=True)
		for e in tmp:
			f.write(e[0]+'\t'+str(e[1])+'\n')
	print('writing done')

	with open(out_file+'_cond_predicate2num.txt', 'w') as f:
		tmp = sorted(cond_predicate2num.items(), key=lambda item:item[1], reverse=True)
		for e in tmp:
			f.write(e[0]+'\t'+str(e[1])+'\n')
	print('writing done')

	with open(out_file+'_cond2stmt.txt', 'w') as f:
		tmp = sorted(cond2stmt.items(), key=lambda item:len(item[1]), reverse=True)
		for e in tmp:
			if len(e[0].strip().split(' ')) < 2:
				continue
			f.write(e[0]+'\t'+str(e[1])+'\n')
	print('writing done')

	with open(out_file+'_cond2num.txt', 'w') as f:
		tmp = sorted(cond2stmt.items(), key=lambda item:len(item[1]), reverse=True)
		for e in tmp:
			if len(e[0].strip().split(' ')) < 2:
				continue
			f.write(e[0]+'\t'+str(len(e[1]))+'\n')
	print('writing done')

	with open(out_file+'_fact2stmt.txt', 'w') as f:
		tmp = sorted(fact2stmt.items(), key=lambda item:len(item[1]), reverse=True)
		for e in tmp:
			if len(e[0].strip().split(' ')) < 2:
				continue
			f.write(e[0]+'\t'+str(e[1])+'\n')
	print('writing done')

	with open(out_file+'_fact2num.txt', 'w') as f:
		tmp = sorted(fact2stmt.items(), key=lambda item:len(item[1]), reverse=True)
		for e in tmp:
			if len(e[0].strip().split(' ')) < 2:
				continue
			f.write(e[0]+'\t'+str(len(e[1]))+'\n')
	print('writing done')

	with open(out_file+'_tail_fact2stmt.txt', 'w') as f:
		tmp = sorted(tail_fact2stmt.items(), key=lambda item:len(item[1]), reverse=True)
		for e in tmp:
			if len(e[0].strip().split(' ')) < 2:
				continue
			f.write(e[0]+'\t'+str(e[1])+'\n')
	print('writing done')

	with open(out_file+'_tail_fact2num.txt', 'w') as f:
		tmp = sorted(tail_fact2stmt.items(), key=lambda item:len(item[1]), reverse=True)
		for e in tmp:
			if len(e[0].strip().split(' ')) < 2:
				continue
			f.write(e[0]+'\t'+str(len(e[1]))+'\n')
	print('writing done')

	with open(out_file+'_head_fact2stmt.txt', 'w') as f:
		tmp = sorted(head_fact2stmt.items(), key=lambda item:len(item[1]), reverse=True)
		for e in tmp:
			if len(e[0].strip().split(' ')) < 2:
				continue
			f.write(e[0]+'\t'+str(e[1])+'\n')
	print('writing done')

	with open(out_file+'_head_fact2num.txt', 'w') as f:
		tmp = sorted(head_fact2stmt.items(), key=lambda item:len(item[1]), reverse=True)
		for e in tmp:
			if len(e[0].strip().split(' ')) < 2:
				continue
			f.write(e[0]+'\t'+str(len(e[1]))+'\n')
	print('writing done')


if __name__ == '__main__':
	write_meta_data(args.tuples, args.out)

