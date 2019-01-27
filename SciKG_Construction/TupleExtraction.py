import sys, os, io
import numpy as np

from sklearn.utils import shuffle
from tools import *

parser = argparse.ArgumentParser(description='Conditional Statement Extraction')
parser.add_argument('--udata', type=str, default='./udata/stmts-demo-unlabeled-pubmed',
					help='location of the unlabeled data')
parser.add_argument('--postfix', type=str, default='t_lym')
parser.add_argument('--key', type=str, default='')
parser.add_argument('--shuffle', action='store_true',
					help='shuffle the instances')
args = parser.parse_args()

def Extraction(file_name, json_name, dataCenter, keyword, is_shuffle):
	stmtid2tuples = dict()

	nodes = []
	links = []
	graph = dict()
	graph['nodes'] = nodes
	graph['links'] = links
	node_id = 0
	node2id = dict()

	nodes.append({
		"id": node_id,
		"category": 1, 
		"type": "meta", 
		"name": "__stmt__"})
	meta_stmt_id = node_id
	node_id += 1

	nodes.append({
		"id": node_id,
		"category": 2, 
		"type": "meta", 
		"name": "__predicate__"})
	meta_predicate_id = node_id
	node_id += 1

	nodes.append({
		"id": node_id,
		"category": 3, 
		"type": "meta", 
		"name": "__concept_attribute__"})
	meta_concept_attribute_id = node_id
	node_id += 1

	count = 0
	f_id = 1
	c_id = 1

	fact_count = 0
	cond_count = 0
	attribute_set = set()

	fo = open(file_name, 'w')
	instance_list = dataCenter.instance_EVAL
	print(len(instance_list))
	if is_shuffle:
		instance_list = shuffle(instance_list)
	for i in range(len(instance_list)):
		sentence = instance_list[i].SENTENCE
		if keyword not in ' '.join(sentence):
			continue

		fact_tags, cond_tags = instance_list[i].OUT

		is_discarded_fact, fact_predicate_set = is_discarded(fact_tags)
		is_discarded_cond, cond_predicate_set = is_discarded(cond_tags)

		if is_discarded_fact or is_discarded_cond:
			continue
		if fact_predicate_set & cond_predicate_set != set():
			continue

		fact_tags, corrected_fact = smooth_tag_sequence(fact_tags)
		cond_tags, corrected_cond = smooth_tag_sequence(cond_tags)
		if corrected_fact or corrected_cond:
			continue

		stmt_str = str(instance_list[i].paper_id)+' stmt'+str(instance_list[i].stmt_id)
		fo.write('===== '+stmt_str+' =====\n')
		fo.write('%s\n' % ' '.join(sentence))
		nodes.append({
			"name": stmt_str, 
			"label": "S",
			"id": node_id,
			"category": 1, 
			"rate": 1.0, 
			"type": "normal",
			"sent": ' '.join(instance_list[i].SENTENCE)})
		stmtNode_id = node_id
		links.append({
			"source": meta_stmt_id,
			"target": stmtNode_id,
			"type": "metalink"})
		node_id += 1
		# print instance_list[i].paper_id, instance_list[i].stmt_id
		fact_tuples = new_post_decoder(instance_list[i].SENTENCE, fact_tags)
		condition_tuples = new_post_decoder(instance_list[i].SENTENCE, cond_tags)
		for fact_tuple in fact_tuples:
			_1C_id = -1
			_1A_id = -1
			predicate_id = -1
			_3C_id = -1
			_3A_id = -1

			assert len(fact_tuple) == 5
			_1C, _1A, predicate, _3C, _3A = fact_tuple
			if predicate != 'NIL':
				nodes.append({
					"name": predicate[0], 
					"label": "P",
					"id": node_id,
					"category": 2,
					"rate": 1.0, 
					"type": "normal",
					"tuples": ""})
				predicate_id = node_id
				node_id += 1
				predicate = predicate[0]+'#'+str(predicate[1])
			else:
				nodes.append({
					"name": 'null', 
					"label": "P",
					"id": node_id,
					"category": 2,
					"rate": 1.0, 
					"type": "normal",
					"tuples": ""})
				predicate_id = node_id
				node_id += 1
			links.append({
				"source": meta_predicate_id,
				"target": predicate_id,
				"type": "metalink"})
			links.append({
				"source": stmtNode_id,
				"target": predicate_id,
				"type": "outer",
				"name": "FACT",
				"category": 101})
			nodes[stmtNode_id]['rate'] += 1
			fact_count += 1

			# subject nodes generation
			if _1C != 'NIL':
				if _1C[0] not in node2id:
					nodes.append({
						"name": _1C[0], 
						"label": "C",
						"id": node_id,
						"category": 3,
						"rate": 1.0, 
						"type": "normal",
						"tuples": ""})
					_1C_id = node_id
					node2id[_1C[0]] = _1C_id
					node_id += 1
				else:
					_1C_id = node2id[_1C[0]]
					nodes[_1C_id]["rate"] += 1.0
				_1C = _1C[0]+'#'+str(_1C[1])
				links.append({
					"source": meta_concept_attribute_id,
					"target": _1C_id,
					"type": "metalink"})

			if _1A == 'NIL':
				subject = _1C
				subject2 = _1C.split('#')[0].replace('_', ' ')
				if _1C != 'NIL':
					links.append({
						"source": predicate_id,
						"target": _1C_id,
						"type": "outer",
						"name": "SUBJECT",
						"category": 103})
			else:
				attribute_set.add(_1A[0])
				nodes.append({
					"name": _1A[0], 
					"label": "A",
					"id": node_id,
					"category": 4,
					"rate": 1.0, 
					"type": "normal",
					"tuples": ""})
				_1A_id = node_id
				node_id += 1
				links.append({
					"source": meta_concept_attribute_id,
					"target": _1A_id,
					"type": "metalink"})
				links.append({
					"source": predicate_id,
					"target": _1A_id,
					"type": "outer",
					"name": "SUBJECT",
					"category": 103})
				_1A = _1A[0]+'#'+str(_1A[1])
				subject = '{'+_1C+':'+_1A+'}'
				subject2 = '{'+_1C.split('#')[0].replace('_',' ')+':'+_1A.split('#')[0].replace('_',' ')+'}'

				if _1C != 'NIL':
					links.append({
						"source": _1C_id,
						"target": _1A_id,
						"type": "inner",
						"name": "ATTRIBUTE",
						"category": 104})

			# object nodes generation
			if _3C != 'NIL':
				if _3C[0] not in node2id:
					nodes.append({
						"name": _3C[0], 
						"label": "C",
						"id": node_id,
						"category": 3,
						"rate": 1.0, 
						"type": "normal",
						"tuples": ""})
					_3C_id = node_id
					node2id[_3C[0]] = _3C_id
					node_id += 1
				else:
					_3C_id = node2id[_3C[0]]
					nodes[_3C_id]["rate"] += 1.0
				_3C = _3C[0]+'#'+str(_3C[1])
				links.append({
					"source": meta_concept_attribute_id,
					"target": _3C_id,
					"type": "metalink"})

			if _3A == 'NIL':
				_object = _3C
				_object2 = _3C.split('#')[0].replace('_', ' ')
				if _3C != 'NIL':
					links.append({
						"source": predicate_id,
						"target": _3C_id,
						"type": "outer",
						"name": "OBJECT",
						"category": 103})
			else:
				attribute_set.add(_3A[0])
				nodes.append({
					"name": _3A[0], 
					"label": "A",
					"id": node_id,
					"category": 4,
					"rate": 1.0, 
					"type": "normal",
					"tuples": ""})
				_3A_id = node_id
				node_id += 1
				links.append({
					"source": meta_concept_attribute_id,
					"target": _3A_id,
					"type": "metalink"})
				links.append({
					"source": predicate_id,
					"target": _3A_id,
					"type": "outer",
					"name": "OBJECT",
					"category": 103})
				_3A = _3A[0]+'#'+str(_3A[1])
				_object = '{'+_3C+':'+_3A+'}'
				_object2 = '{'+_3C.split('#')[0].replace('_', ' ')+': '+_3A.split('#')[0].replace('_', ' ')+'}'

				if _3C != 'NIL':
					links.append({
						"source": _3C_id,
						"target": _3A_id,
						"type": "inner",
						"name": "ATTRIBUTE",
						"category": 104})

			fo.write('f%d\t%s\t%s\t%s\n' % (f_id,subject,predicate,_object))
			for _id in [predicate_id, _1C_id, _1A_id, _3C_id, _3A_id]:
				if _id != -1:
					nodes[_id]["tuples"] += ('f[%s, %s, %s]<br/>' % (subject2,predicate.split('#')[0].replace('_', ' '),_object2))
			f_id += 1
		for condition_tuple in condition_tuples:
			_1C_id = -1
			_1A_id = -1
			predicate_id = -1
			_3C_id = -1
			_3A_id = -1

			assert len(condition_tuple) == 5
			_1C, _1A, predicate, _3C, _3A = condition_tuple
			if predicate != 'NIL':
				nodes.append({
					"name": predicate[0], 
					"label": "P",
					"id": node_id,
					"category": 2,
					"rate": 1.0, 
					"type": "normal",
					"tuples": ""})
				predicate_id = node_id
				node_id += 1
				predicate = predicate[0]+'#'+str(predicate[1])
			else:
				nodes.append({
					"name": 'null', 
					"label": "P",
					"id": node_id,
					"category": 2,
					"rate": 1.0, 
					"type": "normal",
					"tuples": ""})
				predicate_id = node_id
				node_id += 1
			links.append({
				"source": meta_predicate_id,
				"target": predicate_id,
				"type": "metalink"})
			links.append({
				"source": stmtNode_id,
				"target": predicate_id,
				"type": "outer",
				"name": "CONDITION",
				"category": 102})
			cond_count += 1

			# subject nodes generation
			if _1C != 'NIL':
				if _1C[0] not in node2id:
					nodes.append({
						"name": _1C[0], 
						"label": "C",
						"id": node_id,
						"category": 3,
						"rate": 1.0, 
						"type": "normal",
						"tuples": ""})
					_1C_id = node_id
					node2id[_1C[0]] = _1C_id
					node_id += 1
				else:
					_1C_id = node2id[_1C[0]]
					nodes[_1C_id]["rate"] += 1.0
				_1C = _1C[0]+'#'+str(_1C[1])
				links.append({
					"source": meta_concept_attribute_id,
					"target": _1C_id,
					"type": "metalink"})

			if _1A == 'NIL':
				subject = _1C
				subject2 = _1C.split('#')[0].replace('_', ' ')
				if _1C != 'NIL':
					links.append({
						"source": predicate_id,
						"target": _1C_id,
						"type": "outer",
						"name": "SUBJECT",
						"category": 103})
			else:
				attribute_set.add(_1A[0])
				nodes.append({
					"name": _1A[0], 
					"label": "A",
					"id": node_id,
					"category": 4,
					"rate": 1.0, 
					"type": "normal",
					"tuples": ""})
				_1A_id = node_id
				node_id += 1
				links.append({
					"source": meta_concept_attribute_id,
					"target": _1A_id,
					"type": "metalink"})
				links.append({
					"source": predicate_id,
					"target": _1A_id,
					"type": "outer",
					"name": "SUBJECT",
					"category": 103})
				_1A = _1A[0]+'#'+str(_1A[1])
				subject = '{'+_1C+':'+_1A+'}'
				subject2 = '{'+_1C.split('#')[0].replace('_',' ')+':'+_1A.split('#')[0].replace('_',' ')+'}'

				if _1C != 'NIL':
					links.append({
						"source": _1C_id,
						"target": _1A_id,
						"type": "inner",
						"name": "ATTRIBUTE",
						"category": 104})

			# object nodes generation
			if _3C != 'NIL':
				if _3C[0] not in node2id:
					nodes.append({
						"name": _3C[0], 
						"label": "C",
						"id": node_id,
						"category": 3,
						"rate": 1.0, 
						"type": "normal",
						"tuples": ""})
					_3C_id = node_id
					node2id[_3C[0]] = _3C_id
					node_id += 1
				else:
					_3C_id = node2id[_3C[0]]
					nodes[_3C_id]["rate"] += 1.0
				_3C = _3C[0]+'#'+str(_3C[1])
				links.append({
					"source": meta_concept_attribute_id,
					"target": _3C_id,
					"type": "metalink"})

			if _3A == 'NIL':
				_object = _3C
				_object2 = _3C.split('#')[0].replace('_', ' ')
				if _3C != 'NIL':
					links.append({
						"source": predicate_id,
						"target": _3C_id,
						"type": "outer",
						"name": "OBJECT",
						"category": 103})
			else:
				attribute_set.add(_3A[0])
				nodes.append({
					"name": _3A[0], 
					"label": "A",
					"id": node_id,
					"category": 4,
					"rate": 1.0, 
					"type": "normal",
					"tuples": ""})
				_3A_id = node_id
				node_id += 1
				links.append({
					"source": meta_concept_attribute_id,
					"target": _3A_id,
					"type": "metalink"})
				links.append({
					"source": predicate_id,
					"target": _3A_id,
					"type": "outer",
					"name": "OBJECT",
					"category": 103})
				_3A = _3A[0]+'#'+str(_3A[1])
				_object = '{'+_3C+':'+_3A+'}'
				_object2 = '{'+_3C.split('#')[0].replace('_', ' ')+': '+_3A.split('#')[0].replace('_', ' ')+'}'

				if _3C != 'NIL':
					links.append({
						"source": _3C_id,
						"target": _3A_id,
						"type": "inner",
						"name": "ATTRIBUTE",
						"category": 104})

			fo.write('c%d\t%s\t%s\t%s\n' % (c_id,subject,predicate,_object))
			for _id in [predicate_id, _1C_id, _1A_id, _3C_id, _3A_id]:
				if _id != -1:
					nodes[_id]["tuples"] += ('c[%s, %s, %s]<br/>' % (subject2,predicate.split('#')[0].replace('_', ' '),_object2))
			c_id += 1
		count += 1
		if count % 10000 == 0:
			print(count,f_id-1,c_id-1, 'done')

	fo.write('#'+str(count)+'\n')
	fo.write('nodes:%d,facts:%d,conditions:%d,concepts:%d,attributes:%d\n' % (node_id, fact_count, cond_count, len(node2id), len(attribute_set)))
	fo.close()
	print(node_id, 'nodes', fact_count, 'facts', cond_count, 'conditions')
	#with open(json_name, 'w') as fw:
		#json.dump(graph, fw)


if __name__ == '__main__':
	dataCenter = DataCenter()
	dataCenter.loading_dataset(None, args.udata)
	file_name = './data/tuples_'+args.postfix+'.txt'
	json_name = './data/json_'+args.postfix+'.json'
	Extraction(file_name, json_name, dataCenter, args.key, args.shuffle)

