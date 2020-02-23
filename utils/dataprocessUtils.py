import torch
import json
import config.args as args
import random
from collections import Counter
from tqdm import tqdm
from utils.loggerUtils import init_logger

logger = init_logger("bert_ner",logging_path=args.log_path)
class Normialize:
	@staticmethod
	def ChineseChar(char):
		'''中文字符 转化为 英文字符
		:param char:
		:return: normalized_char
		'''
		char_replace = {
			8220: "\"",
			8221: "\"",
			8212: "-",
			8230: "...",
			8216: "＇",
			8217: "＇",
		}

		if ord(char) in char_replace:
			return char_replace[ord(char)]
		else:
			return char

class InputExample(object):

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""创建一个输入实例
		Args:
			guid: 每个example拥有唯一的id
			text_a: 第一个句子的原始文本，一般对于文本分类来说，只需要text_a
			text_b: 第二个句子的原始文本，在句子对的任务中才有，分类问题中为None
			label: example对应的标签，对于训练集和验证集应非None，测试集为None
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label
class InputFeature(object):
	def __init__(self, input_ids, input_mask, segment_ids, label_id, output_mask):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		self.output_mask = output_mask
class DataProcessor(object):

	"""数据预处理的基类，自定义的MyPro继承该类"""

	def get_train_examples(self):
		"""读取训练集 Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self):
		"""读取验证集 Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""读取标签 Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_json(cls, input_file):
		with open(input_file, "r", encoding='utf-8') as fr:
			lines = []
			for line in fr:
				_line = line.strip('\n')
				lines.append(_line)
			return lines

def convert_tokens_to_features(tokens_a,labels,max_seq_length,tokenizer,label_list,sub_vocab,segment_id = 0):

	if labels == None:
		labels = " ".join(["-1"] * len(tokens_a))
	# ----------------对text_a按max_seq1_length切分-----------------

	labels = labels.split()
	assert len(tokens_a) == len(labels)
	if len(tokens_a) == 0 or len(labels) == 0:
		return 0
	if len(tokens_a) > max_seq_length - 2:
		tokens_a = tokens_a[:(max_seq_length - 2)]
		labels = labels[:(max_seq_length - 2)]

	# ----------------token to Inputids--------------
	## 句子首尾加入标示符
	tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
	segment_ids = [segment_id] * len(tokens)
	## 词转换成数字
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	input_mask = [1] * len(input_ids)
	output_mask = [0 if sub_vocab.get(t) is not None else 1 for t in tokens_a]
	output_mask = [0] + output_mask + [0]

	# ---------------训练时处理target----------------
	## Notes: label_id中不包括[CLS]和[SEP]
	if label_list != None:
		label_map = {label: i for i, label in enumerate(label_list)}
		label_id = [label_map[l] for l in labels]
		label_id = [-1] + label_id + [-1]
	else:
		label_id = [-1] * len(input_ids)

	label_padding = [-1] * (max_seq_length - len(label_id))

	# ----------------训练时加上padding-----------------
	padding = [0] * (max_seq_length - len(input_ids))
	input_ids += padding
	input_mask += padding
	segment_ids += [segment_id] * (max_seq_length - len(segment_ids))
	output_mask += padding
	label_id += label_padding

	assert len(input_ids) == max_seq_length
	assert len(input_mask) == max_seq_length
	assert len(segment_ids) == max_seq_length
	assert len(output_mask) == max_seq_length

	return tokens,label_id,input_ids,input_mask,segment_ids,output_mask
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,show_example,max_seq2_length = 10):
	'''

	:param examples:
	:param label_list: 如果没有对应的序列标签,那么设置label_list = None
	:param max_seq_length:
	:param tokenizer:
	:return:
	'''
	# 标签转换为数字

	sub_vocab = {}
	with open(args.VOCAB_FILE, 'r', encoding="utf-8") as fr:
		for line in fr:
			_line = line.strip('\n')
			if "##" in _line and sub_vocab.get(_line) is None:
				sub_vocab[_line] = 1

	features = []
	for ex_index, example in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)
		labels = example.label
		value = convert_tokens_to_features(
			tokens_a = tokens_a,
			labels = labels,
			max_seq_length = max_seq_length,
			tokenizer = tokenizer,
			label_list = label_list,
			sub_vocab = sub_vocab,
			segment_id =0
		)
		if value != 0:
			tokens, label_id, input_ids, input_mask, segment_ids, output_mask = value
		else:
			continue

		if example.text_b != None:
			tokens_b = tokenizer.tokenize(example.text_b)
			value = convert_tokens_to_features(
				tokens_a=tokens_b,
				labels=None,
				max_seq_length=max_seq2_length,
				tokenizer=tokenizer,
				label_list=None,
				sub_vocab=sub_vocab,
				segment_id=1
			)
			if value != 0:
				tokens_2, label_id_2, input_ids_2, input_mask_2, segment_ids_2, output_mask_2 = value
			else:
				continue
			tokens += tokens_2[1:]
			label_id += label_id_2[1:]
			input_ids += input_ids_2[1:]
			input_mask += input_mask_2[1:]
			segment_ids += segment_ids_2[1:]
			output_mask += output_mask_2[1:]
		# ----------------处理后结果-------------------------
		# for example, in the case of max_seq_length=10:
		# raw_data:          春 秋 忽 代 谢le
		# token:       [CLS] 春 秋 忽 代 谢 ##le [SEP]
		# input_ids:     101 2  12 13 16 14 15   102   0 0 0
		# input_mask:      1 1  1  1  1  1   1     1   0 0 0
		# label_id:          T  T  O  O  O
		# output_mask:     0 1  1  1  1  1   0     0   0 0 0
		# --------------看结果是否合理------------------------
		if ex_index < 1 and show_example :
			logger.info("-----------------Example-----------------")
			logger.info("guid: %s" % (example.guid))
			logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info("segment ids: %s " % " ".join([str(x) for x in segment_ids]))
			logger.info("label_ids : %s " % " ".join([str(x) for x in label_id]))
			logger.info("output_mask: %s " % " ".join([str(x) for x in output_mask]))
		# ----------------------------------------------------

		feature = InputFeature(input_ids=input_ids,
							   input_mask=input_mask,
							   segment_ids=segment_ids,
							   label_id=label_id,
							   output_mask=output_mask)
		features.append(feature)

	return features

def train_val_split(X, y, valid_size=0.2, random_state=2018, shuffle=True):
	"""
	训练集验证集分割
	:param X: sentences
	:param y: labels
	:param random_state: 随机种子
	"""
	logger.info('Train val split')

	data = []
	for data_x, data_y in tqdm(zip(X, y), desc='Merge'):
		data.append((data_x, data_y))
	del X, y

	N = len(data)
	test_size = int(N * valid_size)

	if shuffle:
		random.seed(random_state)
		random.shuffle(data)

	valid = data[:test_size]
	train = data[test_size:]

	return train, valid
def sent2char(line):
	"""
	句子处理成单词
	:param line: 原始行
	:return: 单词， 标签
	"""
	res = line.strip('\n').split()
	return res
