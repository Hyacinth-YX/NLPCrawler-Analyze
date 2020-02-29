'''
用于语料库处理的一些公共库。
包括对一些语料库的构建
@author : ljc
2020-02-20
'''
import sys
sys.path.append("..")
from utils.loggerUtils import init_logger
import os
import json
import re
from bs4 import BeautifulSoup
import config
import numpy as np
from loguru import logger as ConsoleLogger
corpus_param = config.Corpus_Path()
p = config.ProjectConfiguration()
logger = init_logger("corpus process",p.log_path)
def split(seq, label):
	'''按句号分割'''
	sentences = []
	labels = []
	this_sentence = []
	this_label = []
	for i in range(len(seq)):
		if seq[i] == "。":
			sentences.append(this_sentence)
			labels.append(this_label)
			this_sentence = []
			this_label = []
		this_sentence.append(seq[i])
		this_label.append(label[i])
	return sentences, labels
def extract(func):

	def inner(*args, **kwargs):

		self = args[0]

		results = []

		for extraced in func(*args, **kwargs):

			span = extraced["span"]

			text = extraced["text"]

			labels = []

			if extraced['keyWord'] == "是":

				keyword_index = text.index("是")

				object_span = [span[0] + keyword_index + 1, span[1]]

				subject_span = [span[0], keyword_index - 1 + span[0]]

				keyword_span = [span[0] + keyword_index, span[0] + keyword_index]

				labels.append({"span": keyword_span, "label": extraced['event']})

				if subject_span[1] >= subject_span[0]:
					labels.append({"span": subject_span, "label": "主体"})

				if object_span[1] >= object_span[0]:
					labels.append({"span": object_span, "label": "客体"})

			elif extraced['event'] == "时间":

				labels.append({"span": span, "label": "时间"})

			else:

				object_span = [span[0] + len(extraced['keyWord']), span[1]]

				keyword_span = [span[0], span[0] + len(extraced['keyWord']) - 1]

				labels.append({"span": keyword_span, "label": extraced['event']})

				if object_span[1] >= object_span[0]:
					labels.append({"span": object_span, "label": "客体"})

			for label in labels:

				label["text"] = ""

				for i in range(label["span"][0], label["span"][1] + 1):

					try:

						label["text"] += self.corpus[i]

					except:

						continue
			extraced["labels"] = labels

			results.append(extraced)

		return results

	return inner
class Extract_Information_By_Re:
	valid_char = "[\w\"\[\]\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]"
	valid_char_except_comma = "[\w\"\[\]\uff1b\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b]"
	valid_char_except_comma_question = "[\w\"\[\]\uff1b\uff1a\u201c\u201d\uff08\uff09\u3001\u300a\u300b]"
	labels = ["影响", "原因", "结果", "手段", "定义", "属性", "条件", "时间", "主体", "客体", "O"]
	def __init__(self, seq):

		self.seq = seq

		self.corpus = "".join(seq)

		self.index2index = [0 for i in range(len(self.corpus))]

		self.labels = ["O" for i in range(len(self.seq))]

		count = 0

		for i in range(len(seq)):

			this_len = len(seq[i])

			for j in range(count, this_len + count):

				self.index2index[j] = i

			count += this_len

	@extract
	def Find_Inluence(self):

		keywords = ["影响", "有利", "不利"]

		for keyword in keywords:

			index = re.finditer(r"{}{}".format(keyword, Extract_Information_By_Re.valid_char_except_comma + "{3,}"),
								self.corpus)

			for item in index:
				span = item.span()

				text = self.corpus[span[0]:span[1]]

				yield {"event": "影响", "keyWord": keyword, "span": span, "text": text}

	@extract
	def Find_Cause(self):

		keywords = ["随着", "因为", "由于", "为了"]

		for keyword in keywords:

			pattern = "{}{}*".format(keyword, Extract_Information_By_Re.valid_char_except_comma)

			index = re.finditer(r"{}".format(pattern), self.corpus)

			for item in index:
				span = item.span()

				text = self.corpus[span[0]:span[1]]

				yield {"event": "原因", "keyWord": keyword, "span": span, "text": text}

	@extract
	def Find_Result(self):

		keywords = ["形成", "导致", "因而", "因此", "使得"]

		for keyword in keywords:

			pattern = "{}{}*".format(keyword, Extract_Information_By_Re.valid_char)

			index = re.finditer(r"{}".format(pattern), self.corpus)

			for item in index:
				span = item.span()

				text = self.corpus[span[0]:span[1]]

				yield {"event": "结果", "keyWord": keyword, "span": span, "text": text}

	@extract
	def Find_Method(self):

		keywords = ["通过", "利用", "凭借", "借助", "开展了"]

		for keyword in keywords:

			pattern = "{}{}*".format(keyword, Extract_Information_By_Re.valid_char_except_comma_question)

			index = re.finditer(r"{}".format(pattern), self.corpus)

			for item in index:
				span = item.span()

				text = self.corpus[span[0]:span[1]]

				yield {"event": "手段", "keyWord": keyword, "span": span, "text": text}

	@extract
	def Find_DeFinition(self):

		keywords = ["是"]

		for keyword in keywords:

			pattern = "{}*{}{}*".format(Extract_Information_By_Re.valid_char_except_comma_question, keyword,
										Extract_Information_By_Re.valid_char_except_comma_question)

			index = re.finditer(r"{}".format(pattern), self.corpus)

			for item in index:

				span = item.span()

				result = self.corpus[span[0]:span[1]]

				if "但是" not in result and "如何" not in result and "什么" not in result and "于是" not in result and "种是":
					text = self.corpus[span[0]:span[1]]

					yield {"event": "定义", "keyWord": keyword, "span": span, "text": text}

	@extract
	def Find_Property(self):

		keywords = ["具有", "属于", "可以", "作为", "形如", "如同", "称为", "提供", "带来", "始于", "体现"]

		for keyword in keywords:

			pattern = "{}{}*".format(keyword, Extract_Information_By_Re.valid_char)

			index = re.finditer(r"{}".format(pattern), self.corpus)

			for item in index:
				span = item.span()

				text = self.corpus[span[0]:span[1]]

				yield {"event": "属性", "keyWord": keyword, "span": span, "text": text}

	@extract
	def Find_Time(self):

		pattern1 = "[上世纪一二三四五六七八九十代]*[0-9]{1,10}[时月年世纪代前初后]+[0-9一二三四五六七八九十月日]*"

		patterns = [pattern1]

		for pattern in patterns:

			index = re.finditer(r"{}".format(pattern), self.corpus)

			for item in index:
				span = item.span()

				text = self.corpus[span[0]:span[1]]

				yield {"event": "时间", "keyWord": None, "span": span, "text": text}

	@extract
	def Find_Condition(self):
		keywords = ["只要", "条件", "直到", "需要"]

		for keyword in keywords:

			pattern = "{}{}*".format(keyword, Extract_Information_By_Re.valid_char)

			index = re.finditer(r"{}".format(pattern), self.corpus)

			for item in index:
				span = item.span()

				text = self.corpus[span[0]:span[1]]

				yield {"event": "条件", "keyWord": keyword, "span": span, "text": text}

	def Find_All(self):

		func_pointer = [self.Find_DeFinition, self.Find_Cause, self.Find_Method, self.Find_Inluence, self.Find_Property,
						self.Find_Time, self.Find_Result, self.Find_Condition]

		results = []

		for pointer in func_pointer:
			results += pointer()

		return results

	def Createlabels(self):

		results = self.Find_All()

		for result in results:

			labels = result["labels"]

			for label in labels:

				span_indexs = list(range(label["span"][0], label["span"][1] + 1))

				seq_indexs = []

				for index in span_indexs:

					try:

						seq_indexs.append(self.index2index[index])

					except:

						continue
				for seq_index in seq_indexs:
					self.labels[seq_index] = label["label"]

		return self.labels

	@staticmethod
	def label2onehot(label):

		index = Extract_Information_By_Re.labels.index(label)

		vec = np.zeros((len(Extract_Information_By_Re.labels), 1))

		vec[index][0] = 1

		return vec

	@staticmethod
	def labels2index(labels):

		return list(map(lambda label: Extract_Information_By_Re.labels.index(label), labels))

class CTB:
	def __init__(self,version = 7, load_pos = False,load_parsetree = False,load_cpb = False):
		CTB_dir = corpus_param.CTB7_dir if version == 7 else corpus_param.CTB9_dir
		self.parse_tree_dir = CTB_dir + "bracketed/"
		self.postag_dir = CTB_dir + "postagged/"
		self.raw_dir = CTB_dir + "raw/"
		self.segement_dir = CTB_dir + "segmented/"
		load_parsetree = max(load_parsetree,load_cpb)
		ConsoleLogger.debug("加载CTB数据集原文内容...")
		filenames = list(map(lambda x:x[:-4],os.listdir(self.raw_dir)))
		self.files = dict([filename,ctb_file(filename,self.segement_dir)] for filename in filenames)
		additional = {
			self.load_pos : load_pos,
			self.load_parsetree: load_parsetree,
			self.load_cpb: load_cpb,
		}
		for selection in additional:

			if additional[selection]:

				selection()

		ConsoleLogger.info("读取CTP success!")
	@staticmethod
	def AddSemanticLabel_toCTBfile(file , ):
		pass
	def load_cpb(self):

		'''
		将所有的ctb 文件 添加语意标注标签
		:return:
		'''
		ConsoleLogger.debug("加载CPB语义标签...")
		with open(corpus_param.CPB_verb_file,encoding = "utf-8-sig") as f:
			verb_predict_data = f.read().split("\n")
		last_field = None
		count = 0
		error = 0
		valid = 0
		for idx,verb_predict in enumerate(verb_predict_data):
			fields_values = verb_predict.split(" ")
			try:
				file_name,sent_idx,predict_idx,tagger,word_frameid = fields_values[:5]
			except:
				continue
			sent_idx = int(sent_idx)
			if file_name == "chtb_0112.nw" :
				if  sent_idx >= 10 or sent_idx in [3,4,5,6,7,8] or (sent_idx == 9 and (int(predict_idx) >= 14 or "取决于"in word_frameid) ):
					sent_idx = sent_idx - 1
			if file_name == "chtb_0139.nw":
				if sent_idx >= 10 or sent_idx in [4,5,6,7,8,9]:
					sent_idx = sent_idx - 1
			if file_name == "chtb_0307.nw":
				if sent_idx in [1]:
					sent_idx = sent_idx - 1
				if sent_idx >= 8 or sent_idx in [4,5,6,7]:
					sent_idx  = sent_idx - 2
			if  file_name == "chtb_0437.nw":
				if  sent_idx >= 3:
						sent_idx -=1
			if file_name == "chtb_0672.nw":
				if sent_idx >= 10 or sent_idx >= 3:
					sent_idx -= 1
			if file_name == "chtb_0733.nw":
				if sent_idx >= 4 :
					sent_idx -= 1
			if file_name == "chtb_0787.nw":
				if sent_idx >= 3 :
					sent_idx -= 1
			if file_name == "chtb_0792.nw" or file_name == "chtb_0793.nw":
				if sent_idx >= 3 :
					sent_idx -= 1
			if file_name == "chtb_0794.nw"  :
				if sent_idx >= 2 :
					sent_idx -= 1
			if file_name == "chtb_0828.nw"  or file_name == "chtb_0845.nw" or file_name == "chtb_0855.nw" :
				if sent_idx >= 2 :
					sent_idx -= 1
			if file_name == "chtb_0877.nw":
				if sent_idx >= 5:
					sent_idx -= 2
			if file_name == "chtb_1042.mz":
				if sent_idx >= 100:
					sent_idx -=1
			predict_idx = int(predict_idx)
			predict,frame_id = word_frameid.split(".")
			def error_check():
				print("No predict")
				print("cpb:",predict,"terminal:",terminal)
				print(file_name,sent_idx,predict_idx)
				for i in range(sent_idx -2 ,sent_idx + 2):
					print("--------------",i)
					print([str(idx) + "." + label for idx,label in enumerate(self.files[file_name].sents[i]["parseTree"].terminals)])
				exit()
			try:
				file_sent = self.files[file_name].sents[sent_idx]
				sent_terminals = [str(idx) + "." + label for idx, label in enumerate(file_sent["parseTree"].terminals)]
				terminal = sent_terminals[predict_idx]
				if predict in terminal:
					args = dict(["-".join(arg.split("-")[1:]), arg.split("-")[0]] for arg in fields_values[6:])
					self.files[file_name].sents[sent_idx]["PA-structure"].append(
						{
							"predict": predict_idx,
							"predict_type": "verb",
							"args": args
						}
					)
					valid += 1
				else:
					if last_field[-1].split(".")[0] == predict:
						continue
					error += 1
					#error_check()
			except:
				error += 1
				#error_check()
			last_field = [file_name, sent_idx, tagger, word_frameid]
			count += 1
		ConsoleLogger.debug("读入 {}/{} Verb-Argument".format(valid,count))
		ConsoleLogger.error("----> : %d Predict-Argument MisMatch Parse Tree" % error)
		with open(corpus_param.CPB_nouns_file,encoding = "utf-8-sig") as f:
			verb_predict_data = f.read().split("\n")
		last_field = None
		count = 0
		error = 0
		valid = 0
		for idx,verb_predict in enumerate(verb_predict_data):
			fields_values = verb_predict.split(" ")
			try:
				file_name,sent_idx,predict_idx,tagger,word_frameid = fields_values[:5]
			except:
				continue
			sent_idx = int(sent_idx)
			if file_name == "chtb_0112.nw" :
				if  sent_idx >= 10 or sent_idx in [3,4,5,6,7,8] or (sent_idx == 9 and (int(predict_idx) >= 14 or "取决于"in word_frameid) ):
					sent_idx = sent_idx - 1
			if file_name == "chtb_0139.nw":
				if sent_idx >= 10 or sent_idx in [4,5,6,7,8,9]:
					sent_idx = sent_idx - 1
			if file_name == "chtb_0307.nw":
				if sent_idx in [1]:
					sent_idx = sent_idx - 1
				if sent_idx >= 8 or sent_idx in [4,5,6,7]:
					sent_idx  = sent_idx - 2
			if file_name == "chtb_0437.nw":
				if  sent_idx >= 3:
						sent_idx -=1
			if file_name == "chtb_0672.nw":
				if sent_idx >= 10 or sent_idx >= 3:
					sent_idx -= 1
			if file_name == "chtb_0733.nw":
				if sent_idx >= 4 :
					sent_idx -= 1
			if file_name == "chtb_0787.nw":
				if sent_idx >= 3 :
					sent_idx -= 1
			if file_name == "chtb_0792.nw" or file_name == "chtb_0793.nw":
				if sent_idx >= 3 :
					sent_idx -= 1
			if file_name == "chtb_0794.nw"  :
				if sent_idx >= 2 :
					sent_idx -= 1
			if file_name == "chtb_0828.nw"  or file_name == "chtb_0845.nw" or file_name == "chtb_0855.nw" :
				if sent_idx >= 2 :
					sent_idx -= 1
			if file_name == "chtb_0877.nw":
				if sent_idx >= 5:
					sent_idx -= 2
			if file_name == "chtb_1042.mz":
				if sent_idx >= 100:
					sent_idx -=1
			predict_idx = int(predict_idx)
			predict,frame_id = word_frameid.split(".")
			def error_check():
				print("No predict")
				print("cpb:",predict,"terminal:",terminal)
				print(file_name,sent_idx,predict_idx)
				for i in range(sent_idx -2 ,sent_idx + 2):
					print("--------------",i)
					print([str(idx) + "." + label for idx,label in enumerate(self.files[file_name].sents[i]["parseTree"].terminals)])
				exit()

			try:
				file_sent = self.files[file_name].sents[sent_idx]
				sent_terminals = [str(idx) + "." + label for idx, label in enumerate(file_sent["parseTree"].terminals)]
				terminal = sent_terminals[predict_idx]
				if predict in terminal:
					args = dict(["-".join(arg.split("-")[1:]), arg.split("-")[0]] for arg in fields_values[6:])
					self.files[file_name].sents[sent_idx]["PA-structure"].append(
						{
							"predict": predict_idx,
							"args": args,
							"predict_type":"norminalVerb"
						}
					)
					valid += 1
				else:
					if last_field[-1].split(".")[0] == predict:
						continue
					error += 1
					#error_check()
			except:
				error += 1
				#error_check()

			last_field = [file_name, sent_idx, tagger, word_frameid]
			count += 1
		ConsoleLogger.debug("读入 {}/{} norminalVerb-Argument".format(valid,count))
		ConsoleLogger.error("----> : %d  MisMatch Parse Tree" % error)
	def load_pos(self):
		ConsoleLogger.debug("加载词性标签...")
		for file_name in self.files:
			self.files[file_name].addPartOfSpeech(self.parse_tree_dir + file_name)
	def load_parsetree(self):
		ConsoleLogger.debug("加载解析树...")
		for idx,file_name in enumerate(self.files):
			self.files[file_name].addParseTree(self.parse_tree_dir + file_name)

class ctb_file:

	__class__ = "CTB File"
	def __init__(self,name,segement_dir):
		with open(segement_dir + name + ".seg",encoding = "utf-8") as f:
			self.segemented = f.read()
		tree = BeautifulSoup(self.segemented,"html.parser")
		try:
			self.date = tree.find("date").text
		except:
			self.date = "NULL"
		try:
			self.title = tree.find("headline").text
		except:
			self.title = "NULL"
		self.sents = list(map(lambda x:
							  {"seg":x.text.replace("\n","").split(" "),"parseTree":None,"PA-structure":[],"pos":None},
							  tree.find_all('s'),
							  )
						  )
		self.content = "\n".join(["".join(sent) for sent in self.sents])
		self.predict_labels = [[] for i in range(len(self.sents))]

	def argument_label(self,sent_index,predict_index):
		pass
	def addPartOfSpeech(self,path):
		pass

	def addSemanticLabel(self,path):
		pass

	def addParseTree(self,path):
		with open(path,"r",encoding = "utf-8") as f:
			tree = BeautifulSoup(f.read(),"html.parser")

		for idx ,sent in enumerate(tree.find_all("s")):
			self.sents[idx]["parseTree"] = parseTree(sent.text)

class parseTree:

	def __init__(self,bracketContent):
		'''
		:param bracketContent: 带有括号的
		'''
		bracket_stack = []
		getNodeByindex = {}
		self.rootNode  = None
		self.terminalNodes = []
		for idx, char in enumerate(bracketContent):
			if char == "(":
				'''一旦遇到一个左括号 创建一个节点，并添加索引方式'''
				this_node = parseTreeNode()
				getNodeByindex[idx] = this_node
				if len(bracket_stack) > 0:

					'''如果不是第一个左括号，表明不是根节点，那么这个左括号的节点就是上个左括号节点的子节点'''
					last_left_bracket_idx = bracket_stack[-1][0]
					parentNode = getNodeByindex[last_left_bracket_idx]
					this_node.parent = parentNode
					parentNode.sons.append(this_node)
					if parentNode.content == None:
						parentNode.content = bracketContent[last_left_bracket_idx + 1:idx]
				else:
					#find root node
					self.rootNode = getNodeByindex[idx]

				bracket_stack.append([idx, char])
			if char == ")":
				'''如果遇到右括号 左括号出栈'''
				matched_left_bracket = bracket_stack.pop(-1)
				matched_left_bracket_idx = int(matched_left_bracket[0])
				this_node = getNodeByindex[matched_left_bracket_idx]
				if this_node.content == None:
					this_node.content = bracketContent[matched_left_bracket_idx + 1 : idx]
					self.terminalNodes.append(this_node)
		self.terminals = [node.content for node in self.terminalNodes]
class parseTreeNode:

	def __init__(self,content = None,parent = None,sons = []):
		self.content = content
		self.parent = parent
		self.sons = sons
if __name__ == "__main__":

	corpus = CTB(load_cpb=True, version="9")
	file = list(corpus.files.values())[0]
	print(type(file))
	for sent in file.sents:
		print("".join(sent["seg"]))
		PAs = sent["PA-structure"]
		terminals = sent["parseTree"].terminals
		for PA in PAs:
			print("predict",terminals[PA["predict"]])
			print("args",PA["args"])