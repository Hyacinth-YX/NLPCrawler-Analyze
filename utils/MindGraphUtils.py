'''
用于制作思维导图的公共类
'''
from typing import List,Tuple,Dict,Set,Optional
import jieba
import jieba.posseg as psg
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from pyecharts.charts import Page, Tree
from pyecharts import options as opts
class SemanticDomain:

	keyword : List[str] = ["影响", "有利", "不利"] + ["随着", "因为", "由于", "为了"] +  ["形成", "导致", "因而", "因此", "使得"] + ["通过", "利用", "凭借", "借助", "开展了"] + ["是"] + ["具有", "属于", "可以", "作为", "形如", "如同", "称为", "提供", "带来", "始于", "体现"] + ["只要", "条件", "直到", "需要"] + ["找","探索",'揭示']
	semantic_map : Dict = {
		"影响": ["影响", "有利", "不利"],
		"原因": ["随着", "因为", "由于", "为了"],
		"条件": ["只要", "条件", "直到", "需要"],
		"导致": ["形成", "导致", "因而", "因此", "使得"] ,
		"手段": ["通过", "利用", "凭借", "借助", "开展了"],
		"定义": ["是","定义","是指","是说"],
		"属性": ["调节","呈","呈现","具有", "属于", "可以", "作为", "形如", "如同", "称为", "提供", "带来", "始于", "体现"],

	}
	@staticmethod
	def getWordSemantic(word) -> str:
		for semantic in SemanticDomain.semantic_map:
			if word in SemanticDomain.semantic_map[semantic]:
				return semantic
		return "action"
class Node:
	def __init__(self,type = None,value = None,span = None,sons = [],father = None):
		self.type = type    # str
		self.value = value  # str
		self.span = span    # List[int]
		self.sons = sons      # List[node]
		self.father = father# Node
	def __str__(self):
		sons_value = "|".join(["%s(%s)"%(son.value,son.type) for son in self.sons])
		return ("type:%s value:%s span:%s son:%s father:%s"%
			  (self.type,self.value,str(self.span),sons_value,self.father.value if self.father != None else str(None))
			  )
	@staticmethod
	def same(nodeA,nodeB)->bool:
		if nodeA.span == nodeB.span:
			return True
		return False
	def copy(self):
		return Node(self.type,self.value,self.span,self.sons,self.father)

	@staticmethod
	def merge(nodes) -> List:
		VALUE_SET = set()
		for idx, node in enumerate(nodes):
			VALUE_SET.add(node.value)
		MERGED_NODES = dict([value,Node(value=value,sons=[],father=None)] for value in VALUE_SET)
		for node in nodes:
			NODE = MERGED_NODES[node.value]
			NODE.span = node.span
			NODE.type = node.type
			NODE.sons += node.sons
			for son in node.sons:
				son.father = NODE
				assert son.father == NODE
		return list(MERGED_NODES.values())

	@staticmethod
	def mergeBySpan(nodes):
		bigArg0 = []
		for i,node in enumerate(nodes):
			drop = False
			for j,othernode in enumerate(nodes):
				if j == i:
					continue
				if node.span[0] == othernode.span[0] and node.span[1] < othernode.span[1]:
					if node.sons[0].value in othernode.value:
						drop = True
						#print("drop", node)
						#print("by",othernode)
						#print()
					else:
						node.span  = othernode.span
						#print("replace", node)
						#print("by",othernode)
						node.value = othernode.value
						#print()
					break
			if not drop:
				bigArg0.append(node)
		return bigArg0
	def tree_data(self) -> Dict:

		def expand(node : Node):
			node_root = {"name":"%s(%s)"%(node.value,node.type),"children":[]}
			for son in node.sons:
				node_root["children"].append(expand(son))
			return node_root

		return expand(self)
class SematicDomain:
	@staticmethod
	def getDomain(word:str) -> str:
		pass
class Sent_Structure:
	def __init__(self,tokens : List[str],predicts:List[Tuple[str,list]],predicts_args:List[List[Dict]]):
		self.tokens = tokens
		self.sent = "".join(self.tokens)
		self.words = list(jieba.cut(self.sent,HMM=True,cut_all=False))
		self.word_segs = list(psg.cut(self.sent,HMM=True))
		self.tokenidx2wordidx ,self.wordidx2tokenidx =self.token2words()
		self.predictNodes,self.argNodes,self.predictvalues = self.createNodes(predicts,predicts_args)
		self.ARG0 =[]
		self.ARG1 =[]
		for idx,node in enumerate(self.predictNodes + self.argNodes):
			if node.type == "ARG0":
				self.ARG0.append(node)
			if node.type == "ARG1":
				self.ARG1.append(node)
		self.FixArg0()
		self.FixArg1()
	def nounphrase_completion(self,node):
		cuts = list(psg.cut(node.value, HMM=True))
		#print(cuts)
		wordindex = self.tokenidx2wordidx[node.span[1]]
		if list(cuts[-1])[1] in ['uj', 'c'] or list(cuts[-1])[0] in ["、"]:
			old_value = node.value
			for j in range(wordindex + 1, len(self.words)):
				this_word = self.words[j]
				word_pos  = list(psg.cut(this_word,HMM= True))
				last_word_pos = list(psg.cut(self.words[j-1], HMM=True))
				#print(this_word,word_pos)
				if j == wordindex + 1 and len(word_pos)==1 and list(word_pos[0])[1] in ["v","p"] :
					if list(word_pos[0])[1] == "p" and list(last_word_pos[0])[1] == "c":
						node.value = node.value[:  -1 * len(list(last_word_pos[0])[0])]
						node.span[1] -= len(list(last_word_pos[0])[0])
						break
					node.value += "".join(this_word)
					node.span[1] = self.wordidx2tokenidx[j][-1]
					break
				if j >= wordindex + 2 and len(word_pos)==1  and (list(word_pos[0])[1] in ["v","p"] or list(word_pos[0])[0] in [",", "，", "。", ".","”"]):
					node.value +=  "".join(self.words[wordindex + 1 : j])
					node.span[1] = self.wordidx2tokenidx[j-1][-1]
					break
			assert node.value == "".join(self.tokens[node.span[0]:node.span[1] + 1])
			assert node.span[1] > node.span[0]
		else:
			return 0
	@staticmethod
	def IsRef(word_str)->bool:
		'''判断是否是代词'''
		POS = list(psg.cut(word_str,HMM=True))
		for pos in POS:
			if list(pos)[1] == "r" or list(pos)[1] == "p":
				return True
		return False
	def FixArg0(self):
		#首先是消除句内coref
		fixed_arg0 = []
		cofer_map = {}
		for idx,ARG0 in enumerate(self.ARG0):
			if Sent_Structure.IsRef(ARG0.value):
				for j in range(0,idx):
					if not Sent_Structure.IsRef(self.ARG0[j].value):
						cofer_map[idx] = j
						break
		for ref in cofer_map:
			realArg0 = cofer_map[ref]
			self.ARG0[realArg0].sons += self.ARG0[ref].sons
			for son in self.ARG0[ref].sons:
				son.father = self.ARG0[realArg0]
		for idx,ARG0 in enumerate(self.ARG0):
			if idx not in cofer_map:
				fixed_arg0.append(ARG0)
		for node in self.predictNodes + self.argNodes:
			assert node.value == "".join(self.tokens[node.span[0]:node.span[1] + 1])
		self.ARG0 = fixed_arg0

		#接下来是对主语进行清理,过滤掉不要的信息 与 词性不相符的:
		cleared_arg0 = []
		for idx,node in enumerate(self.ARG0):
			cuts = list(psg.cut(node.value,HMM=True))
			#首先过滤掉开头是标点符号或者 主语只为连接词的
			if list(cuts[0])[0] in ["，",",","。","的"] or (list(cuts[0])[1] in ["c"] and len(cuts) == 1):
				word_len = len(list(cuts[0])[0])
				node.value = node.value[word_len:]
				node.span[0] += word_len
			#过滤掉主语结尾是部分不合法的
			if list(cuts[-1])[1] in ["d","a","p"] or list(cuts[-1])[0] in ["是","可能","占","，","。","作为"]: # 主语结尾是 状语 或 形容词
				word_len = len(list(cuts[-1])[0])
				node.value = node.value[:-1 * word_len]
				node.span[1] -= word_len
			#过滤掉主语为动词
			if list(cuts[-1])[1] == "v" and len(cuts) == 1:
				for son in node.sons:
					son.father = None
					assert son.father == None
				continue
			assert node.value == "".join(self.tokens[node.span[0]:node.span[1] + 1])
			if node.span[1] >= node.span[0]:
				cleared_arg0.append(node)
		self.ARG0 = cleared_arg0

		#接下来对主语进行补全,主语结尾是连接词的:
		for idx,node in enumerate(self.ARG0):
			self.nounphrase_completion(node)

		#接下来是扔掉部分
		self.ARG0 = Node.mergeBySpan(self.ARG0)

		#接下来是对ARG0合并: 基于简单的原则,如果
		self.ARG0 = Node.merge(self.ARG0)

	def FixArg1(self):
		for idx,node in enumerate(self.ARG1):
			cuts = list(psg.cut(node.value,HMM=True))
			#首先过滤掉开头是标点符号或者 主语只为连接词的
			if list(cuts[0])[0] in ["，",",","。"]:
				word_len = len(list(cuts[0])[0])
				node.value = node.value[word_len:]
				node.span[0] += word_len

			#过滤掉主语结尾是部分不合法的
			if list(cuts[-1])[0] in ["，",",","。"]: # 主语结尾是 状语 或 形容词
				word_len = len(list(cuts[-1])[0])
				node.value = node.value[:-1 * word_len]
				node.span[1] -= word_len
			assert node.value == "".join(self.tokens[node.span[0]:node.span[1] + 1])
		for idx, node in enumerate(self.ARG1):
			self.nounphrase_completion(node)
	def tokenspan2word(self,span : List[int]) -> Tuple[str,List[int]]:
		word_span = self.tokenidx2wordidx[span[0]], self.tokenidx2wordidx[span[1]]
		new_span  = self.wordidx2tokenidx[word_span[0]][0],self.wordidx2tokenidx[word_span[-1]][-1]
		new_value = "".join(self.tokens[new_span[0]:new_span[1] + 1])
		assert new_value== "".join(self.tokens[new_span[0]:new_span[1] + 1])
		assert new_span[1] >= new_span[0]
		return new_value,list(new_span)
	def createNodes(self,predicts,predicts_args):
		predictNodes,argsNodes,predictvalues = [],[],set()
		for predict,predict_args in zip(predicts,predicts_args):
			if not self.validPredict(predict,predictvalues):
				continue
			span = predict[1]
			type = predict[0]
			value = "".join(self.tokens[span[0]:span[1] + 1])
			if len(list(psg.cut(value,HMM=True))) >= 2:
				#print("drop predict",value)
				continue
			type = SemanticDomain.getWordSemantic(value)
			if type == "action":
				continue
			predict_node = Node(type=type, span=span,sons=[],value=value)
			assert predict_node.value == "".join(self.tokens[predict_node.span[0]:predict_node.span[1] + 1])
			arg_count = 0
			for arg in predict_args:
				if not self.validArg(predict_node,arg):
					continue
				arg_count += 1
				type = list(arg.keys())[0]
				span = list(arg.values())[0]

				#clear overlapping
				if span[0] <= predict_node.span[0] and span[1] == predict_node.span[1]:
					span = span[0],predict_node.span[0]-1
				value,new_span = self.tokenspan2word(span)
				for j in range(len(predict_node.value)):
					subword = predict_node.value[:j + 1]
					if value.endswith(subword):
						value = value[:-1 * j - 1]
						new_span[1] -= j + 1
						break

				#drop invalid node
				if new_span[1] < new_span[0]:
					continue

				if type == "ARG0":
					node = Node(sons = [predict_node],father=None,type=type,span=list(new_span),value = value)
					predict_node.father = node
				else:
					node = Node(father=predict_node,sons=[],type=type, span=list(new_span), value=value)
					predict_node.sons.append(node)
				assert node.value == "".join(self.tokens[node.span[0]:node.span[1] + 1])
				argsNodes.append(node)

			if arg_count > 0: #没有论元的谓词是不需要考虑的
				assert predict_node.sons != [] or predict_node.father != None
				predictNodes.append(predict_node)
				predictvalues.add(value)

		return predictNodes,argsNodes,predictvalues
	def validPredict(self,predict : dict,predictvalues:set)->bool:
		span = predict[1]
		value = "".join(self.tokens[span[0]:span[1] + 1])
		if value in predictvalues:
			return False
		predict_pos = list(psg.cut(value))
		#如果谓词里面没有动词成分,排除掉
		verb = False
		for pos in predict_pos:
			if "v" in list(pos)[1]:
				verb = True
				break
		return verb
	def validArg(self,predictNode : Node,arg : dict) -> bool:
		type = list(arg.keys())[0]
		span = list(arg.values())[0]
		value = self.tokenspan2word(span)
		if span == predictNode.span:
			return False
		return True
	def token2words(self)\
			->Tuple[List[int],List[Tuple[int,int]]]:
		tokenidx2wordidx = []
		wordidx2tokenidx = []
		for idx,word in enumerate(self.words):
			wordidx2tokenidx.append((len(tokenidx2wordidx),len(tokenidx2wordidx) + len(word) - 1))
			for j in range(len(tokenidx2wordidx),len(tokenidx2wordidx) + len(word)):
				tokenidx2wordidx.append(idx)
				assert self.tokens[j] in self.words[idx]
		assert len(tokenidx2wordidx) == len(self.tokens)
		return tokenidx2wordidx,wordidx2tokenidx
	def getNodeByspan(self,span):
		nodes = []
		for node in self.predictNodes:
			print(node)
			if node.span[0] >= span[0] and node.span[1] <= span[1]:
				nodes.append(node)
		for node in self.argNodes:
			if node.span[0] >= span[0] and node.span[1] <= span[1]:
				nodes.append(node)
def MindGraphTreeData(sents_structure,root_name = "Artical"):
	All_arg0 = []
	for sent in sents_structure:
		All_arg0 += sent.ARG0
	ARG0 = Node.merge(All_arg0)
	for idx,node in enumerate(ARG0):
		ARG0[idx].sons = Node.merge(ARG0[idx].sons)
		for node in ARG0[idx].sons:
			node.father = ARG0[idx]
			node.sons   =  Node.merge(node.sons)
			for child in node.sons:
				child.father = node
	value = {"name":root_name,"children":[]}
	for idx,node in enumerate(ARG0):
		value["children"].append(Node.tree_data(node))
	return value
