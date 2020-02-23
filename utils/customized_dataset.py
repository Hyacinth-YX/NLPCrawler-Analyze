import torch.utils.data as torchData
from utils.otherUtils import IO
from utils.dataprocessUtils import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from loguru import logger

class SeqLabeling(torchData.Dataset):
	def __init__(self, seq_path, label_path, label2index, num=-1):
		'''
		:param seq_path:
		:param label_path:
		:param label2index: a dict that can let you map a class label to  its index
		'''
		self.seqs = []
		self.labels = []
		self.seqs = list(map(lambda seq: [char for char in seq], IO.loadJson(seq_path)))[:num]
		self.labels = list(
			map(lambda seq_label: [label2index[label] for label in seq_label],
				IO.loadJson(label_path)))[:num]
		len_seqs = [len(seq) for seq in self.seqs]
		self.MAX_SEQ_LEN = max(len_seqs)
		self.labelNum = sum(len_seqs)
	def __len__(self):
		return len(self.seqs)
	def __getitem__(self,index):
		return self.seqs[index],self.labels[index]
class IOB(SeqLabeling):

	def __init__(self, seq_path, label_path, label2index, num=-1):
		super().__init__(seq_path, label_path, label2index,num = num)
		self.IB_Num = sum([sum([label > 0 for label in seq_label]) for seq_label in self.labels])
		self.O_Num = self.labelNum - self.IB_Num
		self.IB_weight = round(self.O_Num / self.IB_Num, 2)
		self.O_weight = round(self.IB_Num / self.O_Num, 2)
		self.seqs = list(map(lambda seq: ["[CLS]"] + seq  + ["[SEP]"]+ ["[PAD]"] * (self.MAX_SEQ_LEN - len(seq)), self.seqs))
		self.attention = list(map(lambda seq: [1] + [1] * len(seq) + [1] + [0] * (self.MAX_SEQ_LEN - len(seq)) , self.labels))
		self.labels = list(map(lambda seq: [-1] + seq + [-1] + [-1] * (self.MAX_SEQ_LEN - len(seq)), self.labels))



def create_batch_iter(mode,processor,tokenizer):
	"""构造迭代器"""
	if mode == "train":
		examples = processor.get_train_examples(args.data_dir)
		num_train_steps = int(
			len(examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
		batch_size = args.train_batch_size
		logger.info("  Num steps = %d", num_train_steps)
	elif mode == "dev":
		examples = processor.get_dev_examples(args.data_dir)
		batch_size = args.eval_batch_size
	else:
		raise ValueError("Invalid mode %s" % mode)
	label_list = processor.get_labels()
	# 特征
	features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

	logger.info("  Num examples = %d", len(examples))
	logger.info("  Batch size = %d", batch_size)

	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
	all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
	all_output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)
	# 数据集
	data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_output_mask)
	if mode == "train":
		sampler = RandomSampler(data)
	elif mode == "dev":
		sampler = SequentialSampler(data)
	else:
		raise ValueError("Invalid mode %s" % mode)
	# 迭代器
	iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

	if mode == "train":
		return iterator, num_train_steps
	elif mode == "dev":
		return iterator
	else:
		raise ValueError("Invalid mode %s" % mode)


