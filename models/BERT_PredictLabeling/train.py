import sys
sys.path.append("../../")
import pytorch_pretrained_bert as bert
import torch
import config as args
from utils.dataprocessUtils import DataProcessor,InputExample
from utils.dataloaderUtils import create_batch_iter
from utils.modelUtils import SeqLabelingTrainer
from models.Bert_Token_Classfier import Bert_Token_Classfier_Model
import json

from utils.loggerUtils import init_logger
logger = init_logger("BERT Predict Labeling", args.log_path)

class SRL_PREDICT(DataProcessor):

	"""将数据构造成example格式"""
	def _create_example(self, lines, set_type):
		examples = []
		for i, line in enumerate(lines):
			guid = "%s-%d" % (set_type, i)
			line = json.loads(line)
			text_a = line["source"]
			label = line["target"]
			assert len(label.split()) == len(text_a.split())
			example = InputExample(guid=guid, text_a=text_a, label=label)
			examples.append(example)
		return examples
	def get_train_examples(self):
		lines = self._read_json(args.SRL_predict_train)
		examples = self._create_example(lines, "train")
		return examples
	def get_dev_examples(self):
		lines = self._read_json(args.SRL_predict_valid)
		examples = self._create_example(lines, "dev")
		return examples
	def get_labels(self):
		return args.SRL_predict_label

tokenizer = bert.BertTokenizer.from_pretrained(args.bert_model_dir)

data_processor = SRL_PREDICT()

Train_dataloader ,num_train_steps = create_batch_iter(
	processor = data_processor,
	tokenizer = tokenizer ,
	mode = "train",
	show_example = True,
)
Dev_dataloader  = create_batch_iter(
	processor = data_processor,
	tokenizer = tokenizer ,
	mode = "dev",
	show_example=False,
)

num_label = len(data_processor.get_labels())

model = Bert_Token_Classfier_Model(300,num_label).from_pretrained(
	args.SRL_predict_model_dir + "cached\\pytorch_model_4.bin")

loss_weight = torch.FloatTensor([3 for i in range(num_label)]).cuda()

loss_weight[0] = 1

logger.info("")

trainner = SeqLabelingTrainer(

	loss = torch.nn.CrossEntropyLoss(weight = loss_weight,ignore_index = -1),

	model = model,

	train_dataloader = Train_dataloader,

	dev_dataloader= Dev_dataloader,

	cached_path = "cached/",

	cached_point = 1
)

trainner.fit(num_epoch = 50 ,num_train_steps = num_train_steps)


