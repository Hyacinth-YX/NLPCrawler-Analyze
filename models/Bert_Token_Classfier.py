import pytorch_pretrained_bert as bert
import torch
from utils.loggerUtils import init_logger
from utils.modelUtils import load_model
import config.args as args
logger = init_logger("BERT Token Classfier",args.log_path)
import os
class Bert_Token_Classfier_Model(torch.nn.Module):

	def __init__(self,Hidden_num,labelnum):

		super().__init__()

		self.bert = bert.BertModel.from_pretrained(args.bert_model_dir)

		self.classfier = torch.nn.Sequential(

			torch.nn.Linear(768,Hidden_num),

			torch.nn.Tanh(),

			torch.nn.Linear(Hidden_num,labelnum)
		)
		self.prob = torch.nn.Softmax(dim = -1)

	def forward(self, input_ids,token_type_ids,attention_mask):

		bertencoded,_ = self.bert(input_ids, token_type_ids , attention_mask,output_all_encoded_layers=False)

		output = self.classfier(bertencoded)

		prob = self.prob(output)

		return prob

	def from_pretrained(self,path):

		logger.info("load Bert Token_Classfier model from %s "%path)

		state_dict = load_model(path)

		self.load_state_dict(state_dict)

		self.eval()

		return self
class Bert_Token_Classfier_Model2(torch.nn.Module):

	def __init__(self,labelnum):

		super().__init__()

		self.bert = bert.BertModel.from_pretrained(args.bert_model_dir)

		self.classfier = torch.nn.Sequential(

			torch.nn.Linear(768 * 4,labelnum),

		)
		self.prob = torch.nn.Softmax(dim = -1)

	def forward(self, input_ids,token_type_ids,attention_mask):

		bertencoded_layers,_ = self.bert(input_ids, token_type_ids , attention_mask,output_all_encoded_layers=True)

		bertencoded = torch.cat(bertencoded_layers[-4:],dim = 2)

		output = self.classfier(bertencoded)

		prob = self.prob(output)

		return prob

	def from_pretrained(self,path):

		logger.info("load Bert Token_Classfier model 2 from %s "%path)

		state_dict = load_model(path)

		self.load_state_dict(state_dict)

		self.eval()

		return self
