import sys
sys.path.append("..")
import config
from pytorch_pretrained_bert import BertModel
import torch
import numpy as np
import json
from tqdm import tqdm
from loguru import logger
class MyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(MyEncoder, self).default(obj)

def create_bert_embedings():
	p = config.ProjectConfiguration
	with open(p.bert_vocab,encoding = "utf-8") as f:
		lines = f.readlines()
		words = list(map(lambda x:x.replace("\n",""),lines))
	model = BertModel.from_pretrained(p.bert_model_dir)
	for idx,word in tqdm(enumerate(words),desc = "create bert embedding"):
		input_id = torch.tensor([[idx]],dtype= torch.long)
		embedding,_ = model(input_id,output_all_encoded_layers=False)
		embedding =  list(embedding[0,0].detach().numpy())
		with open(p.bert_embedding_path,"a",encoding = "utf-8") as f:
			f.write(str(word) + "\t" + str(embedding) + "\n")
if __name__ == "__main__":
	logger.info("Start create bert embedding numpy array file")
	create_bert_embedings()
	logger.info("Done √")
