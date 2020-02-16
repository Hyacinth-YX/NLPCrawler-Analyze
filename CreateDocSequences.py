import pytorch_pretrained_bert as bert

import torch

import json

import re

import os

files = os.listdir("Corpus")

def Normalize(text):

	text = re.sub(r"<subtitle>[\W|\w]*</subtitle>","",text)

	return re.sub(r"[”“]",'"',text)


bert_model_dir = "bert_model/bert-chinese/"

tokenizer = bert.BertTokenizer.from_pretrained(bert_model_dir)

bert = bert.BertModel.from_pretrained(bert_model_dir )

for idx,file in enumerate(files):

	print("\r",idx,file,end = "",flush = True)

	with open("Corpus/{}".format(file),encoding="utf-8") as f:

		s = f.read()

	s = Normalize(s)

	tokens = tokenizer.tokenize(s)

	ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

	with open("DocSequences/[{}]{}.json".format(idx + 1 ,file[:-4]),"w",encoding="utf-8") as f:

		f.write(json.dumps(tokens))