import pytorch_pretrained_bert as bert

import torch

bert_model_dir = "bert_model/bert-chinese/"

tokenizer = bert.BertTokenizer.from_pretrained(bert_model_dir)

bert = bert.BertModel.from_pretrained(bert_model_dir )

seqs = "DocSequences/"

import os

import json

files =os.listdir(seqs)

for idx,file in enumerate(files):

	print("\r",idx,len(files),file,end = "",flush=True)

	with open(seqs + file,encoding="utf-8") as f:

		seqs = json.loads(f.read())


	for i in range(100000):

		if i * 50 > len(seqs):

			break

		ids = torch.tensor([tokenizer.convert_tokens_to_ids(seqs[i * 50 : min((i + 1) * 50,len(seqs))])])

		result = bert(ids, output_all_encoded_layers=False)


