import sys
sys.path.append("..")
import config
from models.Bert_Token_Classfier import Bert_Token_Classfier_Model
from typing import List,Dict,Tuple
from utils.dataprocessUtils import text2TokenClassfierDataProcessor,load_bert_tokenizer,tensor2list,convert_tokens_to_features,load_sub_vocab,parseIOBlabel
from loguru import logger
import os
import torch
import json
def run(file : str,model :str):
	with open(file,encoding="utf-8") as f:
		text = f.read()
	filename = file.split("/")[-1][:-4]
	logger.debug("run model: %s with file %s " % (model,filename))
	param = config.BertTokenClassfierParam()
	device = torch.device("cpu" if param.no_cuda else param.device)
	model_name = model.upper()
	model_labels = \
		{"NER_CLUE": config.NER_Param.labels,
		"NER_PEOPLEDAILY": config.NER_Param.PEOPLEDAILY_labels,
		"PREDICT_LABELING": config.PredictLabeling_Param.label,
		"SRL": config.SRL_Param.label}[model_name]
	model_functions =\
		{"NER_CLUE":ner_or_predictlabeling,
		"NER_PEOPLEDAILY": ner_or_predictlabeling,
		"PREDICT_LABELING":ner_or_predictlabeling,
		"SRL":srl}
	model_trained = \
		{"NER_CLUE": config.ProjectConfiguration.trained_NER_CLUE,
		"NER_PEOPLEDAILY": config.ProjectConfiguration.trained_NER_PEOPLEDAILY,
		"PREDICT_LABELING": config.ProjectConfiguration.trained_PredictLabeling,
		"SRL": config.ProjectConfiguration.trained_SRL}
	text  = text.replace("\n", "").replace(" ", "").replace("\ufeff","").lstrip().rstrip().replace("","")
	lines = text.split("。")  # 按逗号分割
	lines = list(map(lambda x: " ".join(x)+ " 。", lines))
	dataprocessor = text2TokenClassfierDataProcessor(lines)
	subvocab = load_sub_vocab()
	tokenizer = load_bert_tokenizer()
	if model_name != "SRL":
		model = Bert_Token_Classfier_Model(300, len(model_labels)).from_pretrained(
			os.path.join(model_trained[model_name])).to(device)
		_func = model_functions[model_name]
		value = _func(model,model_labels,tokenizer,subvocab,dataprocessor,device)
	else:
		predict_labeling_model = Bert_Token_Classfier_Model(300, len(config.PredictLabeling_Param.label)).from_pretrained(
			os.path.join(model_trained["PREDICT_LABELING"])).to(device)
		origin_tokens, output_labels,predict_IOB_parsed = \
			ner_or_predictlabeling(predict_labeling_model,
								   config.PredictLabeling_Param.label, tokenizer, subvocab, dataprocessor, device)
		del predict_labeling_model # 释放显存
		torch.cuda.empty_cache()
		lines = []
		semantic_role_labels = []
		for idx,sent_predicts in enumerate(predict_IOB_parsed):
			tokens = origin_tokens[idx]
			text_a = " ".join(tokens)
			for predict in sent_predicts:
				predict_type = list(predict.keys())[0]
				span = list(predict.values())[0]
				text_b = " ".join(tokens[span[0]:span[1] + 1])
				lines.append((text_a,text_b))
				semantic_role_labels.append({"sent_idx":idx,"predict_span":span,"predict_type":predict_type})
		dataprocessor = text2TokenClassfierDataProcessor(lines)
		model = Bert_Token_Classfier_Model(300, len(model_labels)).from_pretrained(
		os.path.join(model_trained["SRL"])).to(device)
		origin_tokens_for_srl,output_labels,_IOB_parsed = srl(model,model_labels,tokenizer,subvocab,dataprocessor,device)
		if device != "cpu":
			torch.cuda.empty_cache()
		assert len(semantic_role_labels) == len(_IOB_parsed)
		for idx,arguments in enumerate(_IOB_parsed):
			semantic_role_labels[idx]["args"] = arguments
			semantic_role_labels[idx]["Labelseq"] = output_labels[idx]
		value = (origin_tokens,semantic_role_labels)
	if device != "cpu":
		torch.cuda.empty_cache()
	del model
	import json
	output_name = config.ProjectConfiguration.output_path + filename + "_%s" % model_name + ".txt"
	with open(output_name,"w",encoding="utf-8") as f:
		f.write(json.dumps(list(value)))
		logger.info("Success ! output -> : %s " % output_name )
	return value
def ner_or_predictlabeling(model,model_labels,tokenizer,subvocab,dataprocessor,device)\
		-> Tuple[List[List[str]],List[List[str]],List[List[Dict]]]:
	origin_tokens = []
	output_labels = []
	for step, example in dataprocessor:
		_tokens = example.text_a.split(" ")
		tokens_a = tokenizer.tokenize(example.text_a)
		tokens,label_id,input_ids,input_mask,segment_ids,output_mask = \
			convert_tokens_to_features(
				tokens_a = tokens_a,
				labels = None,
				max_seq_length = len(example.text_a.split(" ")) + 2,
				tokenizer = tokenizer,
				label_list = None,
				sub_vocab = subvocab,
				segment_id = 0)
		#if step == 0:
		#	logger.info("-----------------Example-----------------")
		#	logger.info("guid: %s" % (example.guid))
		#	logger.info("tokens: %s" % " ".join([str(x) for x in tokens]) + "len: %d"%len(tokens))
		#	logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
		#	logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
		#	logger.info("segment ids: %s " % " ".join([str(x) for x in segment_ids]))
		#	logger.info("label_ids : %s " % " ".join([str(x) for x in label_id]))
		#	logger.info("output_mask: %s " % " ".join([str(x) for x in output_mask]))
		input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
		input_mask = torch.tensor([input_mask], dtype=torch.long).to(device)
		segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(device)
		prob = model(input_ids, segment_ids, input_mask)[0]
		predict = prob.argmax(dim=1)
		line_labels = tensor2list(predict[1:-1])
		line_labels = list(map(lambda label_id: model_labels[label_id], line_labels))
		assert len(line_labels) == len(_tokens)
		origin_tokens.append(_tokens)
		output_labels.append(line_labels)
	_IOB_parsed = list(map(parseIOBlabel,output_labels))
	return origin_tokens,output_labels,_IOB_parsed
def srl(model,model_labels,tokenizer,subvocab,dataprocessor,device) \
		-> Tuple[List[List[str]],List[List[str]],List[List[Dict]]]:
	origin_tokens = []
	output_labels = []
	for step, example in dataprocessor:
		_tokens = example.text_a.split(" ")
		tokens_a = tokenizer.tokenize(example.text_a)
		tokens,label_id,input_ids,input_mask,segment_ids,output_mask = \
			convert_tokens_to_features(
				tokens_a = tokens_a,
				labels = None,
				max_seq_length = len(example.text_a.split(" ")) + 2,
				tokenizer = tokenizer,
				label_list = None,
				sub_vocab = subvocab,
				segment_id = 0)
		tokens_b = tokenizer.tokenize(example.text_b)
		value = convert_tokens_to_features(
			tokens_a=tokens_b,
			labels=None,
			max_seq_length=len(example.text_b.split(" ")) + 2,
			tokenizer=tokenizer,
			label_list=None,
			sub_vocab=subvocab,
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
		#if step == 0:
		#	logger.info("-----------------Example-----------------")
		#	logger.info("guid: %s" % (example.guid))
		#	logger.info("tokens: %s" % " ".join([str(x) for x in tokens]) + "len: %d"%len(tokens))
		#	logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
		#	logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
		#	logger.info("segment ids: %s " % " ".join([str(x) for x in segment_ids]))
		#	logger.info("label_ids : %s " % " ".join([str(x) for x in label_id]))
		#	logger.info("output_mask: %s " % " ".join([str(x) for x in output_mask]))
		input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
		input_mask = torch.tensor([input_mask], dtype=torch.long).to(device)
		segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(device)
		prob = model(input_ids, segment_ids, input_mask)[0]
		predict = prob.argmax(dim=1)

		line_labels = tensor2list(predict[segment_ids[0] == 0][1:-1])
		line_labels = list(map(lambda label_id: model_labels[label_id], line_labels))
		assert len(line_labels) == len(_tokens)
		origin_tokens.append(_tokens)
		output_labels.append(line_labels)
	_IOB_parsed = list(map(parseIOBlabel,output_labels))
	return origin_tokens,output_labels,_IOB_parsed
if __name__ == "__main__":
	file,model = sys.argv[1:3]
	logger.info("run file %s "%file)
	run(file,model)
