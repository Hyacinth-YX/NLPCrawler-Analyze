from utils.loggerUtils import init_logger
import config.args as args
logger = init_logger("main",args.log_path)
import pytorch_pretrained_bert as bert
from utils.dataprocessUtils import convert_examples_to_features,InputExample
from models.Bert_Token_Classfier import Bert_Token_Classfier_Model
import os
from utils.visualizeUtils import colorful_text
from utils.otherUtils import IO
from utils.dataloaderUtils import features_2_batch_iter
if __name__ == "__main__":

	dir = r"D:\git_of_skydownacai\NLP_PROJECT\temp\ourData\Corpus\“二师兄”教你新知识，甲减患者为什么容易贫血和免疫力差.txt"
	#texts = IO.readdoc(dir).split("。")
	texts = "百度百科是百度公司推出的一部内容开放、自由的网络百科全书。".split("。")
	texts =  list(map(lambda text : " ".join(text),texts))
	tokens = list(map(lambda text : text.split(" "),texts))

	examples = []
	for text in texts:
		examples.append(InputExample(guid=0,text_a=text))
	tokenizer = bert.BertTokenizer.from_pretrained(args.bert_model_dir)
	labels = args.SRL_predict_label
	max_seq_len = 50
	model_dir = os.path.join(args.SRL_predict_model_dir,r"cached/pytorch_model_4.bin")
	features = convert_examples_to_features(examples,None,max_seq_len,tokenizer,True)
	dataloader = features_2_batch_iter(features)
	label_map = {label: i for i, label in enumerate(labels)}
	model = Bert_Token_Classfier_Model(300,len(labels)).from_pretrained(path = model_dir)
	output_tokens = []
	output_labels = []
	colormap = dict([label,"red" if "norminalVerb" in label else "blue"] for label in args.SRL_predict_label)
	colormap["O"] = "black"
	for idx,batch in enumerate(dataloader):
		input_ids, input_mask, segment_ids, label_ids, output_mask = batch
		b_prob = model(input_ids, segment_ids, input_mask)
		shape = list(b_prob.size())
		# 修改shape 方便计算
		b_predict = b_prob.argmax(dim=2).view(shape[0] * shape[1])
		b_y = label_ids.view(shape[0] * shape[1])
		tokens = texts[idx].split(" ")
		for i in range(min(len(tokens),max_seq_len)):
			output_tokens.append(tokens[i])
			output_labels.append(labels[b_predict[i].item()])