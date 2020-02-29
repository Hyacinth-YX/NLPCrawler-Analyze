import sys
import torch
sys.path.append("..")
from config import Seq2SeqPram,ProjectConfiguration
from loguru import logger
from utils.seq2seqUtils import TestDataset
from utils.seq2seqTestUtils import decode_batch_output,format_tokens
from models.pointer_network import Seq2Seq
from utils.seq2seqUtils import Vocab
import jieba
def run(file):
	with open(file,encoding="utf-8") as f :
			text = f.read()
	filename = file.split("/")[-1][:-4]
	DEVICE = torch.device(Seq2SeqPram.device)
	v = Vocab.load(Seq2SeqPram.vocab)
	p = Seq2SeqPram()
	m = Seq2Seq(v,p).from_pretrained(ProjectConfiguration.trained_Pointer_Network)
	sentens =  text.replace("\n","").split("。")
	text_pieces = [[]]
	for idx,sent in enumerate(sentens):
		sent_words = list(jieba.cut(sent,HMM=True))
		if len(text_pieces[-1]) + len(sent_words) > Seq2SeqPram.max_src_len and idx <= len(sentens) - 2:
			text_pieces.append(sent_words)
		else:
			text_pieces[-1] += sent_words + ["。"]
	text_pieces = list(map(lambda x : "".join(x),text_pieces))
	dataset = TestDataset(text_pieces,vocab=v)
	batch_iter = dataset.generator()
	contents = []
	for batch in batch_iter:
		with torch.no_grad():
			hypotheses = m.beam_search(
				input_tensor  = batch.input_tensor.to(DEVICE),
				input_lengths = batch.input_lengths if p.pack_seq else None,
				ext_vocab_size= batch.ext_vocab_size,
				min_out_len = p.min_out_len,
				max_out_len = p.max_out_len,
				len_in_words= True)
		best_summarizes =[ hypotheses[0].tokens ] if len(hypotheses) > 0 else [""]
		decoded_batch = decode_batch_output(best_summarizes, v, batch.oov_dict)
		contents.append(format_tokens(decoded_batch[0]))
	value = "。".join(contents)
	output_name = ProjectConfiguration.output_path + filename + "_summary"+ ".txt"
	with open(output_name,"w",encoding="utf-8") as f:
		f.write(value.replace(" ","").replace("(组图)","").replace("(图)","").replace("组图","").replace("\n",""))
		logger.info("Success ! output -> : %s " % output_name )
if __name__ == "__main__":
	file = sys.argv[1]
	logger.info("get summary of  %s "%file)
	run(file)