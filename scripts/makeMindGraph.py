import sys
sys.path.append("..")
from utils.MindGraphUtils import *
from scripts.runBertTokenClassfier import run
import config
import json
from loguru import logger
def make(file):
	filename = file.split("/")[-1][:-4]
	tokens, srl = run(file,"SRL")
	sents_srl = {}
	for item in srl:
		sent_idx = item["sent_idx"]
		if sent_idx not in sents_srl:
			sents_srl[sent_idx] = {"tokens":tokens[sent_idx],"predicts":[],"args":[]}
		sents_srl[sent_idx]["predicts"].append((item["predict_type"],item["predict_span"]))
		sents_srl[sent_idx]["args"].append(item["args"])
	sent_structures = []
	for idx,sent_srl in enumerate(sents_srl.values()):
		sent = Sent_Structure(sent_srl["tokens"],sent_srl["predicts"],sent_srl["args"])
		sent_structures.append(sent)
	value = MindGraphTreeData(sent_structures,root_name = filename)
	logger.debug(value)
	output_name = config.ProjectConfiguration.output_path + filename + "_treedata"+ ".txt"
	with open(output_name,"w",encoding="utf-8") as f:
		f.write(json.dumps(value))
		logger.info("Success ! output -> : %s " % output_name )
if __name__ == "__main__":
	file = sys.argv[1]
	logger.info("run file %s "%file)
	make(file)

