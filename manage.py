import argparse
import subprocess
import config
import os
from loguru import logger
if __name__ == "__main__":
	valid_models = ["SRL","PredictRecoginize","NER","TEXT_SUMMARY"]
	config = config.ProjectConfiguration()
	parser = argparse.ArgumentParser(description = "QuickReader Library")
	parser.add_argument("-m",
						metavar ="model",
						nargs = "+",
						help = "models to use : `NER`,`PredictRecoginize`,`SRL`,`TEXT_SUMMARY`",
						default = [],
						choices=valid_models
						)
	parser.add_argument("-p",metavar ="produce",
						help = "produce data for models : `NER`,`PredictRecoginize`,`SRL`,`TEXT_SUMMARY`",
						default = [],
						nargs="+",
						choices = valid_models)
	parser.add_argument("-G",metavar ="MindGraph",
						help = "Make MindGraph for a input text",
						default=None)
	parser.add_argument("--f",metavar ="filepath",help = "load_text_file",default=None)
	parser.add_argument("--createBertEmbedding",
						help = "Create BertEmbedding numpy array file for Chinese token in bert vocabulary",
						action= "store_true",
						default=False
						)
	parser.add_argument("--env",
						metavar="conda virtual environment",
						help = "run this project using specific conda virtual environment ",
						default=None
						)

	args =parser.parse_args()
	if args.env:
		env_prefix = "conda activate %s & " % args.env
	else:
		env_prefix = ""
	print(env_prefix)
	if args.p != []:
		modelNames = [model.upper() for model in args.p]
		subprocess.call(env_prefix + "python produce_data.py " + " ".join(modelNames),
			cwd=os.path.join(config.ROOT_DIR,"data_preprocess"))
	if args.createBertEmbedding:
		subprocess.call(env_prefix + "python produce_embedding.py",
			cwd=os.path.join(config.ROOT_DIR,"data_preprocess"))
	if args.m != []:
		if args.f == None:
			logger.error("To run a NLP model,you must input a text.Try: --f path/to/your/text.txt")
			exit()
		modelNames = [model.upper() for model in args.m]
		values = []
		for model in modelNames:
			if model == "TEXT_SUMMARY":
				subprocess.call(
					env_prefix + "python runTextSummary.py %s %s" % (args.f,model),
					shell=True,
					cwd=os.path.join(config.ROOT_DIR,"scripts"))
			elif model == "NER":
				subprocess.call(
					env_prefix + "python runBertTokenClassfier.py %s %s"  % (args.f,"NER_CLUE"),
					shell=True,
					cwd=os.path.join(config.ROOT_DIR,"scripts"))
				subprocess.call(
					env_prefix + "python runBertTokenClassfier.py %s %s"  % (args.f,"NER_PEOPLEDAILY"),
					shell=True,
					cwd=os.path.join(config.ROOT_DIR,"scripts"))
			else:
				subprocess.call(
					env_prefix + "python runBertTokenClassfier.py %s %s" % (args.f,model),
					shell=True,
					cwd=os.path.join(config.ROOT_DIR,"scripts"))
	if args.G:
		subprocess.call(
			env_prefix + "python makeMindGraph.py %s" % (args.G),
			shell=True,
			cwd=os.path.join(config.ROOT_DIR, "scripts"))