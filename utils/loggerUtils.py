# encoding:utf-8
import sys
sys.path.append("..")
import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler

def init_logger(logger_name,logging_path):
	if logger_name not in Logger.manager.loggerDict:
		logger = logging.getLogger(logger_name)
		logger.setLevel(logging.DEBUG)
		handler = TimedRotatingFileHandler(filename=logging_path + "all.log", when='D', backupCount=7)
		datefmt = '%Y-%m-%d %H:%M:%S'
		format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
		formatter = logging.Formatter(format_str, datefmt)
		handler.setFormatter(formatter)
		handler.setLevel(logging.INFO)
		logger.addHandler(handler)
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		console.setFormatter(formatter)
		logger.addHandler(console)

		handler = TimedRotatingFileHandler(filename=logging_path + "error.log", when='D', backupCount=7)
		datefmt = '%Y-%m-%d %H:%M:%S'
		format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
		formatter = logging.Formatter(format_str, datefmt)
		handler.setFormatter(formatter)
		handler.setLevel(logging.ERROR)
		logger.addHandler(handler)
	logger = logging.getLogger(logger_name)
	return logger



