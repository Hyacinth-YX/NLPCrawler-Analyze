
'''
用于构建网络请求,数据读入等
'''

import requests
import requests.exceptions as ERRORS
import time
import re
import pandas as pd
import socket
import ssl
import random
import threading
import os
import json
import numpy as np
import abc
from loguru import logger


socket.setdefaulttimeout(30)
ssl._create_default_https_context = ssl._create_unverified_context

class CallBack:

	def __init__(self,func = lambda res:res,*args,**kwargs):

		self.args = args

		self.kwargs = kwargs

		self.func = func

	def __call__(self,res):

		return self.func(res,*self.args,**self.kwargs)

class Connection:

	default_config = {
		"timeout": 15,

		"stream" : False

	}

	def __init__(self,callback =  CallBack(),**kwargs):


		self.callback = callback

		self._config = Connection.default_config.copy()

		self._config.update(kwargs)

		self.max_retry = 3

		self.status = "Ready"


	def __getattr__(self, item):

		if item == "config":

			return  self._config

		return self._config[item]

	def __str__(self):

		return "[con]" + str(self.url)

	def get(self,**kwargs):
		'''
		通过url伪装用户请求网页内容，如果失败再尝试一次，仍然失败则返回False
		:param url: url地址 （string）
		:return: html 请求所得内容（string）
		'''

		if "http" not in self.url:

			url = "http://" + self.url

		self.status = "Connecting"

		if self.stream == True:

			html = requests.get(self.url, headers=Connection.get_header(), proxies=Connection.get_proxy(),
								stream=self.stream, **kwargs)

			self.response = StreamObj()


			for chunk in html.iter_content(1000):

				self.response += chunk

				self.status = "downloading:" + self.response.size

			self.status = "Finish"

			return True

		else:

			for idx in range(self.max_retry):

				try:
					html = requests.get(self.url, headers=Connection.get_header(), proxies=Connection.get_proxy(),
										stream=self.stream, **kwargs)

					html.encoding = "utf-8"

					self.response =  self.callback(html.text)

					self.status = "Finish"

					return True

				except:

					import traceback

					print(traceback.format_exc())

			else:
					self.status = "Error"

					return False

	@staticmethod
	def get_proxy():
		proxy_pool = ["39.137.69.7:8080", "221.178.176.25:3128",
					  "39.106.66.178:80", "58.222.32.77:8080",]
		proxy = {"http": "http://" + random.choice(proxy_pool)}

		a = random.random()

		if a > 1.1:

			return  proxy

		return  {}

	@staticmethod
	def get_header():
		'''
		随机获取一个headers
		:return:
		'''
		user_agents = ['Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
					   'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50' +
					   ' (KHTML, like Gecko) Version/5.1 Safari/534.50',
					   'Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11',
					   'Mozilla/5.0(Macintosh;U;IntelMacOSX10_6_8;' +
					   'en-us)AppleWebKit/534.50(KHTML,likeGecko)Version/5.1Safari/534.50 ',
					   'Mozilla/5.0(compatible;MSIE9.0;WindowsNT6.1;Trident/5.0;',
					   'Mozilla/4.0(compatible;MSIE8.0;WindowsNT6.0;Trident/4.0)',
					   'Mozilla/4.0(compatible;MSIE7.0;WindowsNT6.0)',
					   'Mozilla/5.0(Macintosh;IntelMacOSX10.6;rv:2.0.1)Gecko/20100101Firefox/4.0.1',
					   'Mozilla/5.0(WindowsNT6.1;rv:2.0.1)Gecko/20100101Firefox/4.0.1',
					   'Opera/9.80(Macintosh;IntelMacOSX10.6.8;U;en)Presto/2.8.131Version/11.11',
					   'Mozilla/5.0(Macintosh;IntelMacOSX10_7_0)AppleWebKit/535.11(KHTML,likeGecko)Chrome/' +
					   '17.0.963.56Safari/535.11',
					   'Mozilla/4.0(compatible;MSIE7.0;WindowsNT5.1;Trident/4.0;SE2.XMetaSr1.0;SE2.XMetaSr1.0;' +
					   '.NETCLR2.0.50727;SE2.XMetaSr1.0)',
					   'Mozilla/4.0(compatible;MSIE7.0;WindowsNT5.1;TheWorld)'
					   ]
		headers = {
			'User-Agent': random.choice(user_agents),
		}
		return headers

class StreamObj:

	def __init__(self):

		self.content = b''

	def append(self,other):

		self.content += other

	def __add__(self, other : bytes):

		self.content += other

		return  self

	@property
	def size(self):

		Kilobyte = self.bytesize / 1024

		Megabyte = self.bytesize  / 1024 ** 2

		if Megabyte > 10:

			BYTES_flag = str(round(Megabyte, 2)) + " MB"

		elif Kilobyte > 1:

			BYTES_flag = str(round(Kilobyte, 2)) + " KB"

		else:

			BYTES_flag = str(self.bytesize ) + " B"

		return BYTES_flag

	@property
	def bytesize(self,str = False):

		return len(self.content)

	def to_file(self,path):

		with open(path,'wb') as f:

			f.write(self.content)

	def __str__(self):

		return "<StreamObj>:" + self.size

class MultiConnections:

	def __init__(self,connections,n_jobs = 5 , type = "promise"):

		'''

		:param connections:
		:param n_jobs:
		:param type: str promise or generator
		'''
		self.type = type

		self.connections= dict([idx,connections[idx]] for idx in range(len(connections)))

		self.threads = dict([idx,MultiConnections.Async(connections[idx])] for idx in range(len(connections)))

		self.n_jobs = n_jobs

	def start(self):

		left_threads_idx = list(self.threads.keys())

		now_processes = []

		finish_count = 0

		while finish_count < len(self.threads.values()):

			if len(now_processes) < min(self.n_jobs,len(self.threads)) and len(left_threads_idx) > 0:

				'''add a job'''
				choice = random.choice(left_threads_idx)

				now_processes.append((choice,self.connections[choice]))

				self.threads[choice].start()

				left_threads_idx.remove(choice)

			message = "\r TIME__{}__: Now jobs:".format(int(time.time()))

			for process in now_processes:

				message += "[{:>2d}_{:<11s}]".format(process[0],process[1].status)

				if process[1].status == "Finish":

					finish_count += 1

					now_processes.remove(process)


				elif process[1].status == "Error":

					left_threads_idx.append(process[0])

			message += " || {} finish / {} Total".format(finish_count,len(self.threads))

			time.sleep(0.1)

			print(message,end = "",flush= True)

		if self.type == "promise":

			return [conn.response for conn in self.connections.values()]

	@staticmethod
	def Async(connection : Connection):

		thread = threading.Thread(target=connection.get)

		thread.name = str(connection)

		return thread


class IO(metaclass = abc.ABCMeta):

	def __init__(self, rootdir, whether_print=True):

		self.rootdir = rootdir

		self.whether_print = whether_print

	@abc.abstractmethod
	def __call__(self, *args, **kwargs):

		pass

	@staticmethod
	def loadJson(path):

		with open(path, "r", encoding="utf-8") as f:

			return  json.loads(f.read())

	@staticmethod
	def saveJson(path,value):

		with open(path, "w", encoding="utf-8") as f:

			f.write(json.dumps(value))

	@staticmethod
	def readdoc(path):
		with open(path,encoding="utf-8") as f:
			return f.read().replace(" ","").replace("\n","").replace("<subtitle>","").replace("</subtitle>","")
class saver(IO):

	def __init__(self,rootdir,whether_print = True):

		super().__init__(rootdir,whether_print)

	def __call__(self,path,savemethod = IO.saveJson):

		def out(func):

			def inner(*args, **kwargs):

				value = func(*args, **kwargs)

				if self.whether_print:

					logger.debug("[output]{} -> : {}".format(str(func), path))

				savemethod(path,value)

				return value

			return inner

		return out

class loader(IO):

	def __init__(self, rootdir, whether_print=True):

		super().__init__(rootdir,whether_print)

	def __call__(self,**inputkwargs):

		print(inputkwargs)

		def out(func):

			def inner(*args, **otherkwargs):

				if "loadmethod" not in inputkwargs:

					loadmethod = IO.loadJson

				else:
					loadmethod = inputkwargs["loadmethod"]

				for keyword in inputkwargs:

					inputvalue = loadmethod(inputkwargs[keyword])

					if self.whether_print:

						logger.info("[input]{} -> :{}".format(inputkwargs[keyword], str(func)))

					inputkwargs[keyword] = inputvalue

				otherkwargs.update(inputkwargs)

				value = func(*args, **otherkwargs)

				return value

			return inner

		return out


if __name__ == "__main__":

	#urls = pd.read_csv("pdfURLs.csv")["url"].values[:10]

	#schedule.downLoadAllpdfs(urls)

	pass
