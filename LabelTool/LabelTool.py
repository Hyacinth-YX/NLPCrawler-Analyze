import json

import sys

import re

sys.path.append("./")

from utils.otherUtils import CallBack

import PyQt5.QtWidgets as QW

from PyQt5.QtGui import QIcon

import os


class EditTable(QW.QWidget):


	def __init__(self,grid_row,grid_column,EditContainer,TextContainer):

		super().__init__()


		grid = QW.QGridLayout()

		grid.setSpacing(10)

		grid.addWidget(QW.QLabel("快捷输入"), 0, 0)

		fast_input_left = QW.QLineEdit(str(interface.fastinput[0]))

		fast_input_left.textChanged.connect(CallBack(func = interface.fastInput,flag = "left",target = fast_input_left))

		grid.addWidget(fast_input_left, 0, 1)

		grid.addWidget(QW.QLabel("左:右"), 0, 2)

		fast_input_right = QW.QLineEdit(str(interface.fastinput[1]))

		fast_input_right.textChanged.connect(CallBack(func = interface.fastInput,flag = "right" ,target = fast_input_right))

		grid.addWidget(fast_input_right, 0, 3)

		grid.addWidget(QW.QLabel("字符:"), 0, 4)

		fast_input_char = QW.QLineEdit(interface.fastinput[2])

		fast_input_char.textChanged.connect(CallBack(func = interface.fastInput,flag = "char",target = fast_input_char))

		grid.addWidget(fast_input_char, 0, 5)

		btn = QW.QPushButton("确认")

		btn.clicked.connect(CallBack(func = interface.fastInput,flag = "check"))

		grid.addWidget(btn, 0, 6)


		for i in range(grid_row):

			for j in range(grid_column):

				try:
					grid.addWidget(EditContainer[i][j],1 + 2 * i,j)

					grid.addWidget(TextContainer[i][j],1 + 2 * i+ 1,j)

				except:


					grid.addWidget(QW.QLabel("<span style = 'color:red'>END</span>"), 2 * i , j)

					grid.addWidget(QW.QLabel("<span style = 'color:red'>END</span>"), 2 * i + 1, j)

		self.setLayout(grid)

		self.show()



class Interface(QW.QMainWindow):

	def __init__(self,width = 600,height = 600):

		super().__init__()

		self.fastinput = ["","",""]

		self.resize(width, height)

		self.grid_row = 12

		self.grid_column = 18

		self.saved = True

		with open("config.json",'r',encoding="utf-8") as f:

			self.config = json.loads(f.read())


		self.construct()

		self.setWindowTitle('NLP序列标注工具')


		self.show()

	def construct(self):

		saveAction = QW.QAction(QIcon('resourses/save.jpg'), '保存', self)
		saveAction.setShortcut('Ctrl+S')
		saveAction.setStatusTip('保存序列标签')
		saveAction.triggered.connect(self.save)

		openAction = QW.QAction(QIcon('resourses/open.jpg'), '加载', self)
		openAction.setShortcut('Ctrl+P')
		openAction.setStatusTip('加载文档')
		openAction.triggered.connect(self.load)


		upPageAction = QW.QAction(QIcon('resourses/previouspage.jpg'), '上一页', self)
		upPageAction.setShortcut("↑")
		upPageAction.setStatusTip('上一页')
		upPageAction.triggered.connect(self.upPage)

		downPageAction = QW.QAction(QIcon('resourses/nextpage.jpg'), '下一页', self)
		downPageAction.setShortcut("↓")
		downPageAction.setStatusTip('下一页')
		downPageAction.triggered.connect(self.downPage)

		upDocAction = QW.QAction(QIcon('resourses/previousDoc.jpg'), '上一页', self)
		upDocAction.setStatusTip('上篇文章')
		upDocAction.triggered.connect(self.upDoc)

		downDocAction = QW.QAction(QIcon('resourses/nextDoc.jpg'), '下一篇', self)
		downDocAction.setStatusTip('下篇文章')
		downDocAction.triggered.connect(self.downDoc)

		savebar = self.addToolBar('保存')
		savebar.addAction(saveAction)

		loadbar = self.addToolBar('加载')
		loadbar.addAction(openAction)

		upPagebar = self.addToolBar('上一页')
		upPagebar.addAction(upPageAction)

		downPagebar = self.addToolBar('下一页')
		downPagebar.addAction(downPageAction)

		upDocbar = self.addToolBar('上一篇')
		upDocbar.addAction(upDocAction)

		downDocbar = self.addToolBar('下一篇')
		downDocbar.addAction(downDocAction)

		self.statusBar()

	def index_to_postion(self,index):


		row = (index + 1) / self.grid_column

		row += 1

		if row == int(row):

			row -= 1

		row = int(row)

		column = (index + 1) - (row - 1) * self.grid_column


		return (int(row),int(column))

	def construct_editor(self,file):

		self.filename = file.split("/")[-1]

		with open(file,encoding="utf-8") as f:

			seq = json.loads(f.read())

		try:
			with open("..\\DocLabel/"+self.filename,encoding="utf-8") as f:

				self.labelSequence = json.loads(f.read())
		except:

			self.labelSequence = ["O" for O in range(len(seq))]

		self.textSequence = seq

		self.pageSequenceLenth = self.grid_column * self.grid_row

		self.MaxPage =  len(self.textSequence) / self.pageSequenceLenth

		if self.MaxPage != int(self.MaxPage):

			self.MaxPage = int(self.MaxPage) + 1

		self.now_page = 0

	def showEditor(self,first = False):

		try:

			self.editor.deleteLater()

		except:

			pass

		self.TextContainer = [] #存放Sequence 每个字符的文本

		self.EditContainer  = [] #存放Sequence 每个字符的标签编辑器


		left_index = self.now_page * self.pageSequenceLenth

		right_index = min((self.now_page + 1) * self.pageSequenceLenth,len(self.textSequence))

		count = -1

		for i in range(len(self.textSequence[left_index:right_index])):

			count += 1

			char = self.textSequence[left_index + i]

			row, column = self.index_to_postion(i)

			if len(self.TextContainer) < row:

				self.TextContainer.append([])

			if len(self.EditContainer) < row:

				self.EditContainer.append([])

			label = QW.QLabel(str(count) + "." +char)

			self.TextContainer[row - 1].append(label)

			Edit = QW.QLineEdit()

			Edit.__dict__["index"] = left_index + i

			Edit.textChanged.connect(CallBack(func = self.handleChange,absoluteIndex = left_index + i,relativeIndex = (row, column) ))

			Edit.setText(self.labelSequence[left_index + i])

			self.EditContainer[row - 1].append(Edit)



		self.editor = EditTable(self.grid_row,self.grid_column,self.EditContainer,self.TextContainer)

		self.setCentralWidget(self.editor)

		if first:

			self.saved = True

		windows_title = 'NLP序列标注工具--{},{}/{}页'.format(self.filename,self.now_page + 1,self.MaxPage)

		windows_title += "(未保存)" if self.saved == False else ""

		self.setWindowTitle(windows_title)

	def handleChange(self,text,absoluteIndex,relativeIndex):

		row,column = relativeIndex

		self.labelSequence[absoluteIndex] = text

		self.saved = False

		if text.lower() in self.config:

			bg = self.config[text.lower()]["bg"]

			fg = "white"

		else:
			fg = "black"

			bg = "white"

		pageindex = (row - 1) * self.grid_column + column - 1

		oldText = self.textSequence[absoluteIndex]

		self.TextContainer[row - 1][column-1].setText("<span style = 'font-size:16px;background:{};color:{};'>{}</span>".format(bg,fg,str(pageindex) + "." + oldText))

		self.setWindowTitle('NLP序列标注工具--{},{}/{}页(未保存)'.format(self.filename,self.now_page + 1,self.MaxPage))

	def save(self):

		try:
			with open("..\\DocLabel/"+self.filename,"w",encoding="utf-8") as f:

				f.write(json.dumps(self.labelSequence))

			self.saved = True

			self.setWindowTitle('NLP序列标注工具--{},{}/{}页'.format(self.filename,self.now_page + 1,self.MaxPage))

		except:

			pass

	def load(self):

		fname = QW.QFileDialog.getOpenFileName(self, 'open file', '/')[0]

		try:

			self.filename = fname.split("/")[-1]

			self.filedir  = fname.replace(self.filename,"")

			idx = re.search(r'\[[\w]*\]', self.filename).span()

			self.fileindex = int(self.filename[idx[0] + 1: idx[1] - 1])

		except:

			return 0

		try:

			self.construct_editor(file = fname)

			self.showEditor(first = True)

		except:

			pass

	def upPage(self):

		if self.now_page > 0:

			self.now_page -= 1

			self.showEditor()

	def downPage(self):

		if self.now_page < self.MaxPage - 1:

			self.now_page += 1

			self.showEditor()

	def upDoc(self):

		self.save()

		files = os.listdir(self.filedir)

		for file in files:

			if file.startswith("[{}]".format(self.fileindex - 1)):

				self.fileindex -= 1

				self.construct_editor(file=self.filedir + file)

				self.showEditor(first=True)

				break

	def downDoc(self):

		self.save()

		files = os.listdir(self.filedir)

		for file in files:

			if file.startswith("[{}]".format(self.fileindex + 1)):

				self.fileindex += 1

				self.construct_editor(file=self.filedir + file)

				self.showEditor(first=True)

				break

	def fastInput(self,text,**kwargs):

		if text == "":

			return 0

		try:

			if kwargs["flag"] == "check":

				left,right,char = self.fastinput


				for i in range(left,right + 1):

					row, column = self.index_to_postion(i)

					absoluteIndex = self.now_page * self.pageSequenceLenth  + i

					self.EditContainer[row - 1][column - 1].setText(char)

					self.labelSequence[absoluteIndex] = char

				self.save()

			if kwargs["flag"] == "left":

				self.fastinput[0] = max(0,int(text))

			if kwargs["flag"] == "right":

				self.fastinput[1] = min(int(text),self.pageSequenceLenth - 1)

			if kwargs["flag"] == "char":

				self.fastinput[2] = text
		except:

			import traceback

			kwargs["target"].setText("")

if __name__ == '__main__':

	app = QW.QApplication(sys.argv)

	interface = Interface()

	sys.exit(app.exec_())