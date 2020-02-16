import tkinter as tk

import json

import sys

sys.path.append("./")

from utils import CallBack

import PyQt5.QtWidgets as QW

from PyQt5.QtGui import QIcon,QFont,QKeySequence

import os


class EditTable(QW.QWidget):


	def __init__(self,grid_row,grid_column,EditContainer,TextContainer):

		super().__init__()


		grid = QW.QGridLayout()

		grid.setSpacing(10)

		for i in range(grid_row):

			for j in range(grid_column):

				try:
					grid.addWidget(EditContainer[i][j],2 * i,j)

					grid.addWidget(TextContainer[i][j],2 * i+ 1,j)

				except:


					grid.addWidget(QW.QLabel("<span style = 'color:red'>END</span>"), 2 * i , j)

					grid.addWidget(QW.QLabel("<span style = 'color:red'>END</span>"), 2 * i + 1, j)

		self.setLayout(grid)

		self.show()



class Interface(QW.QMainWindow):

	def __init__(self,width = 600,height = 600):

		super().__init__()

		self.resize(width, height)

		self.grid_row = 12

		self.grid_column = 18

		with open("config.json",'r',encoding="utf-8") as f:

			self.config = json.loads(f.read())


		self.construct()

		self.construct_editor()

		self.showEditor()

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

		savebar = self.addToolBar('保存')
		savebar.addAction(saveAction)

		loadbar = self.addToolBar('加载')
		loadbar.addAction(openAction)

		upPagebar = self.addToolBar('上一页')
		upPagebar.addAction(upPageAction)

		downPagebar = self.addToolBar('下一页')
		downPagebar.addAction(downPageAction)

		self.statusBar()

	def index_to_postion(self,index):


		row = (index + 1) / self.grid_column

		row += 1

		if row == int(row):

			row -= 1

		row = int(row)

		column = (index + 1) - (row - 1) * self.grid_column


		return (int(row),int(column))

	def construct_editor(self,file = "D:/git_of_skydownacai/NLP_PROJECT/DocSequences/“二师兄”教你新知识，甲减患者为什么容易贫血和免疫力差.json"):

		self.filename = file.split("/")[-1]

		with open(file,encoding="utf-8") as f:

			seq = json.loads(f.read())[:600]

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

	def showEditor(self):

		try:

			self.editor.deleteLater()

		except:

			pass

		self.TextContainer = [] #存放Sequence 每个字符的文本

		self.EditContainer  = [] #存放Sequence 每个字符的标签编辑器


		left_index = self.now_page * self.pageSequenceLenth

		right_index = min((self.now_page + 1) * self.pageSequenceLenth,len(self.textSequence))

		for i in range(len(self.textSequence[left_index:right_index])):

			char = self.textSequence[left_index + i]

			row, column = self.index_to_postion(i)

			if len(self.TextContainer) < row:

				self.TextContainer.append([])

			if len(self.EditContainer) < row:

				self.EditContainer.append([])

			label = QW.QLabel(char)

			self.TextContainer[row - 1].append(label)

			Edit = QW.QLineEdit()

			Edit.__dict__["index"] = left_index + i

			Edit.textChanged.connect(CallBack(func = self.handleChange,absoluteIndex = left_index + i,relativeIndex = (row, column) ))

			Edit.setText(self.labelSequence[left_index + i])

			self.EditContainer[row - 1].append(Edit)



		self.editor = EditTable(self.grid_row,self.grid_column,self.EditContainer,self.TextContainer)

		self.setCentralWidget(self.editor)

		self.setWindowTitle('NLP序列标注工具--{},{}/{}页'.format(self.filename,self.now_page + 1,self.MaxPage))


	def handleChange(self,text,absoluteIndex,relativeIndex):

		row,column = relativeIndex

		self.labelSequence[absoluteIndex] = text

		if text.lower() in self.config:

			bg = self.config[text.lower()]["bg"]

			fg = "white"

		else:
			fg = "black"

			bg = "white"

		self.TextContainer[row - 1][column-1].setText("<span style = 'font-size:16px;background:{};color:{};'>{}</span>".format(bg,fg,self.textSequence[absoluteIndex]))

		self.setWindowTitle('NLP序列标注工具--{},{}/{}页(未保存)'.format(self.filename,self.now_page + 1,self.MaxPage))

	def save(self):

		with open("..\\DocLabel/"+self.filename,"w",encoding="utf-8") as f:

			f.write(json.dumps(self.labelSequence))

		self.setWindowTitle('NLP序列标注工具--{},{}/{}页'.format(self.filename,self.now_page + 1,self.MaxPage))

	def load(self):


		fname = QW.QFileDialog.getOpenFileName(self, 'open file', '/')[0]

		self.filename = fname.split("/")[-1]

		try:

			self.construct_editor(file = fname)

			self.showEditor()

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



if __name__ == '__main__':

	app = QW.QApplication(sys.argv)

	interface = Interface()

	sys.exit(app.exec_())