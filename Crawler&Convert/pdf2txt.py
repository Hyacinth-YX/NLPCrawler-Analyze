# encoding: utf-8
import sys
import importlib
from concurrent.futures import ProcessPoolExecutor
import os

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from pdfminer.pdfpage import PDFPage

importlib.reload(sys)



def pdf_to_txt(pdf_file_path,txt_file_path):
    fp = open(pdf_file_path, 'rb') # 以二进制读模式打开
    #用文件对象来创建一个pdf文档分析器
    praser = PDFParser(fp)
    # 创建一个PDF文档
    doc = PDFDocument(praser, password='')
    # 连接分析器 与文档对象
    praser.set_document(doc)
    # 检测文档是否提供txt转换，不提供就忽略
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        # 创建PDf 资源管理器 来管理共享资源
        rsrcmgr = PDFResourceManager()
        # 创建一个PDF设备对象
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        # 创建一个PDF解释器对象
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # 循环遍历列表，每次处理一个page的内容
        for page in PDFPage.get_pages(fp): # doc.get_pages() 获取page列表
            interpreter.process_page(page)
            # 接受该页面的LTPage对象
            layout = device.get_result()
            for x in layout:
                if (isinstance(x, LTTextBoxHorizontal)):
                    with open(txt_file_path,'a') as f:
                        content = x.get_text ()
                        f.write(content + '\n')


def convert_all_pdfs(fold_path,dest_fold_path):
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        dest_fold_set = set(os.listdir(dest_fold_path))
        for file in os.listdir(fold_path) :
            extension_name = os.path.splitext(file)[1]
            if extension_name != '.pdf':
                continue
            file_name = os.path.splitext(file)[0]
            pdf_file = fold_path + '/' + file
            txt_file = dest_fold_path + '/' + file_name + '.txt'
            if file_name + '.txt' not in dest_fold_set:
                print(u'processing pdf: ',file)
                result = executor.submit(pdf_to_txt,pdf_file,txt_file)
                tasks.append(result)
    while True:
        exit_flag = True
        for task in tasks:
            if not task.done():
                exit_flag = False
        if exit_flag:
            print('All convert finished')
            return 0

max_workers = 3

if __name__ == '__main__':
    fold_path = '/Users/hyacinth/program/pdf2word/pdf2word/pdf'
    dest_fold_path = '/Users/hyacinth/program/pdf2word/pdf2word/txt'
    convert_all_pdfs(fold_path, dest_fold_path)