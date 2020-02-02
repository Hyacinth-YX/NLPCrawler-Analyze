# encoding: utf-8
import sys
import importlib
from concurrent.futures import ProcessPoolExecutor
import os
import re

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from pdfminer.pdfpage import PDFPage

importlib.reload(sys)


PREP_AND_ART_RE = [r'\b a\b', r'\b an\b', r'\b the\b', r'\b in\b', r'\b on\b', r'\b with\b',
                   r'\b by\b', r'\b for\b', r'\b at\b', r'\b about\b', r'\b under\b', r'\b of\b',
                   r'\b into\b', r'\b within\b', r'\b throughout\b', r'\b inside\b',
                   r'\b outside\b', r'\b without\b'] #标题中这些词语会小写

prep_art_pattern = re.compile('|'.join(PREP_AND_ART_RE))


def content_filter(content):
    '''
    删除每一个自然段多出来的换行符，转化特定的行
    TODO: 改进对列表（制表符）的转化，改进对公式的转化
    :param content: 待转化的自然段
    :return: perfect_content 转换好的自然段，结尾是一个换行符
    '''
    perfect_content = ''
    for row in content.split ('\n'):
        tmp_row = re.sub(prep_art_pattern, '', row)
        if tmp_row.istitle() or tmp_row.isupper():
            row += '\n' # 如果这一行是标题（大写字母开头）或者全大写，保留他的换行
        else:
            row += ' '
        perfect_content += row
    return perfect_content


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
                        perfect_content = content_filter(content)# perfect_content是一个以换行结尾的自然段
                        f.write(perfect_content + '\n')# 多加一个换行所以每个自然段之间有两个换行符
        print('finished')


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
    '''
    fold_path = '/Users/hyacinth/program/pdf2word/pdf2word/pdf'
    dest_fold_path = '/Users/hyacinth/program/pdf2word/pdf2word/txt'
    convert_all_pdfs(fold_path, dest_fold_path)
    '''
    # test code
    pdf_to_txt('/Users/hyacinth/program/pdf2word/pdf2word/pdf/J19-1001.pdf',
               '/Users/hyacinth/program/pdf2word/pdf2word/txt/J19-1001.txt')