import NLPCrawler3
import pdf2txt

pdf_file_path = '/Users/hyacinth/program/pdf2word/pdf2word/pdf'  # 指定目标文件夹（确定已有该文件夹）
fold_path = '/Users/hyacinth/program/pdf2word/pdf2word/pdf'
dest_fold_path = '/Users/hyacinth/program/pdf2word/pdf2word/txt'

if __name__ == '__main__':
    NLPCrawler3.NLPCrawler3(pdf_file_path)
    pdf2txt.convert_all_pdfs(fold_path, dest_fold_path)