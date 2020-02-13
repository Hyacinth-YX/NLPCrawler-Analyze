# NLPCrawler-Analyze
爬取论文并储存为txt格式，对每篇论文进行标记、分析、处理获得容易理解的思维导图

2020.1.31 爬虫部分+pdf2txt基本完成，爬虫位于Crawler&Convert文件夹中
    Crawler&Convert：
    
        main.py: 整合爬虫与文本类型转化
        
        NLPCrawler3.py：用python写的特制爬虫，无法爬取其他网站
        爬取网站（https://www.aclweb.org/anthology/）
        使用:参数pdf_file_path(储存pdf的目标文件夹)
        
        pdf2txt:异步pdf2txt转化器
        使用：参数fold_path = ''储存pdf文件的文件夹路径，应该与上面爬虫的pdf_file_path相同
            dest_fold_path = ''储存txt文件的文件夹路径

# 2020.2.13 思路：

基于LSTM+CRF+CNN的思维导图自动转化器
基本思路：从纯文本中经过命名实体识别（NER）和关系提取（RE）后得到所有语句语义的三元组，将三元组按照一定规则构建思维导图
## 第一部分：从纯文本中提取三元组
// 通过Bi-LSTM(Long Short Term Memory)对文本进行编码，通过CRF(Condition random filed)进行解码，进行命名实体识别
 NER()
//通过CNN(Convolutional Neural Network)+Attention进行关系抽取，获得三元组
 semantic_triples = RE()
 ## 第二部分：根据三元组生成思维导图

思路一（最简单生成）：给定思维导图树的模版，固定父子关系和关系名称，从提取出来的语义三元组中选取信息填入叶子结点

思路二（思路一的扩展）：给定多种模版和多个起点，按照思路一的方式自动填充每个模版，生成多个树枝（brunch），通过某种排序方式从其中挑出主干（trunk）（比如计算每个brunch各个实体出度的总和，按照排序连接）。最终获得的树可能很大，按照层级控制变量裁剪修枝

思路三（非思维导图的关系图形）：用textrank算法获得关键词排序，与命名实体集合做交集，按照排序从最高的开始向画布中加入结点，每加入一个实体变将他作为主体的相关三元组信息加入到画布上（关系+受体），直到关键词或者三元组用完。这样做出来的其实只是一堆按照权重排序的关系图的总和
