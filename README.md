# NLPCrawler-Analyze
爬取论文并储存为txt格式，对每篇论文进行标记、分析、处理获得容易理解的思维导图

2020.1.31 爬虫部分+pdf2txt基本完成，爬虫位于Crawler&Convert文件夹中
    Crawler&Convert：
    
        main.py: 整合爬虫与文本类型转化
        
        NLPCrawler3.py：用python写的特制爬虫，无法爬取其他网站
        爬取网站（https://www.aclweb.org/anthology/）
        使用:参数pdf_file_path(储存pdf的目标文件夹)
        #TODO: 
        爬虫速度慢，没有异步化，有时间可以改进为异步爬虫
        ip池需要增加，之前没有伪装ip我已经被封了，可能是我的网络原因，使用代理ip后仍然无法快速正常下载，经常失败
        
        pdf2txt:异步pdf2txt转化器
        使用：参数fold_path = ''储存pdf文件的文件夹路径，应该与上面爬虫的pdf_file_path相同
            dest_fold_path = ''储存txt文件的文件夹路径
        #TODO：
        使用时需要修改fold
        经过测试大小比较小的pdf可以快速转化，但是大的pdf会卡住，甚至进程拥堵，不知道怎么设置一个超时结束或者改成同步方法？
        对于pdf的目录的处理，很多'.'和数字之类的处理不好，出现很多无意义换行

下一步需要拉一个服务器，让服务器自己跑着收集素材，并且完成素材的清洗工作

但是最主要的还是语言模型和工具的构建和寻找
