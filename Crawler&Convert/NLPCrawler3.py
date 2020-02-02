# coding=utf-8
import urllib.response
import urllib.request
import urllib.error
import re
import time
import socket
import ssl
import random

socket.setdefaulttimeout(30)
ssl._create_default_https_context = ssl._create_unverified_context

root_url = 'http://www.aclweb.org' # 根网址
list_url = 'https://www.aclweb.org/anthology/venues' # 会议目录，加上会议地点名称可以找到会议档案
selecable = ['acl','anlp','cl','conll','eacl','nemnlp','naacl',
             'semeval','tacl','WS','alta','coling','hlt','ijcnlp',
             'jep-taln-recital','muc','paclic','ranlp','rocling-ijclclp',
             'tinlap','tipster']

def printPercent(blocknum, blocksize, totalsize):
    '''
    :param blocknum: 当前块编号
    :param blocksize: 单次传输块大小
    :param totalsize: 网页文件总大小
    :打印下载进度
    '''
    percent = 100 * blocknum * blocksize / totalsize
    if percent > 100.0:
        percent = 100.0
    print("download : %.2f%%" % (percent))


# url:下载地址 filename:保存到本地路径
def downPaper(url, filename, no):
    '''
    :param url: 下载地址
    :param filename: 保存到本地路径
    :param no: 第几号论文正在下载
    :从远程下载论文pdf文件
    '''
    print ('PDF file[' + str(no) + '] downloading\n')
    try:
        set_proxy()
        urllib.request.urlretrieve (url, filename, printPercent)
        time.sleep (random.random())
    except urllib.error.URLError as e:
        print(e.reason)
        count = 1
        while count <= 5:
            time.sleep (random.random ())
            try:
                set_proxy()
                urllib.request.urlretrieve(url,filename,printPercent)
                break
            except urllib.error.URLError as e:
                print(e.reason)
                err_info = 'Reloading for '+str(count)+'time.\n'
                print(err_info)
                count += 1
        if count > 5:
            print ('downloading picture failed')
            return False
    print ('PDF file[' + str(no) + '] downloaded\n')


def get_headers():
    '''
    随机获取一个headers
    :return:
    '''
    user_agents =  ['Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
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
    headers = {'User-Agent':random.choice(user_agents)}
    return headers

def set_proxy():
    proxy_pool = ["39.137.69.7:8080", "223.199.18.30:9999", "221.178.176.25:3128",
                  "39.106.66.178:80", "58.222.32.77:8080", "47.99.236.251:3128"]
    proxy = {"http": random.choice(proxy_pool)}
    proxy_handler = urllib.request.ProxyHandler(proxy)
    opener = urllib.request.build_opener(proxy_handler)
    urllib.request.install_opener(opener)


def loadPage(url):
    '''
    通过url伪装用户请求网页内容，如果失败再尝试一次，仍然失败则返回False
    :param url: url地址 （string）
    :return: html 请求所得内容（string）
    '''

    try:
        set_proxy()
        print(url + u' loading first time...')
        request = urllib.request.Request(url, headers=get_headers())
        html = urllib.request.urlopen(request,timeout=15).read()
        time.sleep(random.random())
    except urllib.error.URLError as e:
        print(e.reason)
        try:
            set_proxy()
            print(url + u' loading second time...')
            request = urllib.request.Request (url, headers=get_headers())
            html = urllib.request.urlopen (request,timeout=15).read ()
        except urllib.error.URLError as e:
            print(e.reason)
            print(url + u' loading false...')
            return False
    return html

def getUrlListByLocationAndYear ():
    '''
    获得需要下载的pdf的所有url
    :return: urlList (list)
    '''
    urlList = []
    selections = input(u'please input the location you want to search, seperate by space.(sach as: acl cl emnlp)\nacl\tanlp\tcl\tconll\teacl\nemnlp\tnaacl\tsemeval\ttacl\tWS\nalta\tcoling\thlt\tijcnlp\tjep-taln-recital\nmuc\tpaclic\tranlp\trocling-ijclclp\ntinlap\ttipster\n')
    selections = selections.lower().split(' ')
    locationUrls = []
    for selection in selections:
        if selection not in selecable:
            print ('no such location')
            return False
        else:
            locationUrls.append(list_url + '/' + selection)
    # 获得想要选择的时间内容
    timeSelect = input(u'please input the year you want to search, seperate by "-".(such as 2019 or 2000-2019)')
    timeSelect = timeSelect.split('-')
    timeSelected = set()
    if len(timeSelect) == 1 and timeSelect[0].isdigit() :
        timeSelected.add(int(timeSelect[0]))
    elif timeSelect[0].isdigit() and timeSelect[1].isdigit() and int(timeSelect[0]) < int(timeSelect[1]):
        for year in range(int(timeSelect[0]),int(timeSelect[1])+1):
            timeSelected.add(int(year))
    else:
        print ('time error')
        return False
    # 得该地点各年份的url，并按照timeselected请求该年份的论文的所有pdf的url链接，加入到urllist
    count = 1
    selectedYearList = []
    for locationUrl in locationUrls:
        print(str(count) + u' loading location url index...\n')
        html = loadPage(locationUrl)
        if html == False:
            continue
        print ('seccess\n')
        pattern = r'/anthology/events/[\w]*-[0-9]{1,4}/'
        regex = re.compile (pattern)
        timeList = re.findall(regex,str(html))
        for item in timeList:
            year = int(str(item).split('-')[1].split('/')[0])
            if year in timeSelected:
                url = root_url + str (item)
                selectedYearList.append (url)
        count += 1
    if len(selectedYearList) == 0:
        print ('get year list failed.\n')
    else:
        count = 1
        for indexUrl in selectedYearList:
            print(str(count) + u') loading selected url index...\n')
            html = loadPage(indexUrl)
            count += 1
            if html == False:
                print ('Failed\n')
            else:
                print('success\n')
                pattern = r'https://www.aclweb.org/anthology/\w{3}-\d*\.pdf'
                regex = re.compile(pattern)
                urls = re.findall(regex,str(html))
                for url in urls:
                    urlList.append(url)
    return urlList



def NLPCrawler3(pdf_file_path):
    no = 1
    while True:
        selection = int (input (
            u'select funtion:\n1--catch pdf by year and conference location\n2--catch pdf by searching key word\n0--quit\n'))
        if (selection == 1):
            maximum = int (input (u'the maximum number of papers you want:'))
            urlList = getUrlListByLocationAndYear ()
            if len (urlList) == 0:
                continue
            else:
                for url in urlList:
                    if no > maximum:
                        break
                    newFilePath = pdf_file_path + '/' + str (url).strip ().split ('/')[-1]
                    downPaper (url, newFilePath, no)
                    no += 1
        elif (selection == 2):
            print ('sorry,this function is in building. please select once more.\n')
            continue
        else:
            print ('well down!')
            break


if __name__ == '__main__':
    pdf_file_path = 'D:\\nlp_pdf'  # 指定目标文件夹（确定已有该文件夹）
    NLPCrawler3(pdf_file_path)