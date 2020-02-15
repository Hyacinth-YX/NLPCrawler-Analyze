import sys

sys.path.append("../")

from bs4 import  BeautifulSoup

import json

import re

from utils import *

page_url = lambda i:"http://www.cas.cn/kx/kpwz/index_{}.shtml".format(i) if i > 1 else "http://www.cas.cn/kx/kpwz"


def parse_text_url(res):

    page = BeautifulSoup(res,"html.parser")

    content = page.find(attrs={"id":"content"})

    links = content.find_all('a')

    return [link['href'] for link in links]


def getTextUrls():


    get_text_url_cons = [Connection(callback = CallBack(func = parse_text_url),url = page_url(i)) for i in range(60)]

    muti = MultiConnections(get_text_url_cons)

    urls = json.dumps(muti.start())

    with open("text_urls.json",'w') as f:

        f.write(urls)

def parseContent(res):

    res = BeautifulSoup(res,"html.parser")

    content = res.find(attrs={"class":"TRS_Editor"})

    styles =content.find_all("style")

    Invalid_char = ["\\","/","*",":","?","*","<",">","|"]

    title = res.find(attrs={"class":"xl_title"}).text

    for char in Invalid_char:

        title = title.replace(char,"")

    subtitles = content.find_all("strong")

    content = content.text

    for style in styles:

        content = content.replace(style.text,"")

    for subtitle in subtitles:

        text = subtitle.text

        content = content.replace(text,"<subtitle>{}</subtitle>".format(text))

    return {"title":title,"content":content.replace(" ","").replace("\u3000","").replace("\n\n","")}

def getContent():


    with open("text_urls.json", 'r') as f:

        pagesUrls = json.loads(f.read())


    urls = []

    for i in range(60):

        for j in range(len(pagesUrls[i])):

            if "weixin" not in pagesUrls[i][j]:

                urls.append("http://www.cas.cn/kx/kpwz" + pagesUrls[i][j][1:])

    print("total doc num :",len(urls))

    get_content_cons = [Connection(callback = CallBack(parseContent),url = url) for url in urls]

    muti = MultiConnections(get_content_cons)

    pageContents = muti.start()

    for content in pageContents:

        with open("data/{}.txt".format(content["title"]),'w',encoding="utf-8") as f:

            f.write(content["content"])

if __name__ == "__main__":

    #getTextUrls()

    #getContent()

    import os

    texts = os.listdir("./data/")

    count = 0

    for text in texts:

        with open("./data/"+text,'r',encoding="utf-8") as f:

            data = f.read()

            count += len(data.replace("<subtitle>","").replace("</subtitle>",""))

            print("\r字数:",count,end="",flush=True)