from bs4 import BeautifulSoup
import requests


class downloader(object):

    # 初始化
    def __init__(self):
        self.server = 'http://www.biqukan.com'
        self.target = 'https://www.bqkan.com/3_3026'
        self.names = []  # 存放章节名
        self.urls = []  # 存放章节链接
        self.nums = 0  # 章节数

    # 获取完整章节地址
    def get_download_url(self):
        req = requests.get(url=self.target)
        html = req.text
        div_bf = BeautifulSoup(html, "html.parser")
        div = div_bf.find_all('div', class_='listmain')
        a_bf = BeautifulSoup(str(div[0]), "html.parser")
        a = a_bf.find_all("a")
        self.nums = len(a)  # 统计章节数
        for each in a:
            print(each.string, self.server + each.get('href'))
            self.names.append(each.string)
            self.urls.append(self.server + each.get('href'))

    # 获取对应链接的地址
    def get_contents(self, target):
        req = requests.get(url=target)
        html = req.text
        bf = BeautifulSoup(html, "html.parser")
        texts = bf.find_all('div', class_='showtxt')
        texts = texts[0].text.replace('\xa0' * 8, '\n\n')
        return texts

    # 将内容写入磁盘中
    def writer(self, name, path, text):
        write_flag = True
        with open(path, 'w', encoding='utf-8') as f:
            f.write(name + '\n')
            f.writelines(text)
            f.write('\n\n')


if __name__ == "__main__":
    dl = downloader()
    dl.get_download_url()
    print('《斗罗大陆》开始下载：')
    for i in range(dl.nums):
        print("正在下载=>", dl.names[i])
        dl.writer(dl.names[i], 'E:\\斗罗大陆\\' + dl.names[i] +
                  '.txt', dl.get_contents(dl.urls[i]))
    print('《斗罗大陆》下载完成!')
