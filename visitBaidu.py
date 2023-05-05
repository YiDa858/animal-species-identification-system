import re
import time
import requests
import json
import urllib
import os

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'}


class visitBaidu:
    def __init__(self):
        pass

    def loads_str(self, data_str):
        try:
            result = json.loads(data_str)
            # print("最终json加载结果：{}".format(result))
            return result
        except Exception as e:
            print("异常信息e：{}".format(e))
            error_index = re.findall(r"char (\d+)\)", str(e))
            if error_index:
                error_str = data_str[int(error_index[0])]
                data_str = data_str.replace(error_str, "<?>")
                # print("替换异常字符串{} 后的文本内容{}".format(error_str, data_str))
                # 该处将处理结果继续递归处理
                return self.loads_str(data_str)

    def SearchPhoto(self, key):
        word = str(key)
        page = 2
        path = 'static/search/' + str(key)  # 图片保存的路径，Search文件夹要提前建好

        if path[:-1] != '/':
            path = path + '/'
        if not os.path.exists(os.path.join("F:/Codefield/CODE_Python/BigDesign/src/flask/2022_Program_Design", path)):
            os.mkdir(os.path.join("F:/Codefield/CODE_Python/BigDesign/src/flask/2022_Program_Design", path))
        Image_path = self.Image(word, page, path, key)
        print('全部下载完成!')
        return Image_path

    def Image(self, word, page, path, key):
        n = 0
        rn = 15  # 保存的图片数量，可以修改
        pn = 1  # pn表示从第几张图片获取，默认值1
        for i in range(1, page):
            url = 'https://image.baidu.com/search/acjson?'

            param = {
                'tn': 'resultjson_com',
                'logid': '',
                'ipn': 'rj',
                'ct': '201326592',
                'is': '',
                'fp': 'result',
                'fr': '',
                'word': word,  # 图片类型
                'queryWord': word,
                'cl': '2',
                'lm': '-1',
                'ie': 'utf-8',
                'oe': 'utf-8',
                'adpicid': '',
                'st': '-1',
                'z': '',
                'ic': '0',
                'hd': '',
                'latest': '',
                'copyright': '',
                's': '',
                'se': '',
                'tab': '',
                'width': '',
                'height': '',
                'face': '0',
                'istype': '2',
                'qc': '',
                'nc': '1',
                'expermode': '',
                'nojc': '',
                'isAsync': '',
                'pn': pn,  # 从第几张图片开始
                'rn': rn,  # 爬取多少张图片
                'gsm': '',
            }
            jpgs_url = []
            names = []
            result = []
            jpgs = requests.get(url=url, headers=header, params=param)
            jpg = self.loads_str(jpgs.text)
            jpg = jpg['data']

            del jpg[-1]
            for jd in jpg:
                jpgs_url.append(jd['thumbURL'])
                names.append(jd['fromPageTitleEnc'])
            time.sleep(1)

            for (jpg_url, name) in zip(jpgs_url, names):
                try:
                    # urllib.request.urlretrieve(jpg_url, path + str(name) + '.jpg')
                    file = requests.get(jpg_url)
                    open(os.path.join("F:/Codefield/CODE_Python/BigDesign/src/flask/2022_Program_Design", path + str(name) + '.jpg'), "wb").write(
                        file.content)
                    n += 1
                    print("正在下载第" + str(n) + "张图片！")
                    file_path = path + str(name) + '.jpg'
                    result.append(file_path)
                    # print('path_result',path_result)
                except:
                    print('error')  # 有的图片不能成功下载，实际图片会小于15个，建议前端判别一下list长度
                    continue
            pn += rn
            return result

    def getFileList(self, key):
        return self.SearchPhoto(key)

    if __name__ == '__main__':
        animal = ['狗', '猫', '猪', '兔子', '鼠']
        for item in animal:
            Image_Path = SearchPhoto(item)
            print(Image_Path)
