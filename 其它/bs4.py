from bs4 import BeautifulSoup
import requests

url = "http://www.xinfadi.com.cn/priceDetail.html"

resp = requests.get(url)

#解析数据
#页面源代码交给bs处理，生成bs对象
page = BeautifulSoup(resp.text, "html.parser")
table = page.find("table", attrs={"class":""})
#find(标签, 属性=值)
#find_all(标签, 属性=值)
#table = page.find_all("table", attrs={"class":""})