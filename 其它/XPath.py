from lxml import etree
import requests

url = ""
resp = requests.get(url)

#解析数据
html = etree.HTML(resp.text)
divs = html.xpath("/html/body/div")
for div in divs:
    text = div.xpath("./div/a/text()")
    print(text)