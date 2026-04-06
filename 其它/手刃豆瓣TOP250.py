import requests
import re

url = "https://movie.douban.com/top250"

headers = {
    "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0"
}

resp = requests.get(url, headers=headers)
s = resp.text

obj = re.compile(r'<div class="pic">.*?<img width="100" alt="(?P<title>.*?)".*?</div>'
                 r'.*?<br>(?P<year>.*?)&nbsp.*?</p>'
                 r'.*?<span class="rating_num" property="v:average">(?P<rating>.*?)</span>'
                 r'.*?<span>(?P<comments>.*?)</span>'
                 , re.S)

import csv
f = open("data.csv", mode="w", encoding="utf-8")
csvwriter = csv.writer(f)

list = obj.finditer(s)
for i in list:
    # print(i.group("title"))
    # print(i.group("year").strip())
    # print(i.group("rating"))
    # print(i.group("comments"))
    dic = i.groupdict()
    dic["year"] = dic["year"].strip()
    dic["comments"] = dic["comments"].strip()
    csvwriter.writerow(dic.values())

f.close()
print("over!")