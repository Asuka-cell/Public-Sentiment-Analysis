import requests

session = requests.Session()
# data ={


# }

url = "https://www.weibo.com/mygroups?gid=110017453949916"
resp = requests.get(url)
resp.encoding = 'gb2312'
print(resp.text)
