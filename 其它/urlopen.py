from urllib.request import urlopen

url = "https://www.baidu.com"
response = urlopen(url)

print(response.read().decode("utf-8"))