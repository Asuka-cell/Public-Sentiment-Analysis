import requests

url = "https://movie.douban.com/j/chart/top_list"

param = {
    "type": "24",
    "interval_id": "100:90",
    "action":"",
    "start":0,
    "limit":20
}

headers = {
    "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0"
}

response = requests.get(url, params=param, headers=headers)

print(response.json())
response.close()
