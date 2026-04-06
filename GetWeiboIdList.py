import requests
import json
import urllib.parse
import re
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_pc_search_weibo_ids(keyword, max_pages=50):
    """
    使用 PC 版搜索 (s.weibo.com) 获取博文ID，支持翻页
    :param keyword: 搜索关键词
    :param max_pages: 最大抓取页数，默认50页
    """
    # 1. 加载 Cookies
    try:
        with open("cookies.txt", "r", encoding="utf-8") as f:
            cookies_list = json.load(f)
    except FileNotFoundError:
        print("cookies.txt not found.")
        return []

    # 将 Cookie 列表转换为字典供 requests 使用
    cookies = {c['name']: c['value'] for c in cookies_list}
    
    encoded_query = urllib.parse.quote(keyword)
    base_url = f"https://s.weibo.com/weibo?q={encoded_query}"
    
    # 随机 User-Agent 列表
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
    ]

    headers = {
        'Referer': 'https://s.weibo.com/',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    }
    
    # --- 代理池 ---
    proxy_list = [
        # "220.197.44.36:3128",
        # "203.196.8.6:3128",
        # "36.133.208.130:8888"
    ]
    
    all_weibo_ids = []
    
    for page in range(1, max_pages + 1):
        print(f"\n>>> 正在抓取第 {page} 页...")
        
        # 构造分页 URL
        # PC版微博搜索分页参数: &page=2
        current_url = f"{base_url}&page={page}"
        
        page_ids = []
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 动态切换 User-Agent
                headers['User-Agent'] = random.choice(user_agents)
                
                # 代理选择逻辑
                if not proxy_list:
                    proxies = None
                    chosen_proxy = "Localhost"
                else:
                    chosen_proxy = random.choice(proxy_list)
                    proxies = {
                        "http": f"http://{chosen_proxy}",
                        "https": f"http://{chosen_proxy}"
                    }
                
                # 指数退避与随机延时
                if attempt > 0:
                    backoff_time = (2 ** attempt) + random.uniform(1, 3)
                    print(f"   [重试] 第 {attempt} 次重试，等待 {backoff_time:.2f} 秒后切换代理 {chosen_proxy}...")
                    time.sleep(backoff_time)
                else:
                    # 翻页时的随机延时 (2-5秒，比单页更长一点)
                    sleep_time = random.uniform(2, 5)
                    print(f"   [提示] 随机等待 {sleep_time:.2f} 秒，使用代理: {chosen_proxy}")
                    time.sleep(sleep_time)

                print(f"   正在访问: {current_url}")
                
                response = requests.get(current_url, cookies=cookies, headers=headers, timeout=15, proxies=proxies)
                
                if response.status_code == 200:
                    html = response.text
                    
                    if "passport.weibo.com" in response.url:
                        print("   !!! 错误: 被重定向到登录页，Cookies 可能失效。")
                        return all_weibo_ids # 停止后续抓取
                    
                    if "抱歉，未找到" in html:
                        print("   未找到更多结果，停止翻页。")
                        return all_weibo_ids

                    # 解析 HTML 提取 mid
                    pattern = re.compile(r'mid="(\d{16,})"', re.DOTALL)
                    found_ids = pattern.findall(html)
                    found_ids = list(set(found_ids)) # 去重
                    
                    if not found_ids:
                        print("   当前页未提取到ID (可能被反爬或无数据)。")
                        # 如果连续两页都没数据，可能需要考虑停止，这里暂时仅跳过
                    else:
                        print(f"   成功找到 {len(found_ids)} 条微博 ID")
                        for mid in found_ids:
                            # 避免全局重复
                            if mid not in all_weibo_ids:
                                all_weibo_ids.append(mid)
                                page_ids.append(mid)
                    
                    # 成功抓取本页，跳出重试循环，进行下一页
                    break 
                    
                else:
                    print(f"   请求失败: {response.status_code}")

            except (requests.exceptions.ProxyError, requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                print(f"   [警告] 代理 {chosen_proxy} 连接失败或超时: {e}")
                continue
            except Exception as e:
                print(f"   [错误] 发生未预期错误: {e}")
                break
        else:
            print("   [失败] 本页达到最大重试次数，跳过。")
            
    return all_weibo_ids

if __name__ == "__main__":
    keyword = input("请输入搜索关键词 (例如 '西贝预制菜'): ")
    # 默认抓取 10 页，您可以修改这个数字
    ids = get_pc_search_weibo_ids(keyword, max_pages=50)
    
    if ids:
        print(f"\n共获取到 {len(ids)} 个有效 ID (已去重)")
        with open("weibo_ids.txt", "w") as f:
            for i in ids:
                f.write(i + "\n")
        print("ID 已保存到 weibo_ids.txt")
