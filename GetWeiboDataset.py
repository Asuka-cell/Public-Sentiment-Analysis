import time
import json
import requests
import random
import csv
import os
from bs4 import BeautifulSoup
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 全局配置 ---

# 随机 User-Agent 列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
]

import concurrent.futures
import threading

# 代理池 (如果有有效代理，请取消注释并填入)
PROXY_LIST = [
    "39.106.192.29:8443",
    "120.55.240.71:8647",
    "47.96.42.36:80",
    "182.43.32.170:7890",
    "121.230.9.45:1080",
    "112.13.209.132:8080"
]

PROGRESS_FILE = 'progress.txt'
csv_lock = threading.Lock()

def get_random_headers(referer_url, cookies=None):
    """生成包含随机UA的Headers"""
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': referer_url,
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': 'application/json, text/plain, */*',
        # 尝试伪装成 PWA
        'MWeibo-Pwa': '1' 
    }
    if cookies and 'XSRF-TOKEN' in cookies:
        headers['X-XSRF-TOKEN'] = cookies['XSRF-TOKEN']
    return headers

def get_proxy_dict():
    """从代理池随机选择一个代理，如果池为空则返回None"""
    if not PROXY_LIST:
        return None
    chosen_proxy = random.choice(PROXY_LIST)
    return {
        "http": f"http://{chosen_proxy}",
        "https": f"http://{chosen_proxy}"
    }

def request_with_retry(url, cookies, headers, params=None, max_retries=3):
    """
    带有重试、代理轮换和指数退避的请求函数
    """
    for attempt in range(max_retries):
        proxies = get_proxy_dict()
        proxy_info = proxies['http'] if proxies else "Localhost"
        
        # 指数退避
        if attempt > 0:
            # 如果是本地IP且触发了反爬，给予更长的冷静期
            if proxy_info == "Localhost":
                backoff_time = (5 ** attempt) + random.uniform(5, 10) # 大幅增加本地重试等待 5s, 30s, ...
                print(f"      [风控冷静] 本地IP触发限制，第 {attempt} 次重试，强制等待 {backoff_time:.2f} 秒...")
            else:
                backoff_time = (2 ** attempt) + random.uniform(1, 3)
                print(f"      [重试] 第 {attempt} 次重试，等待 {backoff_time:.2f} 秒 (代理: {proxy_info})...")
            
            time.sleep(backoff_time)
        
        try:
            # 每次请求随机切换UA
            headers['User-Agent'] = random.choice(USER_AGENTS)
            
            response = requests.get(url, params=params, cookies=cookies, headers=headers, timeout=15, proxies=proxies)
            
            if response.status_code == 200:
                # 检查是否被重定向到登录页
                if "passport.weibo.com" in response.url:
                    print("      [警告] 被重定向到登录页，Cookies可能失效或IP被封。")
                    return None
                return response
            elif response.status_code == 418 or response.status_code == 403:
                print(f"      [警告] 触发反爬 (状态码 {response.status_code})，尝试切换代理重试...")
                # 如果是最后一次尝试依然403，不要直接返回None，让外层有机会处理（或者在这里多sleep一下）
            else:
                print(f"      [警告] 请求失败 (状态码 {response.status_code})")
                
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            print(f"      [网络错误] {e}，将重试...")
            continue
        except Exception as e:
            print(f"      [错误] {e}")
            break
            
    return None

def get_cookie_dict():
    """从cookies.txt加载cookie并返回字典"""
    try:
        with open("cookies.txt", "r", encoding="utf-8") as f:
            cookies_list = json.load(f)
        return {c['name']: c['value'] for c in cookies_list}
    except FileNotFoundError:
        print("cookies.txt not found.")
        return None

def clean_html(html_text):
    """去除HTML标签"""
    if not html_text:
        return ""
    return re.sub(r'<[^>]+>', '', html_text).strip()

def save_progress(index):
    """保存当前处理到的索引"""
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(index))

def load_progress():
    """加载上次处理的索引"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0

def fetch_weibo_detail(weibo_id, cookies):
    """
    获取单条微博的博文内容
    """
    # 构造移动端详情页API地址
    api_url = f"https://m.weibo.cn/statuses/show?id={weibo_id}"
    
    headers = get_random_headers(f'https://m.weibo.cn/detail/{weibo_id}', cookies)
    
    # 将 cookie 传递给 request_with_retry
    resp = request_with_retry(api_url, cookies, headers)
    
    if resp and resp.status_code == 200:
        try:
            data = resp.json()
            if data.get('ok') == 1:
                status = data.get('data', {})
                created_at = status.get('created_at')
                text = status.get('text') # 可能含HTML
                user = status.get('user', {}).get('screen_name')
                
                # 清洗文本
                clean_text = clean_html(text)
                
                # 写入博文CSV
                write_weibo_csv(weibo_id, user, created_at, clean_text)
                print(f"   [博文] 获取成功: {clean_text[:20]}...")
                return True
            else:
                print(f"   [博文] API返回非OK: {data}")
        except Exception as e:
            print(f"   [博文] 解析出错: {e}")
            
    return False

def fetch_comments(weibo_id, cookies):
    """
    获取单条微博的评论 (支持自动翻页)
    """
    api_url = 'https://m.weibo.cn/comments/hotflow'
    headers = get_random_headers(f'https://m.weibo.cn/detail/{weibo_id}', cookies)
    
    max_id = 0
    max_id_type = 0
    count = 0
    
    print(f"   [评论] 开始爬取评论...")
    
    # 获取第一页时稍微等待一下
    time.sleep(random.uniform(1, 2))
    
    while True:
        params = {
            'id': weibo_id,
            'mid': weibo_id,
            'max_id_type': max_id_type
        }
        if max_id != 0:
            params['max_id'] = max_id
            
        # 内层循环用于处理单页的重试（针对 -100 错误）
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries:
            resp = request_with_retry(api_url, cookies, headers, params=params)
            
            if resp and resp.status_code == 200:
                try:
                    data = resp.json()
                    if data.get('ok') == 1:
                        success = True
                        break # 成功拿到数据，跳出重试循环
                    elif data.get('ok') == -100:
                        retry_count += 1
                        wait_time = random.uniform(10, 20) * retry_count # 10s, 20s, 30s...
                        print(f"      [警告] 接口返回-100 (需要登录/被限流)，进入冷静期 {wait_time:.1f} 秒后重试 ({retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                        # 刷新一下header中的UA，有时候换个UA就行了
                        headers = get_random_headers(f'https://m.weibo.cn/detail/{weibo_id}', cookies)
                        continue
                    else:
                        print(f"      API返回非OK (ok={data.get('ok')}, msg={data.get('msg')})，停止抓取")
                        # 这种通常是真没数据了或者其他错误，不重试
                        success = False
                        break 
                except json.JSONDecodeError:
                    print("      JSON解析失败")
                    break
            else:
                print("      请求失败，停止抓取该微博评论")
                break
        
        if not success:
            break

        # 处理成功获取的数据
        comments_data = data.get('data', {})
        comments = comments_data.get('data', [])
        # 更新 max_id 以获取下一页
        max_id = comments_data.get('max_id', 0)
        max_id_type = comments_data.get('max_id_type', 0)
        
        if not comments:
            print("      本页无更多评论")
            break

        batch_comments = []
        for comment in comments:
            count += 1
            created_at = comment.get('created_at')
            user = comment.get('user', {}).get('screen_name')
            text = comment.get('text')
            clean_text = clean_html(text)
            
            batch_comments.append([weibo_id, user, created_at, clean_text])
        
        # 批量写入评论CSV
        write_comments_csv(batch_comments)
        print(f"      已获取 {count} 条评论 (下一页 max_id: {max_id})")
            
        if max_id == 0:
            print("      所有评论抓取完毕")
            break
            
        # 翻页随机延时 (稍微调大一点)
        time.sleep(random.uniform(3, 6))

def write_weibo_csv(weibo_id, user, created_at, text):
    with csv_lock:
        file_exists = os.path.isfile('weibo_posts.csv')
        with open('weibo_posts.csv', 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['weibo_id', 'user_name', 'publish_time', 'text'])
            writer.writerow([weibo_id, user, created_at, text])

def write_comments_csv(comments_list):
    """
    comments_list: [[weibo_id, user, created_at, text], ...]
    """
    if not comments_list:
        return
    with csv_lock:
        file_exists = os.path.isfile('weibo_comments.csv')
        with open('weibo_comments.csv', 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['weibo_id', 'user_name', 'publish_time', 'text'])
            writer.writerows(comments_list)

def process_weibo_task(weibo_id, cookies):
    """
    单个微博ID的处理任务
    """
    print(f"\n[线程] 开始处理微博 ID: {weibo_id}")
    
    # 获取博文内容
    fetch_weibo_detail(weibo_id, cookies)
    
    # 获取评论
    fetch_comments(weibo_id, cookies)
    
    # 线程结束时的随机延时
    sleep_time = random.uniform(2, 4)
    time.sleep(sleep_time)

def get_processed_ids():
    """获取已经爬取过的微博ID"""
    processed = set()
    if os.path.exists('weibo_posts.csv'):
        with open('weibo_posts.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None) # 跳过表头
            for row in reader:
                if row:
                    processed.add(row[0]) # weibo_id是第一列
    return processed

def main():
    # 1. 加载 Cookie
    cookies = get_cookie_dict()
    if not cookies:
        return

    # 2. 读取 weibo_ids.txt
    try:
        with open("weibo_ids.txt", "r", encoding="utf-8") as f:
            # 读取所有行，去除空白字符
            all_weibo_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("weibo_ids.txt not found. Please run GetWeiboIdList copy.py first.")
        return

    # 3. 过滤已爬取的ID (替代原来的 progress.txt 逻辑)
    processed_ids = get_processed_ids()
    weibo_ids_to_process = [wid for wid in all_weibo_ids if wid not in processed_ids]
    
    print(f"总共有 {len(all_weibo_ids)} 个微博ID，已处理 {len(processed_ids)} 个，剩余 {len(weibo_ids_to_process)} 个待处理...")

    if not weibo_ids_to_process:
        print("所有微博ID均已处理完毕。")
        return

    # 4. 多线程爬取
    # 建议根据电脑性能和网络情况调整 max_workers，比如 2-4
    # 警告：由于您禁用了代理池，使用本机IP多线程爬取极易触发风控，请谨慎设置线程数
    max_workers = 3 
    print(f"启动多线程爬取 (线程数: {max_workers})...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_weibo_task, wid, cookies): wid for wid in weibo_ids_to_process}
        
        # 等待完成
        for future in concurrent.futures.as_completed(futures):
            wid = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[错误] 线程处理微博 {wid} 时发生异常: {e}")

if __name__ == "__main__":
    main()
