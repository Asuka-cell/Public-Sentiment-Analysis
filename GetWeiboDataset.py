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
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/123.0.0.0 Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36',
    'Mozilla/5.0 (Linux; Android 13; SM-S9180) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
]

import threading

# 代理池 (如果有有效代理，请取消注释并填入)
PROXY_LIST = [
    # "39.106.192.29:8443",
    # "120.55.240.71:8647",
    # "47.96.42.36:80",
    # "182.43.32.170:7890",
    # "121.230.9.45:1080",
    # "112.13.209.132:8080"
]

PROGRESS_FILE = 'progress.txt'
csv_lock = threading.Lock()

REQUEST_DELAY_RANGE = (1.2, 2.5)
PAGE_DELAY_RANGE = (3.0, 6.0)
TASK_DELAY_RANGE = (4.0, 8.0)
PROXY_COOLDOWN_SECONDS = 600
UA_COOLDOWN_SECONDS = 600
PROXY_RECHECK_SECONDS = 1800
PROXY_FAIL_COOLDOWN_SECONDS = 900

proxy_cooldown_until = {}
proxy_block_counts = {}
expected_comment_counts = {}
ua_cooldown_until = {}
ua_block_counts = {}
proxy_last_checked = {}
proxy_valid_cache = {}
proxy_fail_counts = {}
proxy_success_counts = {}

def human_sleep(base_range, long_prob=0.08, long_range=(8, 15)):
    t = random.uniform(*base_range)
    time.sleep(t)
    if random.random() < long_prob:
        time.sleep(random.uniform(*long_range))

def build_proxies(proxy_entry):
    if proxy_entry.startswith('http://') or proxy_entry.startswith('https://'):
        url = proxy_entry
    else:
        url = 'http://' + proxy_entry
    return {'http': url, 'https': url}

def probe_proxy(proxies):
    try:
        r = requests.get('https://weibo.com/robots.txt', timeout=8, proxies=proxies, headers={'User-Agent': random.choice(USER_AGENTS), 'Accept': 'text/plain'})
        return r.status_code < 400
    except Exception:
        return False

def get_random_headers(referer_url, cookies=None):
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Referer': referer_url,
        'Origin': 'https://weibo.com' if 'weibo.com' in referer_url else 'https://m.weibo.cn',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Upgrade-Insecure-Requests': '1',
        'X-Requested-With': 'XMLHttpRequest'
    }
    if cookies and 'XSRF-TOKEN' in cookies:
        headers['X-XSRF-TOKEN'] = cookies['XSRF-TOKEN']
    return headers

def get_proxy_dict():
    if not PROXY_LIST:
        return None, "Localhost"
    now = time.time()
    candidates = [p for p in PROXY_LIST if proxy_cooldown_until.get(p, 0) <= now]
    if not candidates:
        candidates = PROXY_LIST[:]
    random.shuffle(candidates)
    for p in candidates:
        if proxy_valid_cache.get(p) is False:
            continue
        if now - proxy_last_checked.get(p, 0) > PROXY_RECHECK_SECONDS or proxy_valid_cache.get(p) is None:
            prx = build_proxies(p)
            ok = probe_proxy(prx)
            proxy_valid_cache[p] = ok
            proxy_last_checked[p] = now
            if not ok:
                proxy_cooldown_until[p] = now + PROXY_FAIL_COOLDOWN_SECONDS
                proxy_fail_counts[p] = proxy_fail_counts.get(p, 0) + 1
                continue
        prx = build_proxies(p)
        return prx, p
    return None, "Localhost"

def get_random_ua():
    now = time.time()
    available = [ua for ua in USER_AGENTS if ua_cooldown_until.get(ua, 0) <= now]
    if not available:
        return random.choice(USER_AGENTS)
    return random.choice(available)

def request_with_retry(url, cookies, headers, params=None, max_retries=3):
    """
    带有重试、代理轮换和指数退避的请求函数
    """
    for attempt in range(max_retries):
        proxies, proxy_label = get_proxy_dict()
        proxy_info = proxies['http'] if proxies else "Localhost"
        
        # 指数退避
        if attempt > 0:
            if proxy_info == "Localhost":
                backoff_time = min(20, (2 ** attempt) + random.uniform(1, 3))
                print(f"      [风控冷静] 本地IP限制，第 {attempt} 次重试，等待 {backoff_time:.2f} 秒...")
            else:
                backoff_time = min(12, (1.5 ** attempt) + random.uniform(1, 2))
                print(f"      [重试] 第 {attempt} 次重试，等待 {backoff_time:.2f} 秒 (代理: {proxy_info})...")
            
            time.sleep(backoff_time)
        
        try:
            ua_label = get_random_ua()
            headers['User-Agent'] = ua_label

            human_sleep(REQUEST_DELAY_RANGE, long_prob=0.06, long_range=(5, 10))
            response = requests.get(url, params=params, cookies=cookies, headers=headers, timeout=15, proxies=proxies)

            if response.status_code == 200:
                # 检查是否被重定向到登录页
                if "passport.weibo.com" in response.url:
                    ua_block_counts[ua_label] = ua_block_counts.get(ua_label, 0) + 1
                    ua_cooldown_until[ua_label] = time.time() + UA_COOLDOWN_SECONDS
                    print(f"      [反爬] UA 被限制，进入冷却 {UA_COOLDOWN_SECONDS}s")
                    print(f"             UA: {ua_label}")
                    return None
                return response
            elif response.status_code == 400:
                print("      [错误] 400 Bad Request，可能为参数或CSRF校验不通过")
                print(f"             URL: {url}")
                print(f"             Params: {params}")
                return None
            elif response.status_code == 418 or response.status_code == 403:
                if proxy_label != "Localhost":
                    proxy_block_counts[proxy_label] = proxy_block_counts.get(proxy_label, 0) + 1
                    proxy_cooldown_until[proxy_label] = time.time() + PROXY_COOLDOWN_SECONDS
                    print(f"      [反爬] 代理 {proxy_label} 被限制 (状态码 {response.status_code})，进入冷却 {PROXY_COOLDOWN_SECONDS}s")
                else:
                    print(f"      [反爬] 本地IP被限制 (状态码 {response.status_code})")
                ua_block_counts[ua_label] = ua_block_counts.get(ua_label, 0) + 1
                ua_cooldown_until[ua_label] = time.time() + UA_COOLDOWN_SECONDS
                print(f"      [反爬] UA 被限制，进入冷却 {UA_COOLDOWN_SECONDS}s")
                print(f"             UA: {ua_label}")
                print("      [警告] 触发反爬，准备重试/切换代理...")
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
    pc_url = f"https://weibo.com/ajax/statuses/show?id={weibo_id}"
    headers = get_random_headers('https://weibo.com/', cookies)
    resp = request_with_retry(pc_url, cookies, headers)
    
    if resp and resp.status_code == 200:
        try:
            data = resp.json()
            if data.get('ok') == 1:
                status = data.get('data', {})
                created_at = status.get('created_at')
                text = status.get('text') or status.get('text_raw')
                user = (status.get('user') or {}).get('screen_name')
                expected_comment_counts[weibo_id] = status.get('comments_count', 0) or status.get('comments_count', 0)
                
                # 清洗文本
                clean_text = clean_html(text)
                
                # 写入博文CSV
                write_weibo_csv(weibo_id, user, created_at, clean_text)
                print(f"   [博文] 获取成功: {clean_text[:20]}...")
                return True
            else:
                print(f"   [博文] PC接口返回非OK，尝试移动端")
        except Exception as e:
            print(f"   [博文] 解析出错: {e}")
            
    m_url = f"https://m.weibo.cn/statuses/show?id={weibo_id}"
    headers = get_random_headers('https://m.weibo.cn/', cookies)
    resp = request_with_retry(m_url, cookies, headers)
    if resp and resp.status_code == 200:
        try:
            data = resp.json()
            if data.get('ok') == 1:
                status = data.get('data', {})
                created_at = status.get('created_at')
                text = status.get('text')
                user = (status.get('user') or {}).get('screen_name')
                expected_comment_counts[weibo_id] = status.get('comments_count', 0) or 0
                clean_text = clean_html(text)
                write_weibo_csv(weibo_id, user, created_at, clean_text)
                print(f"   [博文] 移动端获取成功: {clean_text[:20]}...")
                return True
        except Exception as e:
            print(f"   [博文] 移动端解析出错: {e}")
    return False

def fetch_comments(weibo_id, cookies):
    api_urls = [
        ('pc_build', 'https://weibo.com/ajax/statuses/buildComments'),
        ('pc_flow', 'https://weibo.com/ajax/statuses/flowComments'),
        ('m_hotflow', 'https://m.weibo.cn/comments/hotflow')
    ]
    headers = get_random_headers('https://weibo.com/', cookies)

    max_id = 0
    max_id_type = 0
    count = 0
    page_no = 0
    expected_total = int(expected_comment_counts.get(weibo_id, 0) or 0)
    last_progress_count = 0
    stuck_rounds = 0

    print(f"   [评论] 开始爬取评论... (目标: {expected_total if expected_total > 0 else '未知'} )")

    human_sleep(PAGE_DELAY_RANGE, long_prob=0.08, long_range=(8, 20))

    while True:
        page_no += 1
        chosen = None
        for tag, url in api_urls:
            chosen = (tag, url)
            break
        tag, api_url = chosen
        if tag.startswith('pc_'):
            params = {'id': weibo_id, 'count': 20, 'is_reload': 1, 'is_show_bulletin': 2}
            if max_id:
                params['max_id'] = max_id
            params['max_id_type'] = max_id_type
            params['is_hot'] = 0
            if tag == 'pc_flow':
                params['is_new_segment'] = 1
        else:
            params = {'id': weibo_id, 'mid': weibo_id, 'max_id_type': max_id_type}
            if max_id:
                params['max_id'] = max_id
            
        max_retries = 5
        retry_count = 0
        success = False

        while retry_count < max_retries:
            resp = request_with_retry(api_url, cookies, headers, params=params)

            if resp and resp.status_code == 200:
                try:
                    data = resp.json()
                    if data.get('ok') == 1:
                        success = True
                        break
                    if data.get('ok') == -100:
                        retry_count += 1
                        wait_time = random.uniform(6, 12) * retry_count
                        print(f"      [限流] ok=-100，等待 {wait_time:.1f}s 后重试 ({retry_count}/{max_retries})")
                        time.sleep(wait_time)
                        headers = get_random_headers(f'https://weibo.com/detail/{weibo_id}', cookies)
                        continue
                    retry_count += 1
                    wait_time = random.uniform(2, 5) * retry_count
                    print(f"      [异常] ok={data.get('ok')}, msg={data.get('msg')}，等待 {wait_time:.1f}s 后重试 ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                    headers = get_random_headers(f'https://weibo.com/detail/{weibo_id}', cookies)
                    continue
                except json.JSONDecodeError:
                    retry_count += 1
                    wait_time = random.uniform(2, 5) * retry_count
                    print(f"      [异常] JSON解析失败，等待 {wait_time:.1f}s 后重试 ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                    headers = get_random_headers(f'https://weibo.com/detail/{weibo_id}', cookies)
                    continue

            retry_count += 1
            wait_time = random.uniform(2, 5) * retry_count
            print(f"      [异常] 请求失败/无响应，等待 {wait_time:.1f}s 后重试 ({retry_count}/{max_retries})")
            time.sleep(wait_time)
            headers = get_random_headers(f'https://weibo.com/detail/{weibo_id}', cookies)
            continue
        
        if not success:
            if tag != 'm_hotflow':
                api_urls = api_urls[1:] + api_urls[:1]
                print("      [切换接口] 当前PC接口失败，尝试下一个接口")
                continue
            break

        comments_data = data.get('data', {})
        if isinstance(comments_data, list):
            comments = comments_data
            next_max_id = data.get('max_id', 0)
            next_max_id_type = data.get('max_id_type', 0)
        else:
            comments = comments_data.get('data') or comments_data.get('root_comments') or comments_data.get('root') or []
            next_max_id = comments_data.get('max_id', 0)
            next_max_id_type = comments_data.get('max_id_type', 0)

        if not comments:
            print("      本页无更多评论")
            break

        batch_comments = []
        for comment in comments:
            count += 1
            created_at = comment.get('created_at')
            user = (comment.get('user') or {}).get('screen_name')
            text = comment.get('text') or comment.get('text_raw')
            clean_text = clean_html(text)
            batch_comments.append([weibo_id, user, created_at, clean_text])

        write_comments_csv(batch_comments)

        if expected_total > 0 and count >= expected_total:
            print(f"      已达到目标评论数: {count}/{expected_total}")
            break

        if count == last_progress_count:
            stuck_rounds += 1
        else:
            stuck_rounds = 0
        last_progress_count = count

        if stuck_rounds >= 3:
            print("      连续多页无新增，停止翻页")
            break

        if next_max_id == 0:
            print("      所有评论抓取完毕")
            break

        if next_max_id == max_id and next_max_id_type == max_id_type:
            print("      翻页游标无变化，停止翻页")
            break

        max_id = next_max_id
        max_id_type = next_max_id_type
        print(f"      已获取 {count} 条评论 (第{page_no}页, 下一页 max_id: {max_id}, max_id_type: {max_id_type})")
        human_sleep(PAGE_DELAY_RANGE, long_prob=0.08, long_range=(8, 20))

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
    human_sleep((1.0, 3.0), long_prob=0.1, long_range=(6, 12))
    if random.random() < 0.4:
        fetch_comments(weibo_id, cookies)
        human_sleep((1.2, 2.8), long_prob=0.08, long_range=(5, 10))
        fetch_weibo_detail(weibo_id, cookies)
    else:
        fetch_weibo_detail(weibo_id, cookies)
        human_sleep((1.2, 2.8), long_prob=0.08, long_range=(5, 10))
        fetch_comments(weibo_id, cookies)
    human_sleep(TASK_DELAY_RANGE, long_prob=0.15, long_range=(10, 25))

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

    for wid in weibo_ids_to_process:
        try:
            process_weibo_task(wid, cookies)
        except Exception as e:
            print(f"[错误] 处理微博 {wid} 时发生异常: {e}")

if __name__ == "__main__":
    main()
