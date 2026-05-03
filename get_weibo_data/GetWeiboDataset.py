import time
import json
import requests
import random
import csv
import os
import re

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

PROGRESS_FILE = os.path.join(BASE_DIR, "progress.txt")
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

LATEST_ID_FILE = os.path.join(BASE_DIR, "latest_weibo_id.json")
WEIBO_TEXTS_CSV = os.path.join(DATASET_DIR, "weibo_posts.csv")
WEIBO_COMMENTS_CSV = os.path.join(DATASET_DIR, "weibo_comments.csv")
DEBUG_DUMP_DIR = os.path.join(BASE_DIR, "debug_weibo")
DEBUG_DUMP_LIMIT = 5
debug_dump_count = 0
PROXY_SOURCE_FILE = os.path.join(BASE_DIR, "proxies.txt")
proxy_source_mtime = 0.0

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
    global proxy_source_mtime
    try:
        if os.path.exists(PROXY_SOURCE_FILE):
            mtime = os.path.getmtime(PROXY_SOURCE_FILE)
            if mtime != proxy_source_mtime:
                proxy_source_mtime = mtime
                try:
                    with open(PROXY_SOURCE_FILE, "r", encoding="utf-8") as f:
                        loaded = [line.strip() for line in f if line.strip()]
                    loaded = [p for p in loaded if p and not p.startswith("#")]
                    if loaded:
                        PROXY_LIST[:] = loaded
                        print(f"[代理] 已加载 {len(PROXY_LIST)} 个代理 (来自 {PROXY_SOURCE_FILE})")
                except Exception as e:
                    print(f"[代理] 读取 {PROXY_SOURCE_FILE} 失败: {e}")
    except Exception:
        pass

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
        with open(os.path.join(BASE_DIR, "cookies.txt"), "r", encoding="utf-8") as f:
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

def parse_weibo_id(value):
    try:
        return int(str(value).strip())
    except Exception:
        return None

def infer_latest_id_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return 0
    latest = 0
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                wid = parse_weibo_id(row[0])
                if wid is not None and wid > latest:
                    latest = wid
    except Exception:
        return 0
    return latest

def load_latest_ids():
    defaults = {
        "latest_comment_id": 0,
        "latest_post_id": 0,
    }
    if os.path.exists(LATEST_ID_FILE):
        try:
            with open(LATEST_ID_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in list(defaults.keys()):
                v = data.get(k, defaults[k])
                defaults[k] = int(v) if str(v).isdigit() else int(v)
        except Exception:
            pass

    if defaults["latest_comment_id"] <= 0:
        defaults["latest_comment_id"] = infer_latest_id_from_csv(WEIBO_COMMENTS_CSV)
    if defaults["latest_post_id"] <= 0:
        defaults["latest_post_id"] = infer_latest_id_from_csv(WEIBO_TEXTS_CSV)
    return defaults

def save_latest_ids(latest_ids):
    data = {
        "latest_comment_id": int(latest_ids.get("latest_comment_id", 0) or 0),
        "latest_post_id": int(latest_ids.get("latest_post_id", 0) or 0),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(LATEST_ID_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[警告] 写入 {LATEST_ID_FILE} 失败: {e}")

def ensure_csv_header(file_path, header):
    if not os.path.exists(file_path):
        return
    try:
        if os.path.getsize(file_path) == 0:
            with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
            return
    except Exception:
        return

    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            first_line = f.readline()
        if first_line.strip().startswith(header[0]):
            return
    except Exception:
        return

    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            content = f.read()
        with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            f.write(content)
    except Exception as e:
        print(f"[警告] 修复表头失败 {file_path}: {e}")

def extract_status_fields(status):
    if not isinstance(status, dict):
        return None, None, None
    created_at = status.get("created_at")
    user = (status.get("user") or {}).get("screen_name")
    text = status.get("text_raw") or status.get("text") or status.get("raw_text")
    return created_at, user, text


def dump_debug_json(prefix, weibo_id, payload):
    global debug_dump_count
    if debug_dump_count >= DEBUG_DUMP_LIMIT:
        return
    debug_dump_count += 1
    try:
        os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
        path = os.path.join(DEBUG_DUMP_DIR, f"{prefix}_{weibo_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"   [调试] 已保存返回内容到 {path}")
    except Exception as e:
        print(f"   [调试] 保存调试文件失败: {e}")


def fetch_long_text_pc(weibo_id, cookies):
    url = f"https://weibo.com/ajax/statuses/longtext?id={weibo_id}"
    headers = get_random_headers(f"https://weibo.com/detail/{weibo_id}", cookies)
    resp = request_with_retry(url, cookies, headers)
    if not resp or resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    if data.get("ok") != 1:
        return None
    content = (data.get("data") or {}).get("longTextContent")
    if not content:
        content = (data.get("data") or {}).get("content")
    return content


def fetch_long_text_m(weibo_id, cookies):
    url = f"https://m.weibo.cn/statuses/extend?id={weibo_id}"
    headers = get_random_headers(f"https://m.weibo.cn/detail/{weibo_id}", cookies)
    resp = request_with_retry(url, cookies, headers)
    if not resp or resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    if data.get("ok") != 1:
        return None
    content = (data.get("data") or {}).get("longTextContent")
    if not content:
        content = (data.get("data") or {}).get("content")
    return content


def maybe_expand_long_text(status, weibo_id, cookies, current_text):
    if not isinstance(status, dict):
        return current_text

    is_long = bool(status.get("isLongText") or status.get("is_long_text"))
    text_length = status.get("textLength") or status.get("text_length")
    try:
        text_length = int(text_length) if text_length is not None else None
    except Exception:
        text_length = None

    cleaned_current = clean_html(current_text) if current_text else ""
    looks_truncated = False
    if cleaned_current.endswith("..."):
        looks_truncated = True
    if text_length and len(cleaned_current) > 0 and len(cleaned_current) < text_length:
        looks_truncated = True

    if not (is_long or looks_truncated):
        return current_text

    long_html = fetch_long_text_pc(weibo_id, cookies) or fetch_long_text_m(weibo_id, cookies)
    if not long_html:
        return current_text

    long_text = clean_html(long_html)
    if not long_text:
        return current_text

    return long_text

def fetch_weibo_detail(weibo_id, cookies):
    pc_url = f"https://weibo.com/ajax/statuses/show?id={weibo_id}"
    headers = get_random_headers('https://weibo.com/', cookies)
    resp = request_with_retry(pc_url, cookies, headers)
    resp = request_with_retry(pc_url, cookies, headers)
    
    if resp and resp.status_code == 200:
        try:
            data = resp.json()
            if data.get('ok') == 1:
                status = data.get('data')
                if not isinstance(status, dict):
                    status = data
                created_at, user, text = extract_status_fields(status)
                text = maybe_expand_long_text(status, weibo_id, cookies, text)
                expected_comment_counts[weibo_id] = status.get('comments_count', 0) or 0
                
                # 清洗文本
                clean_text = clean_html(text)

                if not clean_text or not created_at or not user:
                    print("   [博文] PC接口字段不完整，尝试移动端")
                    print(
                        f"   [博文] PC字段: created_at={bool(created_at)}, user={bool(user)}, text_len={len(clean_text) if clean_text else 0}"
                    )
                    dump_debug_json(
                        "pc_incomplete",
                        weibo_id,
                        {
                            "url": pc_url,
                            "ok": data.get("ok"),
                            "keys": list(data.keys()) if isinstance(data, dict) else None,
                            "data_type": str(type(status)),
                            "status_keys": list(status.keys()) if isinstance(status, dict) else None,
                            "sample": status if isinstance(status, dict) else status,
                        },
                    )
                else:
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
                status = data.get('data')
                if not isinstance(status, dict):
                    status = data
                created_at, user, text = extract_status_fields(status)
                text = maybe_expand_long_text(status, weibo_id, cookies, text)
                expected_comment_counts[weibo_id] = status.get('comments_count', 0) or 0
                clean_text = clean_html(text)
                if not clean_text or not created_at or not user:
                    print("   [博文] 移动端字段不完整，跳过写入")
                    print(
                        f"   [博文] M字段: created_at={bool(created_at)}, user={bool(user)}, text_len={len(clean_text) if clean_text else 0}"
                    )
                    dump_debug_json(
                        "m_incomplete",
                        weibo_id,
                        {
                            "url": m_url,
                            "ok": data.get("ok"),
                            "keys": list(data.keys()) if isinstance(data, dict) else None,
                            "data_type": str(type(status)),
                            "status_keys": list(status.keys()) if isinstance(status, dict) else None,
                            "sample": status if isinstance(status, dict) else status,
                        },
                    )
                else:
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
        header = ['weibo_id', 'user_name', 'publish_time', 'text']
        if os.path.exists(WEIBO_TEXTS_CSV):
            ensure_csv_header(WEIBO_TEXTS_CSV, header)
        file_exists = os.path.isfile(WEIBO_TEXTS_CSV)
        with open(WEIBO_TEXTS_CSV, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([weibo_id, user, created_at, text])

def write_comments_csv(comments_list):
    """
    comments_list: [[weibo_id, user, created_at, text], ...]
    """
    if not comments_list:
        return
    with csv_lock:
        file_exists = os.path.isfile(WEIBO_COMMENTS_CSV)
        with open(WEIBO_COMMENTS_CSV, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['weibo_id', 'user_name', 'publish_time', 'text'])
            writer.writerows(comments_list)

def process_weibo_task(weibo_id, cookies, latest_ids, fetch_posts=True, fetch_comments_flag=True):
    """
    单个微博ID的处理任务
    """
    print(f"\n[线程] 开始处理微博 ID: {weibo_id}")
    human_sleep((1.0, 3.0), long_prob=0.1, long_range=(6, 12))
    if fetch_posts:
        ok = fetch_weibo_detail(weibo_id, cookies)
        if ok:
            wid_int = parse_weibo_id(weibo_id)
            if wid_int is not None and wid_int > int(latest_ids.get("latest_post_id", 0) or 0):
                latest_ids["latest_post_id"] = wid_int
                save_latest_ids(latest_ids)
        human_sleep((1.2, 2.8), long_prob=0.08, long_range=(5, 10))

    if fetch_comments_flag:
        fetch_comments(weibo_id, cookies)
        wid_int = parse_weibo_id(weibo_id)
        if wid_int is not None and wid_int > int(latest_ids.get("latest_comment_id", 0) or 0):
            latest_ids["latest_comment_id"] = wid_int
            save_latest_ids(latest_ids)

    human_sleep(TASK_DELAY_RANGE, long_prob=0.15, long_range=(10, 25))

def should_fetch_by_latest_id(weibo_id, latest_id_value):
    wid = parse_weibo_id(weibo_id)
    if wid is None:
        return False
    try:
        latest = int(latest_id_value or 0)
    except Exception:
        latest = 0
    return wid > latest

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    # 1. 加载 Cookie
    cookies = get_cookie_dict()
    if not cookies:
        return

    # 2. 读取 weibo_ids.txt
    try:
        with open(os.path.join(BASE_DIR, "weibo_ids.txt"), "r", encoding="utf-8") as f:
            # 读取所有行，去除空白字符
            all_weibo_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("weibo_ids.txt not found. Please run GetWeiboIdList.py first.")
        return

    latest_ids = load_latest_ids()

    parsed_ids = [(parse_weibo_id(w), w) for w in all_weibo_ids]
    parsed_ids = [(i, w) for i, w in parsed_ids if i is not None]
    parsed_ids.sort(key=lambda x: x[0])

    max_weibo_id = parsed_ids[-1][0] if parsed_ids else 0
    print(
        f"总共有 {len(all_weibo_ids)} 个微博ID，最大ID: {max_weibo_id}；"
        f"最新评论ID: {latest_ids.get('latest_comment_id', 0)}；最新帖子ID: {latest_ids.get('latest_post_id', 0)}"
    )

    post_ids_to_process = [
        w for _, w in parsed_ids if should_fetch_by_latest_id(w, latest_ids.get("latest_post_id", 0))
    ]
    comment_ids_to_process = [
        w for _, w in parsed_ids if should_fetch_by_latest_id(w, latest_ids.get("latest_comment_id", 0))
    ]

    fetch_posts_flag = True
    fetch_comments_flag = bool(comment_ids_to_process)

    print(
        f"待爬取帖子ID数量: {len(post_ids_to_process)} (输出: {WEIBO_TEXTS_CSV})；"
        f"待爬取评论ID数量: {len(comment_ids_to_process)} (输出: {WEIBO_COMMENTS_CSV})"
    )

    if not post_ids_to_process and not comment_ids_to_process:
        print("没有需要爬取的新内容。")
        return

    ids_union = []
    if fetch_posts_flag:
        ids_union.extend(post_ids_to_process)
    if fetch_comments_flag:
        ids_union.extend(comment_ids_to_process)
    ids_union = sorted(set(ids_union), key=lambda x: parse_weibo_id(x) or 0)

    for wid in ids_union:
        try:
            do_posts = wid in set(post_ids_to_process)
            do_comments = wid in set(comment_ids_to_process)
            process_weibo_task(
                wid,
                cookies,
                latest_ids,
                fetch_posts=do_posts,
                fetch_comments_flag=do_comments,
            )
        except Exception as e:
            print(f"[错误] 处理微博 {wid} 时发生异常: {e}")

if __name__ == "__main__":
    main()
