import requests
import json
import urllib.parse
import re
import time
import random
import os

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/123.0.0.0 Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-S9180) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

PROXY_LIST = [
]

REQUEST_DELAY_RANGE = (1.2, 2.5)
PAGE_DELAY_RANGE = (3.0, 6.0)
PROXY_COOLDOWN_SECONDS = 600
UA_COOLDOWN_SECONDS = 600
PROXY_RECHECK_SECONDS = 1800
PROXY_FAIL_COOLDOWN_SECONDS = 900

proxy_cooldown_until = {}
proxy_last_checked = {}
proxy_valid_cache = {}
proxy_fail_counts = {}
ua_cooldown_until = {}
ua_block_counts = {}


def human_sleep(base_range, long_prob=0.08, long_range=(8, 15)):
    t = random.uniform(*base_range)
    time.sleep(t)
    if random.random() < long_prob:
        time.sleep(random.uniform(*long_range))


def build_proxies(proxy_entry):
    if proxy_entry.startswith("http://") or proxy_entry.startswith("https://"):
        url = proxy_entry
    else:
        url = "http://" + proxy_entry
    return {"http": url, "https": url}


def probe_proxy(proxies):
    try:
        r = requests.get(
            "https://weibo.com/robots.txt",
            timeout=8,
            proxies=proxies,
            headers={
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/plain",
            },
        )
        return r.status_code < 400
    except Exception:
        return False


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
        if (
            now - proxy_last_checked.get(p, 0) > PROXY_RECHECK_SECONDS
            or proxy_valid_cache.get(p) is None
        ):
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
    for attempt in range(max_retries):
        proxies, proxy_label = get_proxy_dict()
        proxy_info = proxies["http"] if proxies else "Localhost"

        if attempt > 0:
            if proxy_info == "Localhost":
                backoff_time = min(20, (2**attempt) + random.uniform(1, 3))
            else:
                backoff_time = min(12, (1.5**attempt) + random.uniform(1, 2))
            time.sleep(backoff_time)

        try:
            ua_label = get_random_ua()
            headers["User-Agent"] = ua_label

            human_sleep(REQUEST_DELAY_RANGE, long_prob=0.06, long_range=(5, 10))
            response = requests.get(
                url,
                params=params,
                cookies=cookies,
                headers=headers,
                timeout=15,
                proxies=proxies,
            )

            if response.status_code == 200:
                if "passport.weibo.com" in response.url:
                    ua_block_counts[ua_label] = ua_block_counts.get(ua_label, 0) + 1
                    ua_cooldown_until[ua_label] = time.time() + UA_COOLDOWN_SECONDS
                    return None
                return response

            if response.status_code in {418, 403}:
                if proxy_label != "Localhost":
                    proxy_cooldown_until[proxy_label] = time.time() + PROXY_COOLDOWN_SECONDS
                ua_block_counts[ua_label] = ua_block_counts.get(ua_label, 0) + 1
                ua_cooldown_until[ua_label] = time.time() + UA_COOLDOWN_SECONDS
                continue
        except (
            requests.exceptions.ProxyError,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ):
            continue
        except Exception:
            break

    return None


def extract_weibo_ids_from_html(html):
    patterns = [
        r'mid="(\d{16,})"',
        r'"mid":"(\d{16,})"',
        r'weibo_id=(\d{16,})',
        r'mid=(\d{16,})',
    ]
    found = set()
    for p in patterns:
        try:
            found.update(re.findall(p, html))
        except Exception:
            continue
    return [x for x in found if x]


def load_existing_ids(file_path):
    if not file_path:
        return [], set()
    if not os.path.exists(file_path):
        return [], set()
    ids = []
    seen = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                v = line.strip()
                if v and v not in seen:
                    ids.append(v)
                    seen.add(v)
    except Exception:
        return [], set()
    return ids, seen


def append_ids(file_path, ids):
    if not file_path or not ids:
        return
    with open(file_path, "a", encoding="utf-8") as f:
        for wid in ids:
            f.write(str(wid) + "\n")


def get_pc_search_weibo_ids(keyword, max_pages=None, output_path=None):
    """
    使用 PC 版搜索 (s.weibo.com) 获取博文ID，支持翻页
    :param keyword: 搜索关键词
    :param max_pages: 最大抓取页数，默认不限制
    :param output_path: 实时写入的输出文件路径（可选）
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
    
    headers = {
        'Referer': 'https://s.weibo.com/',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    }

    all_weibo_ids, seen_ids = load_existing_ids(output_path)
    if output_path and all_weibo_ids:
        print(f"检测到已有 {len(all_weibo_ids)} 条ID，将继续追加写入: {output_path}")

    page = 1
    consecutive_empty_pages = 0
    consecutive_fail_pages = 0
    consecutive_block_pages = 0
    while True:
        if max_pages is not None and page > int(max_pages):
            break
        print(f"\n>>> 正在抓取第 {page} 页...")
        
        # 构造分页 URL
        # PC版微博搜索分页参数: &page=2
        current_url = f"{base_url}&page={page}"

        human_sleep(PAGE_DELAY_RANGE, long_prob=0.08, long_range=(8, 20))
        resp = request_with_retry(current_url, cookies, headers, max_retries=4)
        if not resp or resp.status_code != 200:
            consecutive_fail_pages += 1
            if consecutive_fail_pages >= 10:
                break
            page += 1
            continue

        consecutive_fail_pages = 0
        html = resp.text
        if "passport.weibo.com" in resp.url:
            return all_weibo_ids

        if "抱歉，未找到" in html:
            break

        blocked_markers = [
            "Sina Visitor System",
            "访问过于频繁",
            "请输入验证码",
            "verify",
        ]
        if any(m in html for m in blocked_markers):
            consecutive_block_pages += 1
            if consecutive_block_pages >= 5:
                break
            time.sleep(random.uniform(12, 25))
            continue
        consecutive_block_pages = 0

        found_ids = extract_weibo_ids_from_html(html)
        found_ids = sorted(set(found_ids), reverse=True)

        if not found_ids:
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= 10:
                break
        else:
            consecutive_empty_pages = 0
            new_ids = []
            for mid in found_ids:
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    all_weibo_ids.append(mid)
                    new_ids.append(mid)
            append_ids(output_path, new_ids)

        page += 1

    return all_weibo_ids

if __name__ == "__main__":
    keyword = input("请输入搜索关键词 (例如 '西贝预制菜'): ")
    output_path = "/Users/asuka/项目/publicSentimentAnalysis/Public-Sentiment-Analysis/weibo_ids.txt"
    ids = get_pc_search_weibo_ids(keyword, max_pages=None, output_path=output_path)
    print(f"\n共获取到 {len(ids)} 个有效 ID (已去重)")
    print(f"ID 已保存到 {output_path}")
