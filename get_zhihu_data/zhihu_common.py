import csv
import json
import os
import random
import re
import time
from datetime import datetime
from urllib.parse import quote
from urllib.parse import urlparse

import requests

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path, default):
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_cookie_jar(cookies_path):
    if not cookies_path or not os.path.exists(cookies_path):
        return {}
    try:
        with open(cookies_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        out = {}
        for it in data:
            if not isinstance(it, dict):
                continue
            name = it.get("name")
            value = it.get("value")
            if name and value is not None:
                out[str(name)] = str(value)
        return out
    return {}


def build_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")

    user_data_dir = os.environ.get("CHROME_USER_DATA_DIR")
    profile_dir = os.environ.get("CHROME_PROFILE_DIR")
    if user_data_dir:
        options.add_argument(f"--user-data-dir={user_data_dir}")
    if profile_dir:
        options.add_argument(f"--profile-directory={profile_dir}")

    driver_path = os.environ.get("CHROMEDRIVER_PATH")
    if driver_path:
        return webdriver.Chrome(service=Service(executable_path=driver_path), options=options)
    return webdriver.Chrome(options=options)


def human_sleep(base_range=(0.8, 1.6), long_prob=0.05, long_range=(6, 12)):
    time.sleep(random.uniform(*base_range))
    if random.random() < long_prob:
        time.sleep(random.uniform(*long_range))


def request_json(session, url, params=None, max_retries=4):
    last_err = None
    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Referer": "https://www.zhihu.com/",
                "Origin": "https://www.zhihu.com",
                "Connection": "keep-alive",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Dest": "empty",
            }
            human_sleep()
            resp = session.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in {400, 401, 403}:
                try:
                    j = resp.json()
                except Exception:
                    j = None
                if isinstance(j, dict):
                    err = j.get("error") or {}
                    code = err.get("code")
                    msg = str(err.get("message") or "")
                    if str(code) in {"10003", "40362"} or ("请求参数异常" in msg) or ("账号当前请求存在异常" in msg):
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code in {401, 403, 429}:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(min(20, 2**attempt + random.uniform(0.5, 1.5)))
                continue
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(min(12, 1.5**attempt + random.uniform(0.5, 1.5)))
        except Exception as e:
            last_err = e
            time.sleep(min(12, 1.5**attempt + random.uniform(0.5, 1.5)))
    raise last_err or RuntimeError("request failed")


def safe_text(v):
    if v is None:
        return ""
    return str(v).replace("\r", " ").replace("\n", " ").strip()


def parse_zhihu_target(url):
    u = (url or "").strip()
    if not u:
        return None
    if not u.startswith("http"):
        return None

    parsed = urlparse(u)
    host = parsed.netloc.lower()
    if "zhihu.com" not in host:
        return None

    m = re.search(r"/question/(\d+)", parsed.path)
    if m:
        qid = m.group(1)
        m2 = re.search(r"/answer/(\d+)", parsed.path)
        if m2:
            return {"type": "answer", "question_id": qid, "answer_id": m2.group(1), "url": u}
        return {"type": "question", "question_id": qid, "url": u}

    m = re.search(r"/p/(\d+)", parsed.path)
    if m:
        return {"type": "article", "article_id": m.group(1), "url": u}

    return None


def load_targets(path, single_url=None):
    targets = []
    if single_url:
        t = parse_zhihu_target(single_url)
        if t:
            targets.append(t)
        return targets
    if not path or not os.path.exists(path):
        return targets
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u:
                continue
            t = parse_zhihu_target(u)
            if t:
                targets.append(t)
    return targets


def ensure_csv(path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def iter_existing_ids(csv_path, id_field):
    if not os.path.exists(csv_path):
        return set()
    seen = set()
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                v = row.get(id_field)
                if v:
                    seen.add(str(v))
    except Exception:
        return set()
    return seen


def append_rows(csv_path, fieldnames, rows):
    if not rows:
        return
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            w.writerow(row)


def get_zhihu_cookies_interactive(cookies_path, non_interactive: bool = False, wait_seconds: int = 600):
    driver = None
    try:
        driver = build_driver()
        driver.get("https://www.zhihu.com/signin")
        print("提示：请在弹出的浏览器中完成登录（可扫码/手机验证码等），登录成功后不要关闭浏览器。")

        start = time.time()
        while True:
            if not non_interactive:
                input("登录完成后回到终端按回车继续... ")
            else:
                time.sleep(2.0)
            try:
                driver.get("https://www.zhihu.com/")
            except Exception:
                pass

            cookies = driver.get_cookies() or []
            has_z_c0 = any(str(c.get("name")) == "z_c0" for c in cookies if isinstance(c, dict))
            print(f"当前URL: {getattr(driver, 'current_url', '')}")
            print(f"获取到 cookies 数量: {len(cookies)}")
            if cookies and has_z_c0:
                break
            print("cookies 未包含 z_c0，通常说明尚未登录成功。请确认已登录后再按回车。")
            if non_interactive and (time.time() - start) >= float(wait_seconds):
                print(f"等待超时（{wait_seconds}s），未检测到有效 cookies。")
                return

        os.makedirs(os.path.dirname(cookies_path), exist_ok=True)
        with open(cookies_path, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        print(f"cookies 已保存: {cookies_path}")
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


def search_targets(session, keyword, limit=20, max_pages=3, allow_types=None):
    kw = (keyword or "").strip()
    if not kw:
        return []
    allow = set(allow_types or [])

    def _add_url(out_list, seen_set, url_str):
        t = parse_zhihu_target(url_str)
        if not t:
            return
        obj_type = str(t.get("type") or "").lower()
        if allow and obj_type not in allow:
            return
        k = t.get("url") or url_str
        if k in seen_set:
            return
        seen_set.add(k)
        out_list.append(t)

    out = []
    seen = set()

    offset = 0
    page = 0
    api_ok = False
    while True:
        url = "https://www.zhihu.com/api/v4/search_v3"
        params = {
            "t": "general",
            "q": kw,
            "correction": 1,
            "offset": int(offset),
            "limit": int(limit),
            "lc_idx": 0,
            "show_all_topics": 0,
        }
        try:
            data = request_json(session, url, params=params)
        except Exception:
            break

        items = data.get("data") or []
        if not items:
            break
        api_ok = True
        for it in items:
            obj = it.get("object") or {}
            obj_type = str(obj.get("type") or it.get("type") or "").lower()

            url_val = obj.get("url") or ""
            u = ""
            if isinstance(url_val, str) and url_val.startswith("http"):
                u = url_val
            elif obj_type == "question":
                qid = obj.get("id")
                if qid:
                    u = f"https://www.zhihu.com/question/{qid}"
            elif obj_type == "answer":
                aid = obj.get("id")
                qid = (obj.get("question") or {}).get("id")
                if qid and aid:
                    u = f"https://www.zhihu.com/question/{qid}/answer/{aid}"
                elif obj.get("url") and isinstance(obj.get("url"), str):
                    u = obj.get("url")
            elif obj_type == "article":
                aid = obj.get("id")
                if aid:
                    u = f"https://zhuanlan.zhihu.com/p/{aid}"
            if u:
                _add_url(out, seen, u)

        paging = data.get("paging") or {}
        if paging.get("is_end") is True:
            break
        offset = int(paging.get("next_offset", offset + limit))
        page += 1
        if max_pages is not None and page >= int(max_pages):
            break

    if api_ok and out:
        return out

    page = 1
    max_pages_i = int(max_pages) if max_pages is not None else 3
    per_page = max(10, int(limit))
    while page <= max_pages_i:
        try:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Referer": "https://www.zhihu.com/",
            }
            params = {"type": "content", "q": kw, "p": int(page)}
            human_sleep()
            resp = session.get("https://www.zhihu.com/search", params=params, headers=headers, timeout=20)
            if resp.status_code != 200:
                break
            html = resp.text or ""
        except Exception:
            break

        for m in re.finditer(r"https?://www\.zhihu\.com/question/\d+(?:/answer/\d+)?", html):
            _add_url(out, seen, m.group(0))
        for m in re.finditer(r"https?://zhuanlan\.zhihu\.com/p/\d+", html):
            _add_url(out, seen, m.group(0))

        if len(out) >= per_page * page:
            pass
        page += 1

    return out


def search_targets_selenium(keyword, cookies_path, max_pages=3, allow_types=None):
    kw = (keyword or "").strip()
    if not kw:
        return []

    allow = set(allow_types or [])
    out = []
    seen = set()

    try:
        from selenium.webdriver.common.by import By
    except Exception:
        return []

    cookies_list = []
    if cookies_path and os.path.exists(str(cookies_path)):
        cookies_raw = load_json(cookies_path, default=[])
        if isinstance(cookies_raw, list):
            cookies_list = [c for c in cookies_raw if isinstance(c, dict)]
        elif isinstance(cookies_raw, dict):
            for k, v in cookies_raw.items():
                cookies_list.append({"name": str(k), "value": str(v), "domain": ".zhihu.com", "path": "/"})

    driver = None
    try:
        driver = build_driver()
        driver.get("https://www.zhihu.com/")
        time.sleep(1.0)

        if cookies_list:
            for c in cookies_list:
                name = c.get("name")
                value = c.get("value")
                if not name or value is None:
                    continue
                ck = {
                    "name": str(name),
                    "value": str(value),
                    "domain": str(c.get("domain") or ".zhihu.com"),
                    "path": str(c.get("path") or "/"),
                }
                if c.get("expiry") is not None:
                    try:
                        ck["expiry"] = int(c.get("expiry"))
                    except Exception:
                        pass
                try:
                    driver.add_cookie(ck)
                except Exception:
                    continue

        for p in range(1, int(max_pages) + 1):
            q = quote(kw)
            url = f"https://www.zhihu.com/search?type=content&q={q}&p={p}"
            driver.get(url)
            time.sleep(1.2)
            for _ in range(3):
                try:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                except Exception:
                    pass
                time.sleep(1.0)

            links = []
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, "a[href]")
                for e in elems:
                    try:
                        href = e.get_attribute("href") or ""
                    except Exception:
                        href = ""
                    if not href:
                        continue
                    if "zhihu.com/question/" in href or "zhuanlan.zhihu.com/p/" in href:
                        links.append(href)
            except Exception:
                links = []

            for href in links:
                t = parse_zhihu_target(href)
                if not t:
                    continue
                obj_type = str(t.get("type") or "").lower()
                if allow and obj_type not in allow:
                    continue
                k = t.get("url") or href
                if k in seen:
                    continue
                seen.add(k)
                out.append(t)

            if len(out) >= 200:
                break
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    return out


def fetch_question(session, question_id):
    url = f"https://www.zhihu.com/api/v4/questions/{question_id}"
    params = {
        "include": "id,title,excerpt,created,updated_time,comment_count,answer_count,follower_count",
    }
    data = request_json(session, url, params=params)
    return {
        "question_id": str(data.get("id", question_id)),
        "title": safe_text(data.get("title")),
        "excerpt": safe_text(data.get("excerpt")),
        "created": str(data.get("created", "")),
        "updated_time": str(data.get("updated_time", "")),
        "answer_count": str(data.get("answer_count", "")),
        "comment_count": str(data.get("comment_count", "")),
        "follower_count": str(data.get("follower_count", "")),
        "url": f"https://www.zhihu.com/question/{question_id}",
    }


def fetch_answers(session, question_id, limit=20, max_pages=None):
    url = f"https://www.zhihu.com/api/v4/questions/{question_id}/answers"
    offset = 0
    page = 0
    while True:
        params = {
            "include": "id,created_time,updated_time,content,voteup_count,comment_count,url,author.name,author.id",
            "limit": int(limit),
            "offset": int(offset),
            "sort_by": "default",
        }
        data = request_json(session, url, params=params)
        items = data.get("data") or []
        if not items:
            break
        for it in items:
            yield it
        paging = data.get("paging") or {}
        is_end = bool(paging.get("is_end"))
        if is_end:
            break
        offset = int(paging.get("next_offset", offset + limit))
        page += 1
        if max_pages is not None and page >= int(max_pages):
            break


def fetch_answer(session, answer_id):
    url = f"https://www.zhihu.com/api/v4/answers/{answer_id}"
    params = {
        "include": "id,created_time,updated_time,content,voteup_count,comment_count,url,question.id,author.name,author.id",
    }
    return request_json(session, url, params=params)


def fetch_answer_comments(session, answer_id, limit=20, max_pages=None):
    url = f"https://www.zhihu.com/api/v4/answers/{answer_id}/comments"
    offset = 0
    page = 0
    while True:
        params = {
            "limit": int(limit),
            "offset": int(offset),
            "order_by": "score",
            "status": "open",
        }
        data = request_json(session, url, params=params)
        items = data.get("data") or []
        if not items:
            break
        for it in items:
            yield it
        paging = data.get("paging") or {}
        is_end = bool(paging.get("is_end"))
        if is_end:
            break
        next_url = paging.get("next") or ""
        m = re.search(r"[?&]offset=(\d+)", next_url)
        if m:
            offset = int(m.group(1))
        else:
            offset += int(limit)
        page += 1
        if max_pages is not None and page >= int(max_pages):
            break


def fetch_article(session, article_id):
    url = f"https://www.zhihu.com/api/v4/articles/{article_id}"
    params = {
        "include": "id,title,excerpt,created,updated,content,voteup_count,comment_count,url,author.name,author.id",
    }
    return request_json(session, url, params=params)


def fetch_article_comments(session, article_id, limit=20, max_pages=None):
    url = f"https://www.zhihu.com/api/v4/articles/{article_id}/comments"
    offset = 0
    page = 0
    while True:
        params = {
            "limit": int(limit),
            "offset": int(offset),
            "order_by": "score",
            "status": "open",
        }
        data = request_json(session, url, params=params)
        items = data.get("data") or []
        if not items:
            break
        for it in items:
            yield it
        paging = data.get("paging") or {}
        if bool(paging.get("is_end")):
            break
        next_url = paging.get("next") or ""
        m = re.search(r"[?&]offset=(\d+)", next_url)
        if m:
            offset = int(m.group(1))
        else:
            offset += int(limit)
        page += 1
        if max_pages is not None and page >= int(max_pages):
            break
