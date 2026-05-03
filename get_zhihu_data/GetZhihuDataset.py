import argparse
import json
import os
import re
import time
from datetime import datetime

import requests

from zhihu_common import (
    append_rows,
    build_driver,
    ensure_csv,
    fetch_answers,
    fetch_question,
    iter_existing_ids,
    load_cookie_jar,
    load_json,
    load_targets,
    now_ts,
    safe_text,
    save_json,
)


def _to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _ts_to_dt_str(ts):
    try:
        x = int(ts)
    except Exception:
        return ""
    if x <= 0:
        return ""
    try:
        return datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _read_csv_rows(path):
    try:
        import csv

        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            return [row for row in r]
    except Exception:
        return []


def _write_csv_rows(path, fieldnames, rows):
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: (row.get(k) if row.get(k) is not None else "") for k in fieldnames})


def _pick_top_k_answers(answers, k):
    items = []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        aid = str(ans.get("id", "")).strip()
        if not aid:
            continue
        vote = _to_int(ans.get("voteup_count", 0), default=0)
        items.append((vote, aid, ans))
    items.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in items[: int(k)]]


def _zh_vote_to_int(text):
    s = str(text or "").strip()
    if not s:
        return 0
    s = s.replace("赞同", "").replace("赞", "").replace(" ", "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)万", s)
    if m:
        try:
            return int(float(m.group(1)) * 10000)
        except Exception:
            return 0
    m = re.search(r"(\d+(?:\.\d+)?)千", s)
    if m:
        try:
            return int(float(m.group(1)) * 1000)
        except Exception:
            return 0
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _extract_answer_id_from_href(href):
    h = str(href or "")
    m = re.search(r"/answer/(\d+)", h)
    if m:
        return m.group(1)
    return ""


def _init_driver_with_cookies(cookies_path):
    cookies_list = []
    if cookies_path and os.path.exists(str(cookies_path)):
        cookies_raw = load_json(cookies_path, default=[])
        if isinstance(cookies_raw, list):
            cookies_list = [c for c in cookies_raw if isinstance(c, dict)]
        elif isinstance(cookies_raw, dict):
            for k, v in cookies_raw.items():
                cookies_list.append({"name": str(k), "value": str(v), "domain": ".zhihu.com", "path": "/"})

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
    return driver


def _extract_initial_data(html):
    s = str(html or "")
    m = re.search(r'<script[^>]+id="js-initialData"[^>]*>(.*?)</script>', s, flags=re.S)
    if not m:
        m = re.search(r'<script[^>]+id="js-initialState"[^>]*>(.*?)</script>', s, flags=re.S)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _extract_question_from_initial_data(initial_data, qid):
    if not isinstance(initial_data, dict):
        return {}
    state = initial_data.get("initialState") or initial_data.get("initialState") or {}
    entities = (state.get("entities") or {}) if isinstance(state, dict) else {}
    questions = (entities.get("questions") or {}) if isinstance(entities, dict) else {}
    q = None
    if isinstance(questions, dict):
        q = questions.get(str(qid)) or questions.get(qid)
    if not isinstance(q, dict):
        return {}

    created_ts = q.get("created") or q.get("created_time") or q.get("createdTime")
    answer_count = q.get("answerCount") or q.get("answer_count")
    comment_count = q.get("commentCount") or q.get("comment_count")
    follower_count = q.get("followerCount") or q.get("follower_count")

    return {
        "title": str(q.get("title") or "").strip(),
        "excerpt": str(q.get("excerpt") or q.get("detail") or "").strip(),
        "publish_time": _ts_to_dt_str(created_ts),
        "answer_count": str(answer_count or "").strip(),
        "comment_count": str(comment_count or "").strip(),
        "follower_count": str(follower_count or "").strip(),
    }


def _fetch_question_and_top_answers_selenium(driver, qid, top_k):
    try:
        from selenium.webdriver.common.by import By
    except Exception as e:
        raise RuntimeError(f"缺少 selenium 依赖: {e}")

    qurl = f"https://www.zhihu.com/question/{qid}"
    driver.get(qurl)
    time.sleep(1.2)

    title = ""
    excerpt = ""
    try:
        els = driver.find_elements(By.CSS_SELECTOR, "h1.QuestionHeader-title")
        if els:
            title = (els[0].text or "").strip()
        if not title:
            els = driver.find_elements(By.CSS_SELECTOR, "h1")
            if els:
                title = (els[0].text or "").strip()
    except Exception:
        title = ""

    try:
        els = driver.find_elements(By.CSS_SELECTOR, ".QuestionHeader-detail")
        if els:
            excerpt = (els[0].text or "").strip()
    except Exception:
        excerpt = ""

    html = ""
    try:
        html = driver.page_source or ""
    except Exception:
        html = ""

    if ("账号当前请求存在异常" in html) or ("40362" in html) or ("请求参数异常" in html) or ("10003" in html):
        raise RuntimeError("页面被知乎风控拦截（40362/10003）。请使用浏览器真实 Profile 登录态，手动通过验证后再重试。")

    initial_data = _extract_initial_data(html)
    q_info = _extract_question_from_initial_data(initial_data, qid=qid)
    if (not title) and q_info.get("title"):
        title = q_info.get("title", "")
    if (not excerpt) and q_info.get("excerpt"):
        excerpt = q_info.get("excerpt", "")

    qrow = {
        "question_id": str(qid),
        "title": title,
        "excerpt": excerpt,
        "publish_time": q_info.get("publish_time", ""),
        "answer_count": q_info.get("answer_count", ""),
        "comment_count": q_info.get("comment_count", ""),
        "follower_count": q_info.get("follower_count", ""),
    }

    collected = {}
    for _ in range(10):
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        except Exception:
            pass
        time.sleep(1.0)
        try:
            items = driver.find_elements(By.CSS_SELECTOR, "div.AnswerItem")
        except Exception:
            items = []

        for el in items:
            aid = ""
            try:
                zop = el.get_attribute("data-zop")
                if zop:
                    data = json.loads(zop)
                    aid = str(data.get("itemId") or data.get("answer_id") or data.get("id") or "")
            except Exception:
                aid = ""

            if not aid:
                try:
                    links = el.find_elements(By.CSS_SELECTOR, "a[href*='/answer/']")
                except Exception:
                    links = []
                for lk in links:
                    try:
                        href = lk.get_attribute("href") or ""
                    except Exception:
                        href = ""
                    aid = _extract_answer_id_from_href(href)
                    if aid:
                        break

            if not aid:
                continue

            if aid in collected:
                continue

            author_name = ""
            author_id = ""
            try:
                a = el.find_elements(By.CSS_SELECTOR, ".AuthorInfo-name a")
                if a:
                    author_name = (a[0].text or "").strip()
                    href = a[0].get_attribute("href") or ""
                    m = re.search(r"/people/([^/?#]+)", href)
                    if m:
                        author_id = m.group(1)
            except Exception:
                pass

            vote_text = ""
            vote_count = 0
            for sel in ["button.VoteButton--up", "button[aria-label*='赞同']"]:
                try:
                    btns = el.find_elements(By.CSS_SELECTOR, sel)
                except Exception:
                    btns = []
                if btns:
                    vote_text = (btns[0].text or btns[0].get_attribute("aria-label") or "").strip()
                    vote_count = _zh_vote_to_int(vote_text)
                    break

            content = ""
            try:
                c = el.find_elements(By.CSS_SELECTOR, ".RichContent-inner")
                if c:
                    content = (c[0].text or "").strip()
            except Exception:
                content = ""

            collected[aid] = {
                "answer_id": aid,
                "question_id": str(qid),
                "author_id": safe_text(author_id),
                "author_name": safe_text(author_name),
                "created_time": "",
                "updated_time": "",
                "voteup_count": str(vote_count),
                "comment_count": "",
                "url": f"https://www.zhihu.com/question/{qid}/answer/{aid}",
                "content": safe_text(content),
            }

        if len(collected) >= int(top_k):
            break

    top = sorted(collected.values(), key=lambda r: _zh_vote_to_int(r.get("voteup_count")), reverse=True)[: int(top_k)]
    return qrow, top


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    dataset_dir = os.path.join(project_dir, "dataset")
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=os.path.join(base_dir, "zhihu_targets.txt"))
    parser.add_argument("--url", default=None)
    parser.add_argument("--cookies", default=os.path.join(base_dir, "zhihu_cookies.json"))
    parser.add_argument("--output_dir", default=dataset_dir)
    parser.add_argument("--answers_limit", type=int, default=20)
    parser.add_argument("--answers_pages", type=int, default=5)
    parser.add_argument("--top_answers", type=int, default=10)
    parser.add_argument("--prefer_selenium", action="store_true")
    parser.add_argument("--refetch_missing", action="store_true")
    args = parser.parse_args()

    cookies = load_cookie_jar(args.cookies)
    if not cookies:
        print(f"未加载到 cookies：{args.cookies}")
        print("请先运行：/usr/bin/python3 get_zhihu_data/GetZhihuCookies.py")
        return

    session = requests.Session()
    session.cookies.update(cookies)

    targets = load_targets(args.targets, args.url)
    if not targets:
        print("未找到可用的知乎目标 URL。你可以：")
        print(f"1) 把链接写入 {args.targets}（每行一个）")
        print("2) 或用 --url 传入单个链接")
        return

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    questions_csv = os.path.join(out_dir, "zhihu_questions.csv")
    answers_csv = os.path.join(out_dir, "zhihu_answers.csv")
    progress_path = os.path.join(base_dir, "zhihu_progress.json")

    question_fields = [
        "question_id",
        "title",
        "excerpt",
        "publish_time",
        "answer_count",
        "comment_count",
        "follower_count",
    ]
    answer_fields = [
        "answer_id",
        "question_id",
        "author_id",
        "author_name",
        "created_time",
        "updated_time",
        "voteup_count",
        "comment_count",
        "url",
        "content",
        "fetched_at",
    ]

    def _ensure_csv_force(path, fieldnames):
        if not os.path.exists(path):
            ensure_csv(path, fieldnames)
            return
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                head = (f.readline() or "").strip("\n").strip("\r")
        except Exception:
            ensure_csv(path, fieldnames)
            return
        existing = [x.strip() for x in head.split(",")] if head else []
        if existing == list(fieldnames):
            return
        rows = []
        try:
            import csv

            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    out = {k: (row.get(k) or "") for k in fieldnames}
                    rows.append(out)
        except Exception:
            rows = []
        try:
            import csv

            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in rows:
                    w.writerow(row)
        except Exception:
            ensure_csv(path, fieldnames)

    _ensure_csv_force(questions_csv, question_fields)
    _ensure_csv_force(answers_csv, answer_fields)

    seen_questions = iter_existing_ids(questions_csv, "question_id")
    seen_answers = iter_existing_ids(answers_csv, "answer_id")

    existing_question_rows = _read_csv_rows(questions_csv)
    question_by_id = {}
    question_order = []
    for row in existing_question_rows:
        qid = str(row.get("question_id") or "").strip()
        if not qid:
            continue
        normalized = {k: (row.get(k) or "") for k in question_fields}
        question_by_id[qid] = normalized
        question_order.append(qid)

    progress = load_json(progress_path, default={})
    done_keys = set(progress.get("done_targets", []))
    driver = None

    try:
        for t in targets:
            qid = None
            if t.get("type") == "question":
                qid = t.get("question_id")
            elif t.get("type") == "answer":
                qid = t.get("question_id")
            else:
                continue

            if not qid:
                continue

            tkey = f"question:{qid}"
            existing = question_by_id.get(str(qid), {})
            has_title = bool(str(existing.get("title") or "").strip())
            has_time = bool(str(existing.get("publish_time") or "").strip())
            need_qmeta = not (has_title and has_time)
            if tkey in done_keys and not need_qmeta:
                continue

            use_selenium = bool(args.prefer_selenium)
            qrow = None
            top_rows = None

            if not use_selenium:
                try:
                    if need_qmeta:
                        qrow = fetch_question(session, qid)
                    answers = list(
                        fetch_answers(
                            session,
                            qid,
                            limit=args.answers_limit,
                            max_pages=args.answers_pages,
                        )
                    )
                    top_answers = _pick_top_k_answers(answers, k=args.top_answers)
                    top_rows = []
                    for ans in top_answers:
                        aid = str(ans.get("id", "")).strip()
                        if not aid:
                            continue
                        author = ans.get("author") or {}
                        top_rows.append(
                            {
                                "answer_id": aid,
                                "question_id": str(qid),
                                "author_id": safe_text(author.get("id")),
                                "author_name": safe_text(author.get("name")),
                                "created_time": safe_text(ans.get("created_time")),
                                "updated_time": safe_text(ans.get("updated_time")),
                                "voteup_count": safe_text(ans.get("voteup_count")),
                                "comment_count": safe_text(ans.get("comment_count")),
                                "url": safe_text(ans.get("url")),
                                "content": safe_text(ans.get("content")),
                            }
                        )
                except Exception as e:
                    msg = str(e)
                    if "HTTP 403" in msg or "code\":10003" in msg or "请求参数异常" in msg:
                        use_selenium = True
                    else:
                        print(f"抓取失败: {qid} -> {e}")
                        continue

            if use_selenium:
                if driver is None:
                    driver = _init_driver_with_cookies(args.cookies)
                try:
                    qrow, top_rows = _fetch_question_and_top_answers_selenium(
                        driver=driver,
                        qid=qid,
                        top_k=args.top_answers,
                    )
                except Exception as e:
                    print(f"抓取问题失败: {qid} -> {e}")
                    continue

            if qrow is not None:
                qid_str = str(qrow.get("question_id") or qid).strip()
                normalized = {k: (qrow.get(k) or "") for k in question_fields}
                if qid_str not in question_by_id:
                    question_order.append(qid_str)
                question_by_id[qid_str] = normalized
                seen_questions.add(qid_str)

            if top_rows:
                to_write = []
                for arow in top_rows:
                    aid = str(arow.get("answer_id", "")).strip()
                    if not aid or aid in seen_answers:
                        continue
                    arow["fetched_at"] = now_ts()
                    to_write.append(arow)
                    seen_answers.add(aid)
                if to_write:
                    append_rows(answers_csv, answer_fields, to_write)

            done_keys.add(tkey)
            progress["done_targets"] = sorted(done_keys)
            progress["updated_at"] = now_ts()
            save_json(progress_path, progress)
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    out_rows = []
    for qid in question_order:
        row = question_by_id.get(str(qid))
        if row:
            out_rows.append(row)
    for qid, row in question_by_id.items():
        if qid not in set(question_order):
            out_rows.append(row)
    _write_csv_rows(questions_csv, question_fields, out_rows)

    print("抓取完成")
    print(f"问题输出: {questions_csv}")
    print(f"回答输出: {answers_csv}")
    print(f"进度文件: {progress_path}")


if __name__ == "__main__":
    main()
