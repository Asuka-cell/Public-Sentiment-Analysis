import argparse
import os

import requests

from zhihu_common import load_cookie_jar, search_targets, search_targets_selenium


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--cookies", default=os.path.join(base_dir, "zhihu_cookies.json"))
    parser.add_argument("--targets", default=os.path.join(base_dir, "zhihu_targets.txt"))
    parser.add_argument("--keyword", default=None)
    parser.add_argument("--search_limit", type=int, default=20)
    parser.add_argument("--search_pages", type=int, default=3)
    parser.add_argument("--search_types", default="question,answer")
    args = parser.parse_args()

    cookies = load_cookie_jar(args.cookies)
    has_profile = bool(os.environ.get("CHROME_USER_DATA_DIR"))
    if not cookies and not has_profile:
        print(f"未加载到 cookies：{args.cookies}")
        print("请先运行：/usr/bin/python3 get_zhihu_data/GetZhihuCookies.py")
        print("或设置 CHROME_USER_DATA_DIR / CHROME_PROFILE_DIR 使用浏览器登录态。")
        return

    kw = (args.keyword or "").strip()
    if not kw:
        try:
            kw = input("请输入知乎搜索关键词: ").strip()
        except Exception:
            kw = ""
    if not kw:
        print("关键词为空，结束。")
        return

    session = requests.Session()
    if cookies:
        session.cookies.update(cookies)

    types = [t.strip().lower() for t in str(args.search_types).split(",") if t.strip()]
    found = []
    if cookies:
        found = search_targets(
            session=session,
            keyword=kw,
            limit=args.search_limit,
            max_pages=args.search_pages,
            allow_types=types,
        )
    if not found:
        found = search_targets_selenium(
            keyword=kw,
            cookies_path=args.cookies,
            max_pages=args.search_pages,
            allow_types=types,
        )

    existing = set()
    if os.path.exists(args.targets):
        with open(args.targets, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u:
                    existing.add(u)

    new_urls = []
    for t in found:
        u = t.get("url")
        if not u:
            continue
        if u in existing:
            continue
        existing.add(u)
        new_urls.append(u)

    os.makedirs(os.path.dirname(args.targets), exist_ok=True)
    with open(args.targets, "a", encoding="utf-8") as _:
        pass

    if new_urls:
        with open(args.targets, "a", encoding="utf-8") as f:
            for u in new_urls:
                f.write(u + "\n")

    print(f"搜索关键词: {kw}")
    print(f"新增目标数: {len(new_urls)}")
    print(f"目标文件: {args.targets}")


if __name__ == "__main__":
    main()
