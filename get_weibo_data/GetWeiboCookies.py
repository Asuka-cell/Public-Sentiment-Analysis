import json
import os
import traceback
import argparse
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


def build_driver():
    options = Options()
    options.add_argument("--start-maximized")

    driver_path = os.environ.get("CHROMEDRIVER_PATH")
    if driver_path:
        return webdriver.Chrome(service=Service(executable_path=driver_path), options=options)

    return webdriver.Chrome(options=options)


def _has_login_cookie(cookies) -> bool:
    try:
        for c in (cookies or []):
            if not isinstance(c, dict):
                continue
            if str(c.get("name") or "") == "SUB":
                v = str(c.get("value") or "")
                return len(v) >= 10
    except Exception:
        return False
    return False


def _is_login_page(url: str) -> bool:
    u = str(url or "").lower()
    return ("newlogin" in u) or ("passport.weibo.com" in u) or ("login" in u)


def get_weibo_cookie():
    driver = None
    try:
        print("正在启动 Chrome...")
        driver = build_driver()
        print("Chrome 已启动，正在打开微博登录页...")
        driver.get("https://weibo.com/login.php")
        print("提示：请不要提前关闭浏览器窗口，登录完成后保持页面打开。")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        cookie_path = os.path.join(base_dir, "cookies.txt")

        non_interactive = bool(os.environ.get("NON_INTERACTIVE"))
        wait_seconds = int(os.environ.get("WAIT_SECONDS") or "600")
        start = time.time()

        last_print = 0.0
        cookies = []
        while True:
            if not non_interactive:
                input("请在弹出的浏览器中完成登录/扫码后，回到终端按回车继续... ")
            else:
                time.sleep(2.0)

            try:
                driver.get("https://weibo.com/")
            except Exception:
                pass

            url = getattr(driver, "current_url", "")
            cookies = driver.get_cookies() or []

            now = time.time()
            if now - last_print >= 2.0:
                last_print = now
                print(f"当前URL: {url}")
                print(f"获取到 cookies 数量: {len(cookies)}")

            if _has_login_cookie(cookies) and (not _is_login_page(url)):
                break

            if non_interactive and (time.time() - start) >= float(wait_seconds):
                print(f"等待超时（{wait_seconds}s），未检测到登录态 cookies。")
                return

            if cookies and not _has_login_cookie(cookies):
                print("已获取到访客 cookies，但尚未检测到登录态（SUB）。请继续在浏览器内扫码/完成登录。")

        with open(cookie_path, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        print("cookies.txt 已保存")
    except Exception:
        traceback.print_exc()
        print("如果没有弹出浏览器：请确认已安装 Google Chrome；或设置 CHROMEDRIVER_PATH 指向 chromedriver 可执行文件")
    finally:
        if driver is not None:
            driver.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--non_interactive", action="store_true")
    parser.add_argument("--wait_seconds", type=int, default=600)
    args = parser.parse_args()
    if args.non_interactive:
        os.environ["NON_INTERACTIVE"] = "1"
        os.environ["WAIT_SECONDS"] = str(int(args.wait_seconds))
    get_weibo_cookie()
