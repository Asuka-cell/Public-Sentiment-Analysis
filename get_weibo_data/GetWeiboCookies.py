import json
import os
import traceback

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

        while True:
            input("请在弹出的浏览器中完成登录/扫码后，回到终端按回车继续... ")
            try:
                driver.get("https://weibo.com/")
            except Exception:
                pass

            cookies = driver.get_cookies() or []
            print(f"当前URL: {getattr(driver, 'current_url', '')}")
            print(f"获取到 cookies 数量: {len(cookies)}")

            if cookies:
                break
            print("cookies 为空，说明很可能还没登录成功，或页面未完成跳转。请回到浏览器确认已登录后再按回车。")

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
    get_weibo_cookie()
