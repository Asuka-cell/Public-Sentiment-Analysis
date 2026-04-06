from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

def get_weibo_cookie():
    # Initialize the browser
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    try:
        # Open Weibo login page
        driver.get("https://weibo.com/login.php")
        
        # Wait for 25 seconds for manual login
        time.sleep(25)
        
        # Get cookies
        cookies = driver.get_cookies()
        
        # Save cookies to cookie.txt
        with open("cookies.txt", "w", encoding="utf-8") as f:
            json.dump(cookies, f)
            
    finally:
        driver.quit()

if __name__ == "__main__":
    get_weibo_cookie() # Run this first if you don't have cookies.txt