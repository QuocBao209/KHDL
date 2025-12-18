from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

def sentiment_analysis(text):
    if not text:
        return "không rõ"
    positive_keywords = ["tốt", "ngon", "đẹp", "ok", "hài lòng", "ổn", "chất lượng", "ăn được", "hợp lý", "tiện lợi", "ủng hộ", "mua lại"]
    negative_keywords = ["xấu", "dở", "tệ", "kém", "thất vọng", "không tốt", "hư", "thiu", "ít", "khó ăn", "sơ sài", "hết hạn"]
    text_lower = text.lower()
    if any(word in text_lower for word in positive_keywords):
        return "tích cực"
    elif any(word in text_lower for word in negative_keywords):
        return "tiêu cực"
    else:
        return "trung tính"

def check_captcha(browser):
    try:
        captcha_element = browser.find_element(By.CSS_SELECTOR, "iframe[src*='captcha'], div.captcha-container")
        if captcha_element.is_displayed():
            print("⚠️ CAPTCHA xuất hiện! Vui lòng giải quyết thủ công...")
            while captcha_element.is_displayed():
                time.sleep(2)
            print("✅ CAPTCHA đã được giải quyết, tiếp tục crawl.")
    except:
        pass

def scroll_to_bottom(browser, pause_time=1.5, max_scrolls=20):
    last_height = browser.execute_script("return document.body.scrollHeight")
    scrolls = 0
    while scrolls < max_scrolls:
        browser.execute_script("window.scrollBy(0, 500)")
        time.sleep(pause_time)
        new_height = browser.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        scrolls += 1

def product_on_page(browser, keyword, max_pages=50):
    url = "https://www.lazada.vn/"
    browser.get(url)
    browser.implicitly_wait(10)
    check_captcha(browser)
    search_box = browser.find_element(By.ID, "q")
    search_box.clear()
    search_box.send_keys(keyword)
    search_box.submit()
    WebDriverWait(browser, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".Bm3ON"))
    )
    check_captcha(browser)
    products = []
    current_page = 1
    while True: 
        print(f" Đang crawl trang {current_page}...")
        for _ in range(3):
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
        soup = BeautifulSoup(browser.page_source, "html.parser")
        product_elements = soup.select(".Bm3ON")
        for product in product_elements:
            try:
                link = product.select_one("a")["href"]
                if not link.startswith("http"):
                    link = "https:" + link  
                name = product.select_one(".RfADt a").get_text(strip=True) if product.select_one(".RfADt a") else None
                price = product.select_one(".ooOxS").get_text(strip=True) if product.select_one(".ooOxS") else None
                sold = product.select_one("._1cEkb").get_text(strip=True) if product.select_one("._1cEkb") else None
                products.append({
                    "name": name,
                    "price": price,
                    "sold": sold,
                    "link": link
                })
            except Exception as e:
                print(" Lỗi khi lấy sp:", e)
        if current_page >= max_pages:
            print(f"Đã đạt {max_pages} trang, dừng lại.")
            break
        try:
            next_button = browser.find_element(By.CSS_SELECTOR, "li.ant-pagination-next")
            if "ant-pagination-disabled" in next_button.get_attribute("class"):
                print("Hết trang, dừng lại.")
                break
            browser.execute_script("arguments[0].click();", next_button)
            time.sleep(3)  
            current_page += 1
            check_captcha(browser)
        except Exception as e:
            print(" Không tìm thấy nút Next:", e)
            break
    return products

def get_product_details(browser, url):
    try:
        browser.get(url)
        time.sleep(2)
        check_captcha(browser)

        WebDriverWait(browser, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
        )

        scroll_to_bottom(browser, pause_time=1.5, max_scrolls=20)
        soup = BeautifulSoup(browser.page_source, "html.parser")

        store_name = soup.select_one(".seller-name-retail-v2__detail-name, .seller-name-v2__detail-name")
        store_name = store_name.get_text(strip=True) if store_name else None

        categories = soup.select("li.breadcrumb_item")
        category = categories[1].get_text(strip=True) if len(categories) > 1 else None

        promo_price = soup.select_one(".pdp-price, .pdp-price_type_normal, .pdp-v2-product-price-content-salePrice-amount")
        promo_price = promo_price.get_text(strip=True) if promo_price else None

        original_price = soup.select_one(".pdp-v2-product-price-content-originalPrice-amount")
        original_price = original_price.get_text(strip=True) if original_price else None

        discount_percent = soup.select_one(".pdp-v2-product-price-content-originalPrice-discount")
        discount_percent = discount_percent.get_text(strip=True) if discount_percent else None

        rating = soup.select_one(".container-star-v2-score")
        rating = rating.get_text(strip=True) if rating else None

        comment_count = soup.select_one(".pdp-mod-review-v2 .title-text, .container-star-v2-count")
        comment_count = comment_count.get_text(strip=True) if comment_count else None

        return {
            "category": category,
            "store_name": store_name,
            "original_price": original_price,
            "discount_percent": discount_percent,
            "price": promo_price,
            "rating": rating,
            "comment_count": comment_count,
        }

    except Exception as e:
        print("Lỗi khi mở chi tiết:", e)
        return None

if __name__ == "__main__":
    browser = webdriver.Chrome()
    browser.maximize_window()
    keyword = "đồ ăn"
    product_links = product_on_page(browser, keyword, max_pages=50)
    product_detail = []
    for i, product in enumerate(product_links):
        print(f" Đang lấy chi tiết {i+1}/{len(product_links)}: {product['link']}")
        details = get_product_details(browser, product["link"])
        if details:
            details.update(product)
            product_detail.append(details)
    browser.quit()
    df = pd.DataFrame(product_detail)
    print(df.head())
    os.makedirs("./data_science/data", exist_ok=True)
    csv_path = "./data_science/data/product_details.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f" Đã lưu {len(df)} sản phẩm vào {csv_path}")
    df_loaded = pd.read_csv(csv_path)
    print(df_loaded.head())
    print(f"Tổng số dòng đọc được: {len(df_loaded)}")
