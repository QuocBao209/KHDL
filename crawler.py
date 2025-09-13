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
    positive_keywords = ["tốt", "ngon", "đẹp", "ok", "hài lòng", "ổn", "chất lượng"]
    negative_keywords = ["xấu", "dở", "tệ", "kém", "thất vọng", "không tốt"]

    text_lower = text.lower()
    if any(word in text_lower for word in positive_keywords):
        return "tích cực"
    elif any(word in text_lower for word in negative_keywords):
        return "tiêu cực"
    else:
        return "trung tính"

def product_on_page(browser, keyword, max_pages=50):
    url = "https://www.lazada.vn/"
    browser.get(url)
    browser.implicitly_wait(10)

    search_box = browser.find_element(By.ID, "q")
    search_box.clear()
    search_box.send_keys(keyword)
    search_box.submit()

    WebDriverWait(browser, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".Bm3ON"))
    )

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
        except Exception as e:
            print(" Không tìm thấy nút Next:", e)
            break

    return products


def get_product_details(browser, url):
    try:
        browser.get(url)
        time.sleep(2)

        WebDriverWait(browser, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
        )

        soup = BeautifulSoup(browser.page_source, "html.parser")

        product_name = soup.select_one("h1.pdp-mod-product-badge-title, h1.pdp-product-title")
        product_name = product_name.get_text(strip=True) if product_name else None

        store_name = soup.select_one("div.seller-name__detail > a")
        store_name = store_name.get_text(strip=True) if store_name else None

        price = soup.select_one(".pdp-price.pdp-price_type_normal")
        price = price.get_text(strip=True) if price else None

        rating = soup.select_one("span.score-average")
        rating = rating.get_text(strip=True) if rating else None

        comment_count = soup.select_one("a.pdp-link.pdp-review-summary__link")
        comment_count = comment_count.get_text(strip=True) if comment_count else None

        review_text = None
        review_element = soup.select_one("div.item-content div.content")
        if review_element:
            review_text = review_element.get_text(strip=True)

        sentiment = sentiment_analysis(review_text)

        print(f" Lấy thành công: {product_name}")

        return {
            "product_name": product_name,
            "store_name": store_name,
            "price": price,
            "rating": rating,
            "comment_count": comment_count,
            "sentiment": sentiment,
        }
    except Exception as e:
        print(" Lỗi khi mở chi tiết:", e)
        return None


if __name__ == "_main_":
    browser = webdriver.Chrome()
    browser.maximize_window()

    keyword = "đồ ăn"
    product_links = product_on_page(browser, keyword, max_pages=5)

    product_detail = []

    for i, product in enumerate(product_links):
        print(f" Đang lấy chi tiết {i+1}/{len(product_links)}: {product['link']}")
        details = get_product_details(browser, product["link"])
        if details:
            details.update(product)
            product_detail.append(details)

    browser.quit()

    df = pd.DataFrame(product_detail)
    print(df)

    os.makedirs("../data/raw", exist_ok=True)
    csv_path = "../data/raw/product_details.csv"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f" Đã lưu {len(df)} sản phẩm vào {csv_path}")