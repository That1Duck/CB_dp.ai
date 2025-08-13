from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from urllib.parse import urlencode, urljoin
import json, time

BASE_URL = "https://www.deeplearning.ai"
COURSES_LIST_BASE = f"{BASE_URL}/courses/"

def collecting_courses(start_page: int = 1, max_pages: int = 7, delay_s: float = 0.3):

    data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for page_num in range(start_page, start_page + max_pages):
            query = urlencode({"courses_date_desc[page]": page_num})
            list_url = f"{COURSES_LIST_BASE}?{query}"

            page.goto(list_url, timeout=60000, wait_until="domcontentloaded")

            try:
                page.wait_for_selector("a.dlai-gtm-course-card-featured", timeout=5000)
            except PlaywrightTimeoutError:
                pass

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")
            course_cards = soup.select("a.dlai-gtm-course-card-featured")


            if not course_cards:
                if page_num == start_page:
                    print(f"Attention: no cards found on page {list_url}.")
                break

            for card in course_cards:
                h3 = card.find('h3')
                p_tag = card.find('p')

                title = h3.text.strip() if h3 else ""
                description = p_tag.text.strip() if p_tag else ""

                href = card.get("href") or ""
                url = urljoin(BASE_URL, href)

                data.append({
                    "title": title,
                    "description": description,
                    "url": url,
                    "type": "course",
                    "source": "deeplearning.ai"
                })

            if delay_s:
                time.sleep(delay_s)

        browser.close()


    with open("data/raw/courses.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Parsing of courses is done. Pages: {page_num - start_page + 1}. Courses: {len(data)}")

if __name__ == "__main__":
    collecting_courses()