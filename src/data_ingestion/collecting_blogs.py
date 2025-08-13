import requests
import json
from bs4 import BeautifulSoup


BASE_URL = "https://www.deeplearning.ai"
BLOG_URL = f"{BASE_URL}/blog/"

def collecting_blogs(pages_count = 4):
    data = []

    for page in range(1, pages_count + 1):
        # Get url
        url = BLOG_URL if page == 1 else f"{BLOG_URL}page/{page}/"
        response = requests.get(url)
        # Check for connection
        if response.status_code != 200:
            print("Error loading https://www.deeplearning.ai/blog/")
            return

        # Parsing
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("article")

        for article in articles:
            title = article.find("h2").text.strip() if article.find('h2') else ""
            description = article.find("p").text.strip() if article.find("p") else ""

            a_tag = article.find("a", href=True)
            href = a_tag.get("href") if a_tag else ""
            url = BASE_URL + href if href.startswith("/") else href

            data.append({
                "title": title,
                "description": description,
                "url": url,
                "type": "blog",
                "source": "deeplearning.ai"
            })

    with open("data/raw/blog.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Parsing of blogs is done. Amount of blogs: {len(data)}")

if __name__ == "__main__":
    collecting_blogs(pages_count=7)