"""Показує HTML блоку ціни в перших картках Telemart (щоб вибрати правильний селектор актуальної ціни)."""
import httpx
from bs4 import BeautifulSoup

from wsbench.collectors.common import DEFAULT_HEADERS

r = httpx.get("https://telemart.ua/ua/laptops/", headers=DEFAULT_HEADERS, timeout=30, follow_redirects=True)
soup = BeautifulSoup(r.text, "lxml")
for card in soup.select(".product-item")[:3]:
    print("НАЗВА:", card.select_one(".product-item__title a").get_text(" ", strip=True))
    for el in card.select(".product-cost"):
        print(el.parent.prettify()[:1500])
    print("-" * 60)
