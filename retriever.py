import os
import requests
from bs4 import BeautifulSoup

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def get_stock_news(ticker):
    """Fetch news articles related to a stock ticker from Alpha Vantage."""
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("Missing API key for Alpha Vantage!")

    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code}, {response.text}")
        return []

    news_data = response.json()
    if "feed" not in news_data:
        print("Unexpected API response:", news_data)
        return []

    news = []
    for article in news_data["feed"]:
        news.append({
            "title": article["title"],
            "link": article["url"]
        })

    return news
