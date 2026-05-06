import os
import requests  # For sending HTTP requests to web pages
from bs4 import BeautifulSoup  # For parsing HTML content
from textblob import TextBlob  # For sentiment analysis
import re
import json

# Azure OpenAI credentials (replace with your actual values)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g., "gpt-4"

# Main news page URL
main_url = "https://www.technologyreview.com/"

def get_article_links(url):
    """
    Extracts likely article links from the main news page using a broader regex pattern.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = []
    article_pattern = re.compile(r'/\d{4}/\d{2}/\d{2}/')
    for a in soup.find_all('a', href=True):
        href = a['href']
        if article_pattern.search(href):
            full_url = requests.compat.urljoin(url, href)
            links.append(full_url)
    if not links:
        all_internal = [requests.compat.urljoin(url, a['href']) for a in soup.find_all('a', href=True) if a['href'].startswith('/')]
        print("All internal links for inspection:", all_internal)
    else:
        print("Sample article links found:", links[:10])
    return list(set(links))

def extract_article_content(url):
    """
    Fetches the content of the given article URL and extracts all paragraph text.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    return ' '.join([p.get_text() for p in paragraphs])

def chunk_text(text, max_words=400):
    """
    Splits text into chunks of max_words words.
    """
    words = text.split()
    for i in range(0, len(words), max_words):
        yield ' '.join(words[i:i + max_words])

def azure_openai_summarize(text):
    """
    Calls Azure OpenAI GPT-4 REST API to summarize the given text.
    """
    url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version=2024-02-15-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    prompt = f"Summarize the following news article in 3-5 sentences:\n\n{text[:50000]}"
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that summarizes news articles."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        print(f"Azure OpenAI API error: {response.status_code} {response.text}")
        return ""

def summarize_and_filter(text, keywords):
    """
    Summarizes the given text using Azure OpenAI, checks if summary contains any keywords,
    and filters based on positive sentiment.
    """
    summaries = []
    for chunk in chunk_text(text):
        summary = azure_openai_summarize(chunk)
        if summary:
            summaries.append(summary)
    full_summary = ' '.join(summaries)
    sentiment = TextBlob(full_summary).sentiment.polarity
    relevance = any(k.lower() in full_summary.lower() for k in keywords)
    return full_summary if relevance and sentiment > 0 else None

# List of keywords to filter relevant summaries
keywords = ["technology", "AI", "innovation", "research"]

# Get individual article URLs from the main page
article_urls = get_article_links(main_url)
print(f"Found {len(article_urls)} article URLs.")

if not article_urls:
    print("No article URLs found. Check your link extraction logic or the website structure.")

# --- Summaries code block commented out ---
# Only process the first article
# if article_urls:
#     url = article_urls[0]
#     article = extract_article_content(url)
#     output_lines = []
#     if not article:
#         output_lines.append(f"No content found for {url}\n")
#     else:
#         def azure_openai_summarize_short(text):
#             """
#             Calls Azure OpenAI GPT-4 REST API to summarize the given text in less than 100 words.
#             Returns the summary and the number of tokens used.
#             """
#             api_url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version=2024-02-15-preview"
#             headers = {
#                 "Content-Type": "application/json",
#                 "api-key": AZURE_OPENAI_KEY
#             }
#             prompt = (
#                 "Summarize the following news article in less than 100 words. "
#                 "Be concise and only include the most important points:\n\n"
#                 f"{text[:5000]}"
#             )
#             payload = {
#                 "messages": [
#                     {"role": "system", "content": "You are a helpful assistant that summarizes news articles."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 "max_tokens": 120,  # Limit tokens to minimize usage
#                 "temperature": 0.5
#             }
#             response = requests.post(api_url, headers=headers, data=json.dumps(payload))
#             if response.status_code == 200:
#                 result = response.json()
#                 summary = result["choices"][0]["message"]["content"].strip()
#                 # Get token usage from API response
#                 tokens_used = result.get("usage", {}).get("total_tokens", "N/A")
#                 return summary, tokens_used
#             else:
#                 print(f"Azure OpenAI API error: {response.status_code} {response.text}")
#                 return "", "N/A"
#         summary, tokens_used = azure_openai_summarize_short(article)
#         output_lines.append(f"Summary for {url}:\n{summary}\n")
#         output_lines.append(f"Tokens used: {tokens_used}\n")
#     with open("summaries.txt", "w", encoding="utf-8") as f:
#         f.writelines(output_lines)
#     print("Summary for first article written to summaries.txt")
# else:
#     print("No article URLs found. Check your link extraction logic or the website structure.")

def get_headlines_and_links(topic_url):
    """
    Extracts headlines and article links from a Technology Review topic page.
    """
    response = requests.get(topic_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    headlines_links = []
    # Articles are often in <article> tags or have headline classes
    for article in soup.find_all('article'):
        # Try headline in <h2> or <h3> inside article
        headline_tag = article.find(['h2', 'h3'])
        if headline_tag:
            headline = headline_tag.get_text(strip=True)
            # Find the first <a> with href inside the headline tag
            a_tag = headline_tag.find('a', href=True)
            if a_tag:
                link = a_tag['href']
                if headline and '/20' in link:
                    full_url = requests.compat.urljoin(topic_url, link)
                    headlines_links.append((headline, full_url))
    # Fallback: check for <a> tags with headline-like text and year in link
    if not headlines_links:
        for a_tag in soup.find_all('a', href=True):
            headline = a_tag.get_text(strip=True)
            link = a_tag['href']
            if headline and '/20' in link:
                full_url = requests.compat.urljoin(topic_url, link)
                headlines_links.append((headline, full_url))
    return headlines_links

# Topic URLs
topic_urls = [
    "https://www.technologyreview.com/topic/artificial-intelligence/",
    "https://www.technologyreview.com/topic/computing/"
]

def extract_headline(url):
    """
    Fetches the headline from the given article URL.
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
    except Exception as e:
        print(f"Error fetching headline from {url}: {e}")
    return ""

output_lines = []
total_tokens = 0

for topic_url in topic_urls:
    # Get all internal links from the topic page
    response = requests.get(topic_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    all_internal = [requests.compat.urljoin(topic_url, a['href']) for a in soup.find_all('a', href=True) if a['href'].startswith('/')]
    output_lines.append(f"Headlines from {topic_url}:\n")
    for url in all_internal:
        headline = extract_headline(url)
        if headline:
            tokens_used = len(headline.split())
            total_tokens += tokens_used
            output_lines.append(f"Headline: {headline}\nLink: {url}\nTokens used: {tokens_used}\n\n")
        # Optionally, skip links without headlines to keep output clean

output_lines.append(f"Total tokens used for all headlines: {total_tokens}\n")

with open("headlines.txt", "w", encoding="utf-8") as f:
    f.writelines(output_lines)

print("Headlines and links written to headlines.txt")
