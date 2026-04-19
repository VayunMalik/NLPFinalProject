import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime


def search_youtube(query, max_results=10):
    interviews = []
    url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return interviews
        match = re.search(r'var ytInitialData = ({.*?});</script>', response.text)
        if not match:
            return interviews
        data = json.loads(match.group(1))
        contents = (
            data['contents']['twoColumnSearchResultsRenderer']
                ['primaryContents']['sectionListRenderer']
                ['contents'][0]['itemSectionRenderer']['contents']
        )
        for item in contents[:max_results]:
            if 'videoRenderer' not in item:
                continue
            video = item['videoRenderer']
            video_id = video.get('videoId', '')
            title = video.get('title', {}).get('runs', [{}])[0].get('text', '')
            channel = ''
            if 'ownerText' in video:
                channel = video['ownerText'].get('runs', [{}])[0].get('text', '')
            date = ''
            if 'publishedTimeText' in video:
                date = video['publishedTimeText'].get('simpleText', '')
            description = ''
            if 'detailedMetadataSnippets' in video:
                snippets = video['detailedMetadataSnippets']
                if snippets:
                    runs = snippets[0].get('snippetText', {}).get('runs', [])
                    description = ''.join(r.get('text', '') for r in runs)
            if video_id and title:
                interviews.append({
                    'title': title,
                    'source': f'YouTube — {channel}' if channel else 'YouTube',
                    'date': date,
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'description': description or f'YouTube video featuring Yann LeCun ({channel})',
                })
    except Exception as e:
        print(f"  YouTube error: {e}")
    return interviews


def search_duckduckgo(query, max_results=15):
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        print("  ddgs not installed — skipping DDG search.")
        return []
    interviews = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        for r in results:
            href = r.get('href', '')
            domain = href.split('/')[2] if href and '/' in href else 'Web'
            interviews.append({
                'title': r.get('title', ''),
                'source': domain,
                'date': '',
                'url': href,
                'description': (r.get('body', '') or '')[:300],
            })
    except Exception as e:
        print(f"  DDG error: {e}")
    return interviews


def search_google_news(query):
    interviews = []
    url = f"https://news.google.com/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return interviews
        soup = BeautifulSoup(response.text, 'html.parser')
        for article in soup.find_all('article')[:20]:
            try:
                title_tag = article.find('a', class_='JtKRv') or article.find('h3') or article.find('h4')
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                link = title_tag.get('href', '')
                if link.startswith('./'):
                    link = 'https://news.google.com' + link[1:]
                time_tag = article.find('time')
                date = time_tag.get('datetime', '') if time_tag else ''
                source_tag = article.find('div', class_='vr1PYe')
                source = source_tag.get_text(strip=True) if source_tag else 'Google News'
                if title:
                    interviews.append({
                        'title': title,
                        'source': source,
                        'date': date,
                        'url': link,
                        'description': f'News article: {title}',
                    })
            except Exception:
                continue
    except Exception as e:
        print(f"  Google News error: {e}")
    return interviews


def deduplicate(interviews):
    seen = set()
    unique = []
    for item in interviews:
        key = item.get('url', '') or item.get('title', '')
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def is_relevant(item):
    text = (item.get('title', '') + ' ' + item.get('description', '')).lower()
    return 'lecun' in text or ('yann' in text and ('ai' in text or 'interview' in text
                                                    or 'machine learning' in text or 'deep learning' in text))


def is_english(item):
    """Return True if the item appears to be English-language content."""
    from langdetect import detect, LangDetectException
    text = ' '.join(filter(None, [item.get('title', ''), item.get('description', '')])).strip()
    if not text:
        return True  # can't determine; keep it
    # Fast path: French keywords in URL or title are a strong signal
    url = item.get('url', '').lower()
    if '/fr/' in url or url.endswith('.fr') or '?lang=fr' in url:
        return False
    try:
        return detect(text) != 'fr'
    except LangDetectException:
        return True  # ambiguous; keep it


def main():
    print("Scraping Yann LeCun interviews...\n")
    all_results = []

    youtube_queries = [
        "Yann LeCun interview",
        "Yann LeCun podcast",
        "Yann LeCun AI discussion",
    ]
    for q in youtube_queries:
        print(f"YouTube: {q}")
        all_results.extend(search_youtube(q, max_results=10))
        time.sleep(1.5)

    ddg_queries = [
        '"Yann LeCun" interview podcast 2023 OR 2024',
        '"Yann LeCun" interview site:youtube.com',
        '"Yann LeCun" lex fridman interview',
        '"Yann LeCun" interview artificial intelligence',
        '"Yann LeCun" podcast npr OR wired OR mit OR verge',
    ]
    for q in ddg_queries:
        print(f"DDG: {q}")
        all_results.extend(search_duckduckgo(q, max_results=15))
        time.sleep(0.8)

    print("Google News: Yann LeCun interview")
    all_results.extend(search_google_news("Yann LeCun interview"))

    unique = deduplicate(all_results)
    filtered = [i for i in unique if is_relevant(i) and is_english(i)]

    # Clean up empty fields
    for item in filtered:
        for k in ('title', 'source', 'date', 'url', 'description'):
            item.setdefault(k, '')

    print(f"\nFound {len(filtered)} unique results.")

    output = {
        'scraped_at': datetime.now().isoformat(),
        'total': len(filtered),
        'interviews': filtered,
    }

    with open('lecun_interviews.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Saved to lecun_interviews.json")


if __name__ == '__main__':
    main()
