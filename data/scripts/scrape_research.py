import requests
import json
import time
from datetime import datetime

# Yann LeCun's Semantic Scholar author ID (Facebook/NYU, hIndex 138, 403 papers)
LECUN_S2_ID = "1688882"

S2_BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"User-Agent": "LeCunResearchScraper/1.0 (academic research)"}


def fetch_s2_papers():
    """Fetch all papers by LeCun from Semantic Scholar with full metadata."""
    papers = []
    fields = "title,year,abstract,venue,url,citationCount,externalIds,authors,publicationTypes,publicationDate"
    limit = 100
    offset = 0

    print("Fetching papers from Semantic Scholar...")
    while True:
        url = (
            f"{S2_BASE}/author/{LECUN_S2_ID}/papers"
            f"?fields={fields}&limit={limit}&offset={offset}"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 429:
                print("  Rate limited — waiting 10s...")
                time.sleep(10)
                continue
            if resp.status_code != 200:
                print(f"  S2 error {resp.status_code} at offset {offset}")
                break
            data = resp.json()
        except Exception as e:
            print(f"  Request error: {e}")
            break

        batch = data.get("data", [])
        if not batch:
            break

        for paper in batch:
            external = paper.get("externalIds") or {}
            arxiv_id = external.get("ArXiv", "")
            doi = external.get("DOI", "")

            # Prefer ArXiv URL, then S2 url, then DOI
            if arxiv_id:
                paper_url = f"https://arxiv.org/abs/{arxiv_id}"
            elif paper.get("url"):
                paper_url = paper["url"]
            elif doi:
                paper_url = f"https://doi.org/{doi}"
            else:
                paper_url = ""

            authors = [a.get("name", "") for a in (paper.get("authors") or [])]

            papers.append({
                "paper_id": paper.get("paperId", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "publication_date": paper.get("publicationDate", ""),
                "venue": paper.get("venue", ""),
                "publication_types": paper.get("publicationTypes") or [],
                "authors": authors,
                "abstract": paper.get("abstract", "") or "",
                "citation_count": paper.get("citationCount", 0),
                "url": paper_url,
                "arxiv_id": arxiv_id,
                "doi": doi,
            })

        total_fetched = offset + len(batch)
        print(f"  Fetched {total_fetched} papers so far...")

        if not data.get("next"):
            break
        offset += limit
        time.sleep(1.0)  # be polite to the API

    return papers


def _parse_arxiv_abstract(xml_text):
    """Extract abstract from ArXiv Atom feed XML."""
    start = xml_text.find("<summary>")
    end = xml_text.find("</summary>")
    if start != -1 and end != -1:
        return xml_text[start + 9:end].strip().replace("\n", " ")
    return ""


def enrich_via_s2_batch(papers):
    """Use S2 batch endpoint to fetch abstracts for papers missing them."""
    missing = [p for p in papers if not p["abstract"] and p["paper_id"]]
    if not missing:
        return papers

    print(f"\nEnriching {len(missing)} papers via Semantic Scholar batch API...")
    id_to_paper = {p["paper_id"]: p for p in missing}
    ids = list(id_to_paper.keys())
    batch_size = 100  # safe batch size
    enriched = 0

    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        try:
            resp = requests.post(
                f"{S2_BASE}/paper/batch",
                params={"fields": "paperId,abstract,externalIds"},
                json={"ids": batch_ids},
                headers=HEADERS,
                timeout=30,
            )
            if resp.status_code == 429:
                print("  Rate limited — waiting 30s...")
                time.sleep(30)
                # Retry once
                resp = requests.post(
                    f"{S2_BASE}/paper/batch",
                    params={"fields": "paperId,abstract,externalIds"},
                    json={"ids": batch_ids},
                    headers=HEADERS,
                    timeout=30,
                )
            if resp.status_code != 200:
                print(f"  Batch error {resp.status_code} at offset {start}")
                time.sleep(5)
                continue

            results = resp.json()
            for result in results:
                if not result:
                    continue
                pid = result.get("paperId", "")
                abstract = result.get("abstract") or ""
                if abstract and pid in id_to_paper:
                    id_to_paper[pid]["abstract"] = abstract
                    enriched += 1
                # Also grab ArXiv ID if missing
                external = result.get("externalIds") or {}
                arxiv_id = external.get("ArXiv", "")
                if arxiv_id and pid in id_to_paper and not id_to_paper[pid]["arxiv_id"]:
                    id_to_paper[pid]["arxiv_id"] = arxiv_id
                    if not id_to_paper[pid]["url"]:
                        id_to_paper[pid]["url"] = f"https://arxiv.org/abs/{arxiv_id}"

            print(f"  Batch {start//batch_size + 1}: enriched {enriched} total so far")
        except Exception as e:
            print(f"  Batch request error: {e}")
        time.sleep(3)  # respect rate limit (100 req/5min without key)

    print(f"  S2 batch enriched {enriched}/{len(missing)} papers.")
    return papers


def fetch_personal_page_data():
    """
    Scrape yann.lecun.com/exdb/publis/ to get:
    - Title -> description mapping from the 'Selected Papers' section
    - Title -> PDF URL mapping from the full list
    Returns (descriptions_dict, pdf_urls_dict) keyed by lowercased title.
    """
    import re
    from bs4 import BeautifulSoup

    descriptions = {}
    pdf_urls = {}

    try:
        resp = requests.get(
            "http://yann.lecun.com/exdb/publis/",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        if resp.status_code != 200:
            return descriptions, pdf_urls

        soup = BeautifulSoup(resp.text, "html.parser")
        text = resp.text

        # --- Extract selected paper descriptions ---
        # Each entry is a <table> row with <td>description</td>
        # Pattern: <b>Title</b> ...description text...
        selected_section_start = text.find('name="selected"')
        fulllist_section_start = text.find('name="fulllist"')
        if selected_section_start != -1 and fulllist_section_start != -1:
            selected_html = text[selected_section_start:fulllist_section_start]
            selected_soup = BeautifulSoup(selected_html, "html.parser")
            for td in selected_soup.find_all("td"):
                bolds = td.find_all("b")
                for bold in bolds:
                    title = bold.get_text(strip=True)
                    if len(title) > 10:
                        # Get all text in the td cell as the description
                        cell_text = td.get_text(" ", strip=True)
                        # Remove the title itself from the front
                        desc = cell_text.replace(title, "", 1).strip()
                        desc = re.sub(r"\s+", " ", desc)
                        if len(desc) > 50:
                            descriptions[title.lower()] = desc

        # --- Extract PDF URLs from the full list ---
        # PDF links are like: <a href='pdf/lecun-89.pdf'>
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("pdf/") or href.startswith("/exdb/publis/pdf/"):
                # Find the nearest <b> sibling that contains a title
                parent = a_tag.find_parent("tr") or a_tag.find_parent("td")
                if parent:
                    bold = parent.find("b")
                    if bold:
                        title = bold.get_text(strip=True).lower()
                        full_url = f"http://yann.lecun.com/exdb/publis/{href}"
                        pdf_urls[title] = full_url

        print(f"  Personal page: {len(descriptions)} descriptions, {len(pdf_urls)} PDF URLs")
    except Exception as e:
        print(f"  Personal page error: {e}")

    return descriptions, pdf_urls


def enrich_via_personal_page(papers):
    """Match personal page descriptions and PDF URLs to papers by title."""
    print("\nFetching Yann LeCun's personal publications page...")
    descriptions, pdf_urls = fetch_personal_page_data()
    if not descriptions and not pdf_urls:
        return papers

    enriched_abstracts = 0
    enriched_urls = 0
    for paper in papers:
        title_lower = paper["title"].lower()

        # Try exact match first, then fuzzy (first 6 words)
        desc = descriptions.get(title_lower)
        if not desc:
            title_prefix = " ".join(title_lower.split()[:6])
            for key, val in descriptions.items():
                if key.startswith(title_prefix) or title_prefix in key:
                    desc = val
                    break

        if desc and not paper["abstract"]:
            paper["abstract"] = desc
            enriched_abstracts += 1

        pdf = pdf_urls.get(title_lower)
        if pdf and not paper["url"]:
            paper["url"] = pdf
            enriched_urls += 1

    print(f"  Personal page enriched {enriched_abstracts} abstracts, {enriched_urls} URLs.")
    return papers


def enrich_via_crossref(papers):
    """Use CrossRef API to fetch abstracts for papers with DOIs."""
    missing = [p for p in papers if not p["abstract"] and p["doi"]]
    if not missing:
        return papers

    print(f"\nEnriching {len(missing)} papers via CrossRef...")
    enriched = 0
    crossref_base = "https://api.crossref.org/works"
    crossref_headers = {**HEADERS, "mailto": "research@example.com"}

    for i, paper in enumerate(missing):
        try:
            resp = requests.get(
                f"{crossref_base}/{paper['doi']}",
                headers=crossref_headers,
                timeout=15,
            )
            if resp.status_code == 200:
                work = resp.json().get("message", {})
                abstract = work.get("abstract", "")
                if abstract:
                    # CrossRef wraps abstract in JATS XML tags — strip them
                    import re
                    abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
                    abstract = re.sub(r"\s+", " ", abstract)
                    paper["abstract"] = abstract
                    enriched += 1
        except Exception:
            pass
        time.sleep(0.3)

    print(f"  CrossRef enriched {enriched}/{len(missing)} papers.")
    return papers


def enrich_via_arxiv_titles(papers):
    """For papers still missing abstracts, search ArXiv by title."""
    missing = [p for p in papers if not p["abstract"]]
    if not missing:
        return papers

    print(f"\nEnriching {len(missing)} papers via ArXiv title search...")
    arxiv_api = "https://export.arxiv.org/api/query"
    enriched = 0

    for i, paper in enumerate(missing):
        abstract = ""

        # Try by ArXiv ID first if we have one
        if paper["arxiv_id"]:
            try:
                resp = requests.get(
                    arxiv_api,
                    params={"id_list": paper["arxiv_id"], "max_results": 1},
                    headers=HEADERS,
                    timeout=15,
                )
                if resp.status_code == 200:
                    abstract = _parse_arxiv_abstract(resp.text)
            except Exception:
                pass
            time.sleep(0.4)

        # Try title search
        if not abstract:
            try:
                resp = requests.get(
                    arxiv_api,
                    params={"search_query": f'ti:"{paper["title"]}"', "max_results": 1},
                    headers=HEADERS,
                    timeout=15,
                )
                if resp.status_code == 200 and "<entry>" in resp.text:
                    candidate = _parse_arxiv_abstract(resp.text)
                    # Light sanity check: first two words of title appear in result
                    title_words = paper["title"].lower().split()[:2]
                    if candidate and any(w in resp.text.lower() for w in title_words):
                        abstract = candidate
                        # Grab ArXiv ID if found
                        id_start = resp.text.find("<id>http://arxiv.org/abs/")
                        id_end = resp.text.find("</id>", id_start)
                        if id_start != -1 and id_end != -1:
                            arxiv_url = resp.text[id_start + 4:id_end].strip()
                            arxiv_id = arxiv_url.split("/abs/")[-1]
                            paper["arxiv_id"] = arxiv_id
                            if not paper["url"]:
                                paper["url"] = arxiv_url
            except Exception:
                pass
            time.sleep(0.4)

        if abstract:
            paper["abstract"] = abstract
            enriched += 1

    print(f"  ArXiv enriched {enriched}/{len(missing)} additional papers.")
    return papers


def deduplicate(papers):
    seen_ids = set()
    seen_titles = set()
    unique = []
    for p in papers:
        pid = p.get("paper_id")
        title_key = p.get("title", "").lower().strip()
        if pid and pid in seen_ids:
            continue
        if title_key and title_key in seen_titles:
            continue
        if pid:
            seen_ids.add(pid)
        if title_key:
            seen_titles.add(title_key)
        unique.append(p)
    return unique


def main():
    print("Scraping Yann LeCun research articles...\n")

    papers = fetch_s2_papers()
    papers = deduplicate(papers)
    papers = enrich_via_s2_batch(papers)
    papers = enrich_via_personal_page(papers)
    papers = enrich_via_crossref(papers)
    papers = enrich_via_arxiv_titles(papers)

    # Filter to papers where LeCun is actually listed as an author
    before = len(papers)
    papers = [
        p for p in papers
        if any("lecun" in a.lower() or "le cun" in a.lower() for a in p.get("authors", []))
    ]
    print(f"Filtered to {len(papers)} papers with LeCun as author (removed {before - len(papers)} misattributions)")

    # Sort by year descending, then citation count
    papers.sort(key=lambda p: (p.get("year") or 0, p.get("citation_count") or 0), reverse=True)

    has_abstract = sum(1 for p in papers if p["abstract"])
    print(f"\nTotal papers: {len(papers)}")
    print(f"Papers with abstracts: {has_abstract}")

    output = {
        "scraped_at": datetime.now().isoformat(),
        "author": "Yann LeCun",
        "semantic_scholar_id": LECUN_S2_ID,
        "total": len(papers),
        "papers_with_abstracts": has_abstract,
        "papers": papers,
    }

    out_path = "lecun_research.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
