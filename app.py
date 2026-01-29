# app.py
# Whole-Site Schema Scanner (Homepage + 3 Product Pages)
# UPDATED: Fixes "scanning XML files" bug by recursively parsing nested sitemaps.

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup


# ----------------------------
# Config
# ----------------------------
DEFAULT_TIMEOUT = 15
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
# REPLACE THIS WITH YOUR CALENDLY LINK
BOOKING_URL = "https://calendly.com/your-handle/fix-it-call"


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class PageAudit:
    url: str
    final_url: str
    ok_fetch: bool
    fetch_error: Optional[str]

    org_present: bool
    org_has_identity: bool
    faq_present: bool
    product_present: bool
    product_has_offers: bool

    score: int
    warnings: List[str]


# ----------------------------
# Networking helpers
# ----------------------------
def safe_base_url(input_url: str) -> str:
    """Ensure URL has scheme and return base origin (scheme + host)."""
    u = input_url.strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    parsed = urlparse(u)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return origin


def fetch_text(url: str, timeout: int) -> Tuple[str, str]:
    """Fetch URL and return (final_url, text). Raises requests exceptions on failure."""
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text


def is_internal(url: str, origin: str) -> bool:
    """True if url is internal to origin host."""
    try:
        return urlparse(url).netloc == urlparse(origin).netloc
    except Exception:
        return False


def normalize_url(u: str) -> str:
    """Remove fragments; keep query."""
    p = urlparse(u)
    return p._replace(fragment="").geturl()


# ----------------------------
# JSON-LD parsing helpers
# ----------------------------
def normalize_schema_type(t: Any) -> List[str]:
    out: List[str] = []
    if t is None:
        return out
    if isinstance(t, str):
        out.append(t.strip().lower())
    elif isinstance(t, list):
        for item in t:
            out.extend(normalize_schema_type(item))
    elif isinstance(t, dict):
        for k in ("@type", "type", "name"):
            if k in t:
                out.extend(normalize_schema_type(t.get(k)))
    return [x for x in out if x]


def iter_json_objects(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from iter_json_objects(v)
    elif isinstance(node, list):
        for item in node:
            yield from iter_json_objects(item)


def _try_json_parse(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Remove JS comments
    no_comments = re.sub(r"//.*?$|/\*.*?\*/", "", raw, flags=re.MULTILINE | re.DOTALL).strip()
    # Remove trailing commas
    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", no_comments)
    try:
        return json.loads(no_trailing_commas)
    except Exception:
        return None


def _split_possible_json_blocks(raw: str) -> List[str]:
    blocks: List[str] = []
    i, n = 0, len(raw)
    while i < n:
        start = None
        for j in range(i, n):
            if raw[j] in "{[":
                start = j
                break
        if start is None:
            break
        open_ch = raw[start]
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        for k in range(start, n):
            ch = raw[k]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == open_ch: depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        blocks.append(raw[start : k + 1].strip())
                        i = k + 1
                        break
        else:
            break
    return blocks or [raw]


def extract_jsonld_payloads(html: str) -> Tuple[List[Any], int]:
    soup = BeautifulSoup(html, "lxml")
    scripts = soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)})
    payloads: List[Any] = []
    for s in scripts:
        raw = (s.get_text(strip=False) or "").strip()
        if not raw: continue
        parsed = _try_json_parse(raw)
        if parsed is not None:
            payloads.append(parsed)
            continue
        for block in _split_possible_json_blocks(raw):
            p = _try_json_parse(block)
            if p is not None:
                payloads.append(p)
    return payloads, len(scripts)


def get_all_objects(payloads: List[Any]) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    for payload in payloads:
        roots = payload if isinstance(payload, list) else [payload]
        for root in roots:
            for obj in iter_json_objects(root):
                if isinstance(obj, dict):
                    if "@graph" in obj and isinstance(obj["@graph"], list):
                        for g in obj["@graph"]:
                            for o in iter_json_objects(g):
                                if isinstance(o, dict):
                                    objs.append(o)
                    objs.append(obj)
    return objs


def has_type(obj: Dict[str, Any], target: str) -> bool:
    return target.lower() in normalize_schema_type(obj.get("@type"))


def first_matching_object(objs: List[Dict[str, Any]], target_type: str) -> Optional[Dict[str, Any]]:
    for o in objs:
        if has_type(o, target_type): return o
    return None


def any_matching_objects(objs: List[Dict[str, Any]], target_type: str) -> List[Dict[str, Any]]:
    return [o for o in objs if has_type(o, target_type)]


def organization_has_identity_fields(org_obj: Dict[str, Any]) -> bool:
    disamb = org_obj.get("disambiguatingDescription")
    if isinstance(disamb, str) and disamb.strip(): return True
    same_as = org_obj.get("sameAs")
    if isinstance(same_as, str) and same_as.strip(): return True
    if isinstance(same_as, list) and any(isinstance(x, str) and x.strip() for x in same_as): return True
    return False


def product_has_offers(product_obj: Dict[str, Any]) -> bool:
    offers = product_obj.get("offers")
    if isinstance(offers, dict) and len(offers.keys()) > 0: return True
    if isinstance(offers, list) and len(offers) > 0: return True
    return False


# ----------------------------
# Strict scoring
# ----------------------------
def compute_page_score(
    org_present: bool,
    org_has_identity: bool,
    faq_present: bool,
    product_present: bool,
    product_has_price_offers: bool,
) -> int:
    score = 0
    if org_present:
        score += 10
        if org_has_identity: score += 20
    if faq_present: score += 20
    if product_present:
        score += 20
        if product_has_price_offers: score += 30
    return score


# ----------------------------
# UPDATED Crawler logic (Handles nested sitemaps)
# ----------------------------
def extract_urls_from_xml(xml_text: str) -> List[str]:
    """Simple regex to find all <loc> content."""
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml_text, flags=re.IGNORECASE)
    return [normalize_url(u.strip()) for u in locs if u.strip()]

def fetch_sitemap_urls_recursive(
    start_url: str, 
    origin: str, 
    timeout: int, 
    status_cb=None,
    depth: int = 0
) -> List[str]:
    """
    Fetch a sitemap. If it contains .xml links (child sitemaps), recurse once.
    Return ONLY html product pages (containing /products/).
    """
    # Safety break
    if depth > 1: 
        return []
    
    found_pages: List[str] = []
    
    try:
        _, xml_text = fetch_text(start_url, timeout)
        all_locs = extract_urls_from_xml(xml_text)
        
        # Split into child sitemaps and potential pages
        child_sitemaps = [u for u in all_locs if u.lower().endswith('.xml')]
        pages = [u for u in all_locs if not u.lower().endswith('.xml')]
        
        # 1. Collect pages from this level
        for p in pages:
            if is_internal(p, origin) and "/products/" in p.lower():
                found_pages.append(p)
                
        # 2. Dive into child sitemaps (especially if we haven't found enough products yet)
        # Prioritize sitemaps that look like product sitemaps
        child_sitemaps.sort(key=lambda x: 0 if "product" in x.lower() else 1)
        
        for child in child_sitemaps:
            # Don't spend too long if we already found 50+ candidates
            if len(found_pages) > 10: 
                break
                
            if status_cb:
                status_cb(f"Digging into {child.split('/')[-1]}...")
                
            # Recurse
            nested_pages = fetch_sitemap_urls_recursive(child, origin, timeout, status_cb, depth + 1)
            found_pages.extend(nested_pages)
            
    except Exception:
        pass
        
    return found_pages

def pick_product_urls_from_candidates(urls: List[str], origin: str, max_count: int = 3) -> List[str]:
    seen: Set[str] = set()
    picked: List[str] = []
    for u in urls:
        if not u: continue
        u = normalize_url(u)
        if u in seen: continue
        if not is_internal(u, origin): continue
        # Double check it is likely a product page and NOT a file
        if "/products/" not in u.lower(): continue
        if u.lower().endswith(('.xml', '.jpg', '.png', '.pdf')): continue
        
        seen.add(u)
        picked.append(u)
        if len(picked) >= max_count: break
    return picked

def scrape_homepage_for_product_links(home_html: str, origin: str, max_count: int = 3) -> List[str]:
    soup = BeautifulSoup(home_html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href: continue
        abs_url = urljoin(origin, href)
        links.append(normalize_url(abs_url))
    return pick_product_urls_from_candidates(links, origin=origin, max_count=max_count)

def find_home_and_products(origin: str, timeout: int, status_cb=None) -> Tuple[str, List[str], List[str]]:
    notes: List[str] = []
    homepage_url = origin + "/"
    product_urls: List[str] = []

    if status_cb: status_cb("Fetching homepage…")
    home_final, home_html = fetch_text(homepage_url, timeout=timeout)
    homepage_url = home_final

    # Priority 1: /sitemap.xml (with recursion)
    try:
        if status_cb: status_cb("Checking /sitemap.xml...")
        sitemap_root = urljoin(origin, "/sitemap.xml")
        
        # Use the recursive function
        candidates = fetch_sitemap_urls_recursive(sitemap_root, origin, timeout, status_cb)
        product_urls = pick_product_urls_from_candidates(candidates, origin, max_count=3)
        
        if product_urls:
            notes.append("✅ Found product URLs via /sitemap.xml")
            return homepage_url, product_urls, notes
        else:
            notes.append("⚠️ /sitemap.xml found, but no product pages inside")
    except Exception:
        notes.append("⚠️ /sitemap.xml not accessible")

    # Priority 2: robots.txt
    try:
        if status_cb: status_cb("Checking /robots.txt...")
        _, robots_txt = fetch_text(urljoin(origin, "/robots.txt"), timeout=timeout)
        sitemaps = re.findall(r"Sitemap:\s*(http.*)", robots_txt, re.IGNORECASE)
        
        found_via_robots = False
        for sm in sitemaps:
            candidates = fetch_sitemap_urls_recursive(sm.strip(), origin, timeout, status_cb)
            product_urls = pick_product_urls_from_candidates(candidates, origin, max_count=3)
            if product_urls:
                notes.append("✅ Found product URLs via robots.txt sitemap")
                found_via_robots = True
                return homepage_url, product_urls, notes
        
        if not found_via_robots:
             notes.append("⚠️ robots.txt had sitemaps, but no product URLs found")
    except Exception:
        notes.append("⚠️ /robots.txt not accessible")

    # Priority 3: Homepage scrape
    if status_cb: status_cb("Fail-safe: Scraping homepage links...")
    product_urls = scrape_homepage_for_product_links(home_html, origin, max_count=3)
    if product_urls:
        notes.append("✅ Found product URLs via homepage scrape")
    else:
        notes.append("❌ Could not auto-discover /products/ links")

    return homepage_url, product_urls, notes


# ----------------------------
# Page audit
# ----------------------------
def audit_page(url: str, timeout: int) -> PageAudit:
    warnings: List[str] = []
    try:
        final_url, html = fetch_text(url, timeout=timeout)
        ok_fetch = True
        fetch_error = None
    except Exception as e:
        return PageAudit(
            url=url, final_url=url, ok_fetch=False, fetch_error=str(e),
            org_present=False
