# app.py — Agentic Infrastructure Audit
# Requirements:
#   pip install streamlit requests beautifulsoup4 lxml streamlit-extras plotly
# Run:
#   streamlit run app.py

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from streamlit_extras.metric_cards import style_metric_cards

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Agentic Infrastructure Audit",
    page_icon="📈",
    layout="centered",
)

# ----------------------------
# Config
# ----------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 15

SHOPIFY_SITEMAP_PRODUCTS_PATH = "/sitemap_products_1.xml"
UNIVERSAL_SITEMAP_PATH = "/sitemap.xml"

PRODUCT_HINT_TOKENS = ("/products/", "/product/", "/shop/", "/store/", "/item/")
DISALLOWED_EXTS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".pdf", ".zip", ".rar", ".7z",
    ".mp4", ".mov", ".avi", ".webm",
    ".css", ".js", ".json", ".xml",
)

SOCIAL_DOMAINS = (
    "instagram.com",
    "facebook.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "linkedin.com",
)

AI_CRAWLERS = {
    "GPTBot": "Mozilla/5.0 (compatible; GPTBot/1.0; +https://openai.com/gptbot)",
    "Claude-Web": "Claude-Web/1.0",
    "CCBot": "CCBot/2.0",
    "Google-Extended": "Google-Extended",
    "PerplexityBot": "PerplexityBot/1.0",
}

AUTHORITY_DOMAINS = {
    "Wikipedia": "wikipedia.org",
    "Wikidata": "wikidata.org",
    "Crunchbase": "crunchbase.com",
    "LinkedIn": "linkedin.com",
    "YouTube": "youtube.com",
    "X": "x.com",
    "Twitter": "twitter.com",
    "GitHub": "github.com",
}

GHOST_TEXT_MIN = 600
INDUSTRY_AVERAGE_SCORE = 72


# ----------------------------
# Data models
# ----------------------------
@dataclass
class PageAudit:
    requested_url: str
    final_url: str
    ok_fetch: bool
    fetch_error: Optional[str]

    score: int

    org_found: bool
    identity_verified: bool
    faq_found: bool
    product_found: bool
    commerce_ready: bool

    ghost: bool
    text_len: int
    html_len: int
    semantic_density: float

    h1_text: str
    h1_has_brand: bool

    schema_types_found: Set[str]

    extractability_score: int
    extractability_verdict: str
    extractability_reasons: List[str]


@dataclass
class SiteAudit:
    origin: str
    brand: str
    homepage_url: str
    scan_urls: List[str]
    notes: List[str]
    pages: List[PageAudit]

    robots_accessible: bool
    robots_error: Optional[str]
    llms_txt_present: bool
    llms_txt_error: Optional[str]

    img_total: int
    img_missing_alt: int
    img_missing_alt_examples: List[str]

    org_present_any: bool
    sameas_social_links: List[str]

    ai_crawler_results: Dict[str, Dict[str, str]]

    authority_score: int
    authority_verdict: str
    authority_results: Dict[str, bool]
    authority_links: List[str]


# ----------------------------
# Gauge helper
# ----------------------------
def create_gauge(score: int, title: str) -> go.Figure:
    score = int(max(0, min(100, score)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 14}},
            number={"font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(0,0,0,0.25)"},
                "bar": {"color": "rgba(0,0,0,0.55)"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(231,76,60,0.75)"},
                    {"range": [40, 70], "color": "rgba(241,196,15,0.75)"},
                    {"range": [70, 100], "color": "rgba(46,204,113,0.75)"},
                ],
                "threshold": {
                    "line": {"color": "rgba(0,0,0,0.65)", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=240,
        font=dict(size=12),
    )
    return fig


# ----------------------------
# URL helpers
# ----------------------------
def ensure_scheme(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    return u


def origin_from_url(u: str) -> str:
    u = ensure_scheme(u)
    if not u:
        return ""
    p = urlparse(u)
    if not p.netloc:
        return ""
    return f"{p.scheme}://{p.netloc}"


def normalize_url(u: str) -> str:
    p = urlparse(u)
    return p._replace(fragment="").geturl()


def is_internal(u: str, origin: str) -> bool:
    try:
        return urlparse(u).netloc == urlparse(origin).netloc
    except Exception:
        return False


def is_disallowed_asset(u: str) -> bool:
    low = (u or "").lower()
    return any(low.endswith(ext) for ext in DISALLOWED_EXTS)


def looks_like_product_url(u: str, origin: str) -> bool:
    if not u:
        return False
    u = normalize_url(u)
    if not is_internal(u, origin):
        return False
    low = u.lower()
    if is_disallowed_asset(low):
        return False
    return any(tok in low for tok in PRODUCT_HINT_TOKENS)


def extract_filename_from_src(src: str) -> str:
    if not src:
        return "(no-src)"
    try:
        path = urlparse(src).path or src.split("?", 1)[0].split("#", 1)[0]
        name = path.rstrip("/").split("/")[-1]
        return name or src[:40]
    except Exception:
        return src[:40]


def infer_brand_name(origin: str) -> str:
    host = urlparse(origin).netloc.lower().replace("www.", "")
    label = host.split(":")[0].split(".")[0]
    label = re.sub(r"[^a-z0-9\-]+", "", label)
    parts = [p for p in label.split("-") if p]
    if not parts:
        return "Your Brand"
    return " ".join(p.capitalize() for p in parts)


def brand_in_h1(brand: str, h1_text: str) -> bool:
    if not brand or not h1_text:
        return False
    tokens = [t.lower() for t in re.split(r"\s+", brand.strip()) if t.strip()]
    h = h1_text.lower()
    return all(t in h for t in tokens)


# ----------------------------
# Networking
# ----------------------------
def fetch_text(url: str, timeout: int) -> Tuple[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return url, ""
        return r.url, r.text
    except requests.exceptions.RequestException:
        return url, ""


def safe_fetch_text(url: str, timeout: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        final_url, text = fetch_text(url, timeout)
        return final_url, text, None
    except Exception as e:
        return None, None, str(e)


def fetch_with_ua(url: str, user_agent: str, timeout: int) -> Dict[str, str]:
    """Fetch a URL with a specific User-Agent and return status info."""
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            text = soup.get_text(" ", strip=True)
            if len(text) < GHOST_TEXT_MIN:
                return {"status": "Thin Render", "code": str(r.status_code)}
            return {"status": "Accessible", "code": str(r.status_code)}
        elif r.status_code in (401, 403):
            return {"status": "Blocked", "code": str(r.status_code)}
        else:
            return {"status": f"HTTP {r.status_code}", "code": str(r.status_code)}
    except Exception as e:
        return {"status": "Error", "code": str(e)[:60]}


# ----------------------------
# Sitemap helpers
# ----------------------------
def extract_loc_urls_from_xml(xml_text: str) -> List[str]:
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml_text, flags=re.IGNORECASE)
    return [normalize_url(u.strip()) for u in locs if (u or "").strip()]


def crawl_sitemap_for_products(
    sitemap_url: str, origin: str, timeout: int, limit: int = 3
) -> Tuple[List[str], List[str]]:
    notes: List[str] = []
    found: List[str] = []
    seen: Set[str] = set()

    _, xml, err = safe_fetch_text(sitemap_url, timeout)
    if err or not xml:
        notes.append(f"⚠️ Sitemap fetch failed: {sitemap_url} ({err})")
        return [], notes

    locs = extract_loc_urls_from_xml(xml)
    for u in locs:
        if len(found) >= limit:
            break
        if u in seen:
            continue
        seen.add(u)
        if looks_like_product_url(u, origin):
            found.append(u)

    if found:
        notes.append(f"✅ Sitemap discovered {len(found)} product URL(s)")
    else:
        notes.append("⚠️ Sitemap returned 0 product URLs")
    return found, notes


def discover_from_homepage_scrape(origin: str, home_html: str, limit: int = 3) -> List[str]:
    soup = BeautifulSoup(home_html, "lxml")
    candidates: List[str] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_u = normalize_url(urljoin(origin, href))
        if looks_like_product_url(abs_u, origin):
            candidates.append(abs_u)

    out: List[str] = []
    seen: Set[str] = set()
    for u in candidates:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= limit:
            break
    return out


def discover_home_and_products(
    origin: str, timeout: int
) -> Tuple[str, List[str], List[str]]:
    notes: List[str] = []
    home_final, home_html = fetch_text(urljoin(origin, "/"), timeout=timeout)
    homepage_url = normalize_url(home_final)

    turbo_url = urljoin(origin, SHOPIFY_SITEMAP_PRODUCTS_PATH)
    turbo_found, turbo_notes = crawl_sitemap_for_products(turbo_url, origin, timeout, limit=3)
    notes.extend([f"Shopify Turbo: {n}" for n in turbo_notes])
    if turbo_found:
        return homepage_url, turbo_found, notes

    sm_url = urljoin(origin, UNIVERSAL_SITEMAP_PATH)
    sm_found, sm_notes = crawl_sitemap_for_products(sm_url, origin, timeout, limit=3)
    notes.extend([f"Universal Sitemap: {n}" for n in sm_notes])
    if sm_found:
        return homepage_url, sm_found, notes

    _, robots_text, robots_err = safe_fetch_text(urljoin(origin, "/robots.txt"), timeout)
    if robots_text:
        sitemap_lines = []
        for line in robots_text.splitlines():
            if line.strip().lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                if sm:
                    sitemap_lines.append(sm)
        if sitemap_lines:
            notes.append(f"Robots: Found {len(sitemap_lines)} sitemap declaration(s)")
            for sm in sitemap_lines:
                found, smn = crawl_sitemap_for_products(sm, origin, timeout, limit=3)
                notes.extend([f"Robots Sitemap: {n}" for n in smn])
                if found:
                    return homepage_url, found, notes
        else:
            notes.append("Robots: No sitemap declarations found")
    else:
        notes.append(f"Robots: not accessible ({robots_err})")

    scrape_found = discover_from_homepage_scrape(origin, home_html, limit=3)
    if scrape_found:
        notes.append(f"Homepage Scrape: Found {len(scrape_found)} product-like URL(s)")
    else:
        notes.append("Homepage Scrape: No product-like URLs found")

    return homepage_url, scrape_found, notes


# ----------------------------
# JSON-LD helpers
# ----------------------------
def try_parse_json(raw: str) -> Optional[Any]:
    try:
        return json.loads(raw)
    except Exception:
        pass
    no_comments = re.sub(
        r"//.*?$|/\*.*?\*/", "", raw, flags=re.MULTILINE | re.DOTALL
    ).strip()
    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", no_comments)
    try:
        return json.loads(no_trailing_commas)
    except Exception:
        return None


def extract_jsonld_payloads(html: str) -> List[Any]:
    soup = BeautifulSoup(html, "lxml")
    scripts = soup.find_all(
        "script", attrs={"type": re.compile(r"application/ld\+json", re.I)}
    )
    payloads: List[Any] = []
    for s in scripts:
        raw = (s.get_text(strip=False) or "").strip()
        if not raw:
            continue
        parsed = try_parse_json(raw)
        if parsed is not None:
            payloads.append(parsed)
    return payloads


def iter_json_objects(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from iter_json_objects(v)
    elif isinstance(node, list):
        for item in node:
            yield from iter_json_objects(item)


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
        if "@type" in t:
            out.extend(normalize_schema_type(t.get("@type")))
    return [x for x in out if x]


def flatten_jsonld_objects(payloads: List[Any]) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    roots = payloads if isinstance(payloads, list) else [payloads]
    for payload in roots:
        for obj in iter_json_objects(payload):
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


def find_first(objs: List[Dict[str, Any]], target: str) -> Optional[Dict[str, Any]]:
    for o in objs:
        if has_type(o, target):
            return o
    return None


def find_all(objs: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    return [o for o in objs if has_type(o, target)]


# ----------------------------
# Scoring rules
# ----------------------------
def identity_ok(org_obj: Dict[str, Any]) -> bool:
    dis = org_obj.get("disambiguatingDescription")
    if isinstance(dis, str) and dis.strip():
        return True
    same = org_obj.get("sameAs")
    if isinstance(same, str) and same.strip():
        return True
    if isinstance(same, list) and any(
        isinstance(x, str) and x.strip() for x in same
    ):
        return True
    return False


def offers_has_price(offers: Any) -> bool:
    def offer_dict_has_price(d: Dict[str, Any]) -> bool:
        if "price" in d and str(d.get("price", "")).strip():
            return True
        ps = d.get("priceSpecification")
        if isinstance(ps, dict) and "price" in ps and str(ps.get("price", "")).strip():
            return True
        if isinstance(ps, list):
            for item in ps:
                if (
                    isinstance(item, dict)
                    and "price" in item
                    and str(item.get("price", "")).strip()
                ):
                    return True
        return False

    if isinstance(offers, dict):
        return offer_dict_has_price(offers)
    if isinstance(offers, list):
        return any(isinstance(o, dict) and offer_dict_has_price(o) for o in offers)
    return False


def commerce_ok(product_obj: Dict[str, Any]) -> bool:
    offers = product_obj.get("offers")
    if offers is None:
        return False
    return offers_has_price(offers)


def compute_score(
    org_found: bool,
    id_verified: bool,
    faq_found: bool,
    prod_found: bool,
    comm_ready: bool,
    ghost: bool,
) -> int:
    if ghost:
        return 0
    score = 0
    if org_found:
        score += 10
        if id_verified:
            score += 20
    if faq_found:
        score += 20
    if prod_found:
        score += 20
        if comm_ready:
            score += 30
    return score


# ----------------------------
# Deep tech checks
# ----------------------------
def check_robots(
    origin: str, timeout: int
) -> Tuple[bool, Optional[str], Optional[str]]:
    _, txt, err = safe_fetch_text(urljoin(origin, "/robots.txt"), timeout)
    if err or txt is None or txt == "":
        return False, None, err
    return True, txt, None


def check_llms_txt(origin: str, timeout: int) -> Tuple[bool, Optional[str]]:
    _, txt, err = safe_fetch_text(urljoin(origin, "/llms.txt"), timeout)
    if err or txt is None or txt == "":
        return False, err or "No response."
    if len(txt.strip()) > 10 and "<html" not in txt.lower():
        return True, None
    return False, "llms.txt exists but appears empty or non-text."


def calc_semantic_density(text_len: int, html_len: int) -> float:
    if html_len <= 0:
        return 0.0
    return (float(text_len) / float(html_len)) * 100.0


def extract_h1_text(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    return (h1.get_text(" ", strip=True) if h1 else "").strip()


def schema_types_set(objs: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for o in objs:
        types = normalize_schema_type(o.get("@type"))
        for x in types:
            out.add(x)
    return out


def extract_social_sameas(org_obj: Dict[str, Any]) -> List[str]:
    same_as = org_obj.get("sameAs")
    links: List[str] = []

    if isinstance(same_as, str) and same_as.strip():
        links = [same_as.strip()]
    elif isinstance(same_as, list):
        links = [x.strip() for x in same_as if isinstance(x, str) and x.strip()]

    socials: List[str] = []
    for link in links:
        h = urlparse(link).netloc.lower().replace("www.", "")
        if any(dom in h for dom in SOCIAL_DOMAINS):
            socials.append(link)

    seen: Set[str] = set()
    out: List[str] = []
    for s in socials:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# ----------------------------
# AI Extractability Score
# ----------------------------
def compute_extractability(
    html: str, soup: BeautifulSoup, text: str
) -> Tuple[int, str, List[str]]:
    reasons: List[str] = []
    score = 0

    # Readable text depth
    text_len = len(text.strip())
    if text_len >= 1000:
        score += 20
        reasons.append("✅ Rich readable text (1000+ chars)")
    elif text_len >= 400:
        score += 10
        reasons.append("⚠️ Moderate text length")
    else:
        reasons.append("❌ Thin text content")

    # Title tag
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        score += 10
        reasons.append("✅ Title tag present")
    else:
        reasons.append("❌ Missing title tag")

    # H1
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        score += 10
        reasons.append("✅ H1 tag present")
    else:
        reasons.append("❌ Missing H1 tag")

    # FAQ patterns
    faq_keywords = ["faq", "frequently asked", "question", "answer"]
    lower_text = text.lower()
    if any(kw in lower_text for kw in faq_keywords):
        score += 15
        reasons.append("✅ FAQ-style content detected")
    else:
        reasons.append("⚠️ No FAQ patterns found")

    # Policy / about content
    policy_keywords = ["privacy", "terms", "about us", "return policy", "shipping"]
    if any(kw in lower_text for kw in policy_keywords):
        score += 10
        reasons.append("✅ Policy / trust content detected")
    else:
        reasons.append("⚠️ No policy content found")

    # Product signals
    product_keywords = ["price", "buy", "add to cart", "checkout", "shop", "order"]
    if any(kw in lower_text for kw in product_keywords):
        score += 15
        reasons.append("✅ Product / commerce signals found")
    else:
        reasons.append("⚠️ No product signals detected")

    # Content depth (paragraphs)
    paragraphs = soup.find_all("p")
    if len(paragraphs) >= 5:
        score += 10
        reasons.append(f"✅ {len(paragraphs)} paragraph tags found")
    else:
        reasons.append(f"⚠️ Only {len(paragraphs)} paragraph tag(s)")

    # Structured headings
    headings = soup.find_all(["h2", "h3"])
    if len(headings) >= 3:
        score += 10
        reasons.append(f"✅ {len(headings)} sub-headings found")
    else:
        reasons.append(f"⚠️ Only {len(headings)} sub-heading(s)")

    score = min(100, score)

    if score >= 70:
        verdict = "High AI Extractability"
    elif score >= 40:
        verdict = "Moderate AI Extractability"
    else:
        verdict = "Low AI Extractability"

    return score, verdict, reasons


# ----------------------------
# Knowledge Graph Authority
# ----------------------------
def compute_authority(
    objs: List[Dict[str, Any]]
) -> Tuple[int, str, Dict[str, bool], List[str]]:
    authority_results: Dict[str, bool] = {k: False for k in AUTHORITY_DOMAINS}
    authority_links: List[str] = []

    for obj in objs:
        same_as = obj.get("sameAs")
        links: List[str] = []
        if isinstance(same_as, str) and same_as.strip():
            links = [same_as.strip()]
        elif isinstance(same_as, list):
            links = [x.strip() for x in same_as if isinstance(x, str) and x.strip()]

        for link in links:
            netloc = urlparse(link).netloc.lower().replace("www.", "")
            for name, domain in AUTHORITY_DOMAINS.items():
                if domain in netloc:
                    authority_results[name] = True
                    if link not in authority_links:
                        authority_links.append(link)

    hits = sum(1 for v in authority_results.values() if v)
    total = len(AUTHORITY_DOMAINS)
    score = int((hits / total) * 100)

    if score >= 60:
        verdict = "Strong Knowledge Graph Presence"
    elif score >= 30:
        verdict = "Moderate Knowledge Graph Presence"
    else:
        verdict = "Weak Knowledge Graph Presence"

    return score, verdict, authority_results, authority_links


# ----------------------------
# Image Alt Scanner
# ----------------------------
def scan_images(soup: BeautifulSoup) -> Tuple[int, int, List[str]]:
    imgs = soup.find_all("img")
    total = len(imgs)
    missing: List[str] = []
    for img in imgs:
        alt = img.get("alt")
        if alt is None or (isinstance(alt, str) and not alt.strip()):
            src = img.get("src", "")
            missing.append(extract_filename_from_src(src))
    return total, len(missing), missing[:5]


# ----------------------------
# AI Crawler Simulation
# ----------------------------
def simulate_ai_crawlers(
    url: str, timeout: int
) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    for crawler_name, ua in AI_CRAWLERS.items():
        results[crawler_name] = fetch_with_ua(url, ua, timeout)
        time.sleep(0.3)
    return results


# ----------------------------
# Page audit
# ----------------------------
def audit_page(url: str, brand: str, timeout: int) -> PageAudit:
    final_url, html, err = safe_fetch_text(url, timeout)

    if err or not html:
        return PageAudit(
            requested_url=url,
            final_url=url,
            ok_fetch=False,
            fetch_error=err or "Empty response",
            score=0,
            org_found=False,
            identity_verified=False,
            faq_found=False,
            product_found=False,
            commerce_ready=False,
            ghost=True,
            text_len=0,
            html_len=0,
            semantic_density=0.0,
            h1_text="",
            h1_has_brand=False,
            schema_types_found=set(),
            extractability_score=0,
            extractability_verdict="Low AI Extractability",
            extractability_reasons=["❌ Page could not be fetched"],
        )

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    text_len = len(text)
    html_len = len(html)
    ghost = text_len < GHOST_TEXT_MIN

    payloads = extract_jsonld_payloads(html)
    objs = flatten_jsonld_objects(payloads)

    org_obj = find_first(objs, "organization") or find_first(objs, "localbusiness")
    org_found = org_obj is not None
    id_verified = identity_ok(org_obj) if org_obj else False

    faq_obj = find_first(objs, "faqpage")
    faq_found = faq_obj is not None

    prod_objs = find_all(objs, "product")
    product_found = len(prod_objs) > 0
    commerce_ready = any(commerce_ok(p) for p in prod_objs) if prod_objs else False

    h1_text = extract_h1_text(soup)
    h1_has_brand = brand_in_h1(brand, h1_text)

    types_found = schema_types_set(objs)
    density = calc_semantic_density(text_len, html_len)

    score = compute_score(
        org_found, id_verified, faq_found, product_found, commerce_ready, ghost
    )

    ext_score, ext_verdict, ext_reasons = compute_extractability(html, soup, text)

    return PageAudit(
        requested_url=url,
        final_url=final_url or url,
        ok_fetch=True,
        fetch_error=None,
        score=score,
        org_found=org_found,
        identity_verified=id_verified,
        faq_found=faq_found,
        product_found=product_found,
        commerce_ready=commerce_ready,
        ghost=ghost,
        text_len=text_len,
        html_len=html_len,
        semantic_density=density,
        h1_text=h1_text,
        h1_has_brand=h1_has_brand,
        schema_types_found=types_found,
        extractability_score=ext_score,
        extractability_verdict=ext_verdict,
        extractability_reasons=ext_reasons,
    )


# ----------------------------
# Full site audit
# ----------------------------
def run_site_audit(raw_url: str, timeout: int = DEFAULT_TIMEOUT) -> SiteAudit:
    origin = origin_from_url(raw_url)
    brand = infer_brand_name(origin)

    homepage_url, product_urls, discovery_notes = discover_home_and_products(
        origin, timeout
    )
    scan_urls = [homepage_url] + product_urls[:3]

    pages: List[PageAudit] = []
    for u in scan_urls:
        pages.append(audit_page(u, brand, timeout))

    robots_ok, robots_txt, robots_err = check_robots(origin, timeout)
    llms_ok, llms_err = check_llms_txt(origin, timeout)

    # Aggregate images from homepage
    _, home_html = fetch_text(homepage_url, timeout)
    home_soup = BeautifulSoup(home_html, "lxml") if home_html else BeautifulSoup("", "lxml")
    img_total, img_missing_alt, img_missing_alt_examples = scan_images(home_soup)

    # Org / sameAs
    all_objs: List[Dict[str, Any]] = []
    for p in pages:
        if p.ok_fetch:
            _, html, _ = safe_fetch_text(p.final_url, timeout)
            if html:
                payloads = extract_jsonld_payloads(html)
                all_objs.extend(flatten_jsonld_objects(payloads))

    org_present_any = any(
        has_type(o, "organization") or has_type(o, "localbusiness")
        for o in all_objs
    )

    sameas_social_links: List[str] = []
    for o in all_objs:
        if has_type(o, "organization") or has_type(o, "localbusiness"):
            sameas_social_links.extend(extract_social_sameas(o))

    seen: Set[str] = set()
    unique_social: List[str] = []
    for s in sameas_social_links:
        if s not in seen:
            seen.add(s)
            unique_social.append(s)
    sameas_social_links = unique_social

    # AI crawlers (test homepage)
    ai_crawler_results = simulate_ai_crawlers(homepage_url, timeout)

    # Authority
    authority_score, authority_verdict, authority_results, authority_links = compute_authority(
        all_objs
    )

    return SiteAudit(
        origin=origin,
        brand=brand,
        homepage_url=homepage_url,
        scan_urls=scan_urls,
        notes=discovery_notes,
        pages=pages,
        robots_accessible=robots_ok,
        robots_error=robots_err,
        llms_txt_present=llms_ok,
        llms_txt_error=llms_err,
        img_total=img_total,
        img_missing_alt=img_missing_alt,
        img_missing_alt_examples=img_missing_alt_examples,
        org_present_any=org_present_any,
        sameas_social_links=sameas_social_links,
        ai_crawler_results=ai_crawler_results,
        authority_score=authority_score,
        authority_verdict=authority_verdict,
        authority_results=authority_results,
        authority_links=authority_links,
    )


# ----------------------------
# Score helpers for display
# ----------------------------
def best_page_score(audit: SiteAudit) -> int:
    if not audit.pages:
        return 0
    return max(p.score for p in audit.pages)


def homepage_page(audit: SiteAudit) -> Optional[PageAudit]:
    if not audit.pages:
        return None
    return audit.pages[0]


# ----------------------------
# UI rendering helpers
# ----------------------------
def render_check(label: str, ok: bool, detail: str = "") -> None:
    icon = "✅" if ok else "❌"
    if detail:
        st.markdown(f"{icon} **{label}** — {detail}")
    else:
        st.markdown(f"{icon} **{label}**")


def density_label(d: float) -> str:
    if d < 5.0:
        return f"{d:.1f}% — Bloated code"
    if d < 10.0:
        return f"{d:.1f}% — Weak semantic density"
    return f"{d:.1f}% — Good"


# ----------------------------
# JSON-LD strategy snippet
# ----------------------------
def build_org_jsonld(brand: str, origin: str) -> str:
    return json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "Organization",
            "name": brand,
            "url": origin,
            "disambiguatingDescription": f"{brand} — describe your brand in one sentence here.",
            "sameAs": [
                "https://www.linkedin.com/company/your-company",
                "https://www.instagram.com/your-handle",
                "https://en.wikipedia.org/wiki/Your_Page",
            ],
        },
        indent=2,
    )


# ----------------------------
# Streamlit App
# ----------------------------
def main() -> None:
    st.title("📈 Agentic Infrastructure Audit")
    st.caption(
        "Audit your website for AI search readiness — ChatGPT, Perplexity, Claude, Google AI Overviews."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        target_input = st.text_input(
            "🎯 Your URL", placeholder="https://yoursite.com", key="target"
        )
    with col2:
        competitor_input = st.text_input(
            "🏁 Competitor URL (optional)", placeholder="https://competitor.com", key="competitor"
        )

    run_btn = st.button("🚀 Run Audit", use_container_width=True, type="primary")

    if not run_btn:
        st.info("Enter your URL above and click **Run Audit** to begin.")
        return

    target_url = ensure_scheme(target_input.strip())
    if not target_url:
        st.error("Please enter a valid URL.")
        return

    competitor_url = ensure_scheme(competitor_input.strip()) if competitor_input.strip() else None

    # ----------------------------
    # Run audits
    # ----------------------------
    with st.spinner("Crawling your site…"):
        target_audit = run_site_audit(target_url)

    competitor_audit: Optional[SiteAudit] = None
    if competitor_url:
        with st.spinner("Crawling competitor site…"):
            competitor_audit = run_site_audit(competitor_url)

    target_score = best_page_score(target_audit)
    competitor_score = best_page_score(competitor_audit) if competitor_audit else None

    # ----------------------------
    # SECTION 1 — Scorecard
    # ----------------------------
    st.markdown("## 🏆 Agentic Score")

    if competitor_audit:
        gc1, gc2 = st.columns(2)
        with gc1:
            st.plotly_chart(
                create_gauge(target_score, f"Your Score\n{target_audit.brand}"),
                use_container_width=True,
            )
        with gc2:
            st.plotly_chart(
                create_gauge(competitor_score, f"Competitor\n{competitor_audit.brand}"),
                use_container_width=True,
            )
    else:
        g_col, _ = st.columns([1, 1])
        with g_col:
            st.plotly_chart(
                create_gauge(target_score, f"Your Score\n{target_audit.brand}"),
                use_container_width=True,
            )

    delta = target_score - INDUSTRY_AVERAGE_SCORE
    delta_str = f"+{delta}" if delta >= 0 else str(delta)
    st.metric(
        label="vs Industry Average",
        value=f"{target_score}/100",
        delta=f"{delta_str} pts vs avg ({INDUSTRY_AVERAGE_SCORE})",
    )
    style_metric_cards()

    # Score breakdown table
    st.markdown("#### Score Breakdown")
    hp = homepage_page(target_audit)
    if hp:
        breakdown_data = {
            "Signal": [
                "Organization Schema",
                "Identity Verification",
                "FAQPage Schema",
                "Product Schema",
                "Offers / Price",
            ],
            "Max Pts": [10, 20, 20, 20, 30],
            "Earned": [
                10 if hp.org_found else 0,
                20 if hp.identity_verified else 0,
                20 if hp.faq_found else 0,
                20 if hp.product_found else 0,
                30 if hp.commerce_ready else 0,
            ],
        }
        rows = ""
        for i in range(len(breakdown_data["Signal"])):
            earned = breakdown_data["Earned"][i]
            mx = breakdown_data["Max Pts"][i]
            icon = "✅" if earned > 0 else "❌"
            rows += f"| {icon} {breakdown_data['Signal'][i]} | {mx} | **{earned}** |\n"
        st.markdown(
            f"| Signal | Max | Earned |\n|---|---|---|\n{rows}"
        )

    st.markdown("---")

    # ----------------------------
    # SECTION 2 — Insight Grid
    # ----------------------------
    st.markdown("## 🔍 Insight Grid")

    ig1, ig2 = st.columns(2)
    ig3, ig4 = st.columns(2)

    # AI Access card
    with ig1:
        st.markdown("**🤖 AI Access**")
        if not target_audit.robots_accessible:
            st.error(f"robots.txt: Not accessible")
        else:
            st.success("robots.txt: Accessible")
        if not target_audit.llms_txt_present:
            st.warning(
                f"llms.txt: Missing — HIGH IMPACT. AI agents cannot find your content manifest."
            )
        else:
            st.success("llms.txt: Present")

    # Visual Semantics card
    with ig2:
        st.markdown("**🖼️ Visual Semantics**")
        if target_audit.img_total == 0:
            st.warning("No images found on homepage.")
        elif target_audit.img_missing_alt == 0:
            st.success(f"All {target_audit.img_total} images have alt text.")
        else:
            pct = int((target_audit.img_missing_alt / target_audit.img_total) * 100)
            st.warning(
                f"{target_audit.img_missing_alt}/{target_audit.img_total} images missing alt ({pct}%)"
            )
        if target_audit.img_missing_alt_examples:
            st.caption("Examples: " + ", ".join(target_audit.img_missing_alt_examples))

    # Semantic Density card
    with ig3:
        st.markdown("**📊 Semantic Density**")
        if hp:
            d = hp.semantic_density
            label = density_label(d)
            if d < 5.0:
                st.error(f"Density: {label}")
            elif d < 10.0:
                st.warning(f"Density: {label}")
            else:
                st.success(f"Density: {label}")
            if hp.ghost:
                st.error("⚠️ Ghost Code Detected — JS render blocking. Score forced to 0.")
        else:
            st.warning("No homepage data.")

    # Trust & Entity card
    with ig4:
        st.markdown("**🏛️ Trust & Entity**")
        if target_audit.org_present_any:
            st.success("Organization schema: Found")
        else:
            st.error("Organization schema: Missing")
        if target_audit.sameas_social_links:
            st.success(f"sameAs social links: {len(target_audit.sameas_social_links)} found")
        else:
            st.warning("sameAs social links: None found")

    st.markdown("---")

    # ----------------------------
    # SECTION 3 — AI Crawl Simulation
    # ----------------------------
    st.markdown("## 🕷️ AI Crawler Simulation")
    st.caption(f"Testing access for: `{target_audit.homepage_url}`")

    crawler_cols = st.columns(len(AI_CRAWLERS))
    for idx, (crawler_name, result) in enumerate(
        target_audit.ai_crawler_results.items()
    ):
        with crawler_cols[idx]:
            status = result.get("status", "Unknown")
            code = result.get("code", "")
            if status == "Accessible":
                st.success(f"**{crawler_name}**\n\n{status}")
            elif status == "Blocked":
                st.error(f"**{crawler_name}**\n\n{status} ({code})")
            elif status == "Thin Render":
                st.warning(f"**{crawler_name}**\n\n{status}")
            else:
                st.warning(f"**{crawler_name}**\n\n{status}")

    st.markdown("---")

    # ----------------------------
    # SECTION 4 — Knowledge Graph Authority
    # ----------------------------
    st.markdown("## 🌐 Knowledge Graph Authority")

    auth_g_col, auth_info_col = st.columns([1, 2])
    with auth_g_col:
        st.plotly_chart(
            create_gauge(target_audit.authority_score, "Authority Score"),
            use_container_width=True,
        )
    with auth_info_col:
        st.markdown(f"**Verdict:** {target_audit.authority_verdict}")
        st.markdown("**Authority Node Detection:**")
        for node_name, found in target_audit.authority_results.items():
            icon = "✅" if found else "❌"
            st.markdown(f"{icon} {node_name}")
        if target_audit.authority_links:
            st.markdown("**Detected sameAs links:**")
            for link in target_audit.authority_links[:5]:
                st.markdown(f"- [{link}]({link})")

    st.markdown("---")

    # ----------------------------
    # SECTION 5 — Detailed Page Analysis
    # ----------------------------
    st.markdown("## 📄 Detailed Page Analysis")

    for i, page in enumerate(target_audit.pages):
        label = "🏠 Homepage" if i == 0 else f"📦 Product Page {i}"
        with st.expander(f"{label} — `{page.requested_url}`", expanded=(i == 0)):
            if not page.ok_fetch:
                st.error(f"Fetch failed: {page.fetch_error}")
                continue

            pc1, pc2 = st.columns(2)
            with pc1:
                st.metric("AEO Score", f"{page.score}/100")
                render_check("Ghost Code", not page.ghost, "Clean render" if not page.ghost else "JS render blocking")
                render_check("H1 Present", bool(page.h1_text), page.h1_text[:80] if page.h1_text else "Missing")
                render_check("Brand in H1", page.h1_has_brand)
                render_check("Organization Schema", page.org_found)
                render_check("Identity Verified", page.identity_verified)
                render_check("FAQPage Schema", page.faq_found)
                render_check("Product Schema", page.product_found)
                render_check("Offers / Price", page.commerce_ready)

            with pc2:
                st.markdown(f"**Semantic Density:** {density_label(page.semantic_density)}")
                st.markdown(
                    f"**Text:** {page.text_len:,} chars / **HTML:** {page.html_len:,} chars"
                )
                if page.schema_types_found:
                    types_display = ", ".join(sorted(page.schema_types_found)[:10])
                    st.markdown(f"**Schema Types:** `{types_display}`")
                else:
                    st.markdown("**Schema Types:** None found")

                st.markdown(
                    f"**AI Extractability:** {page.extractability_score}/100 — *{page.extractability_verdict}*"
                )
                for reason in page.extractability_reasons:
                    st.markdown(f"  {reason}")

    st.markdown("---")

    # ----------------------------
    # SECTION 6 — Strategy Section
    # ----------------------------
    st.markdown("## 🗺️ AI Readiness Strategy")

    st.markdown("### Phase 1 — Fix Entity Identity")
    st.markdown(
        "Your Organization schema is missing or incomplete. "
        "Add this JSON-LD block to your `<head>` on every page:"
    )
    st.code(
        build_org_jsonld(target_audit.brand, target_audit.origin),
        language="json",
    )

    if not target_audit.llms_txt_present:
        st.warning(
            "⚠️ **llms.txt is missing.** Create `/llms.txt` at your domain root to "
            "help AI agents discover and index your content. "
            "See [llmstxt.org](https://llmstxt.org) for the spec."
        )

    if target_audit.img_missing_alt > 0:
        st.warning(
            f"⚠️ **{target_audit.img_missing_alt} images** are missing alt text. "
            "Alt text is a primary signal for multimodal AI understanding."
        )

    st.markdown("### Phase 2 — Competitor Displacement Strategy")

    if competitor_audit:
        gap = target_score - competitor_score
        if gap >= 0:
            st.success(
                f"✅ You are ahead of **{competitor_audit.brand}** by **{gap} points**. "
                "Focus on maintaining your lead with richer FAQ and Product schema."
            )
        else:
            st.error(
                f"❌ **{competitor_audit.brand}** leads you by **{abs(gap)} points**. "
                "Close the gap by implementing the missing schema signals above."
            )

        cc1, cc2 = st.columns(2)
        comp_hp = homepage_page(competitor_audit)
        with cc1:
            st.markdown(f"**Your signals ({target_audit.brand})**")
            if hp:
                render_check("Organization Schema", hp.org_found)
                render_check("Identity Verified", hp.identity_verified)
                render_check("FAQPage Schema", hp.faq_found)
                render_check("Product Schema", hp.product_found)
                render_check("Offers / Price", hp.commerce_ready)
        with cc2:
            st.markdown(f"**Competitor signals ({competitor_audit.brand})**")
            if comp_hp:
                render_check("Organization Schema", comp_hp.org_found)
                render_check("Identity Verified", comp_hp.identity_verified)
                render_check("FAQPage Schema", comp_hp.faq_found)
                render_check("Product Schema", comp_hp.product_found)
                render_check("Offers / Price", comp_hp.commerce_ready)
    else:
        st.info(
            "Add a competitor URL above to unlock competitor displacement analysis."
        )

    st.markdown("---")
    st.markdown("### 📞 Ready to Accelerate?")
    st.markdown(
        "Get expert help implementing AI-ready schema, llms.txt, and "
        "entity optimization for your brand."
    )
    st.link_button(
        "📅 Book a Strategy Call",
        url="https://calendly.com/your-link",
        use_container_width=True,
    )

    # Discovery notes (collapsed)
    with st.expander("🔧 Crawler Discovery Notes", expanded=False):
        for note in target_audit.notes:
            st.caption(note)


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main()
