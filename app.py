# app.py — Agentic Infrastructure Audit (WordLift-style Deep Tech + Insight Card Grid)
# Dependencies: streamlit, requests, beautifulsoup4, urllib.parse, re, json, time, datetime
#
# Install:
#   pip install streamlit requests beautifulsoup4 lxml
# Run:
#   streamlit run app.py

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Agentic Infrastructure Audit", page_icon="🧠", layout="centered")


# ----------------------------
# CONFIG
# ----------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
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

# Extended robots checks
AI_BOTS_EXTENDED = ("GPTBot", "CCBot", "Google-Extended", "anthropic-ai", "Claude-Web")

AUTH_TIER1_DOMAINS = (
    "wikipedia.org",
    "wikidata.org",
    "crunchbase.com",
)

INDUSTRY_AVERAGE_SCORE = 72


# ----------------------------
# DATA
# ----------------------------
@dataclass
class PageAudit:
    requested_url: str
    final_url: str
    ok_fetch: bool
    fetch_error: Optional[str]

    org_found: bool
    identity_verified: bool
    faq_found: bool
    product_found: bool
    commerce_ready: bool

    score: int
    warnings: List[str]

    raw_kb: float
    text_len: int
    ghost: bool

    # page-level semantic checks (best-effort)
    h1_present: bool
    h1_has_brand: bool

    # deep tech metrics
    semantic_density_pct: float
    h2_count: int
    h3_count: int
    img_count: int
    img_missing_alt_count: int
    img_missing_alt_examples: List[str]


@dataclass
class SiteDeepTech:
    llms_txt_accessible: bool
    llms_txt_error: Optional[str]

    robots_access: bool
    robots_error: Optional[str]
    any_ai_blocked: bool
    per_bot_blocked: Dict[str, bool]

    sitemap_found_products: bool
    sitemap_notes: List[str]

    title_text: str
    h1_text: str
    recency_pass: bool
    entity_pass_home: bool

    authority_pass_any: bool
    org_seen_any: bool

    ghost_driver: bool
    trust_driver_fail: bool

    # aggregated deep tech
    avg_semantic_density_pct: float
    total_imgs: int
    total_missing_alt: int
    top_missing_alt_filenames: List[str]

    # heading structure (homepage)
    home_h2_count: int
    home_h3_count: int

    # scanned pages metadata
    pages_scanned: int


@dataclass
class SiteAuditResult:
    origin: str
    brand: str
    homepage_url: str
    scan_urls: List[str]
    audits: List[PageAudit]
    deep: SiteDeepTech

    org_present_any: bool
    product_present_any: bool
    faq_present_any: bool
    commerce_ready_any: bool

    health_score: int
    leakage_pct: int


# ----------------------------
# URL HELPERS
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
    low = u.lower()
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


def looks_like_sitemap_url(u: str, origin: str) -> bool:
    if not u:
        return False
    u = normalize_url(u)
    if not is_internal(u, origin):
        return False
    return u.lower().endswith(".xml")


def domain_host(origin: str) -> str:
    return urlparse(origin).netloc.lower().replace("www.", "")


def host_of_url(u: str) -> str:
    try:
        return urlparse(u).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def extract_filename_from_src(src: str) -> str:
    if not src:
        return ""
    # Remove query strings
    try:
        path = urlparse(src).path
        if not path:
            path = src.split("?", 1)[0].split("#", 1)[0]
        name = path.rstrip("/").split("/")[-1]
        return name or src[:40]
    except Exception:
        return src[:40]


# ----------------------------
# NETWORKING
# ----------------------------
def fetch_text(url: str, timeout: int) -> Tuple[str, str]:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text


# ----------------------------
# ROBOTS / AI BLOCKING (Extended)
# ----------------------------
def parse_robots_for_blocks(robots_text: str, target_agents: Tuple[str, ...]) -> Dict[str, bool]:
    blocked = {a: False for a in target_agents}
    current_agents: List[str] = []
    saw_rule_in_group = False

    def new_group():
        nonlocal current_agents, saw_rule_in_group
        current_agents = []
        saw_rule_in_group = False

    for raw_line in robots_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

        m = re.match(r"(?i)user-agent\s*:\s*(.+)$", line)
        if m:
            agent = m.group(1).strip()
            if saw_rule_in_group and current_agents:
                new_group()
            current_agents.append(agent)
            continue

        m = re.match(r"(?i)disallow\s*:\s*(.*)$", line)
        if m:
            path = (m.group(1) or "").strip()
            saw_rule_in_group = True
            if path == "":
                continue
            blocks_all = path == "/" or path == "/*"
            if not blocks_all:
                continue
            for ga in current_agents:
                for ta in target_agents:
                    if ga.lower() == ta.lower():
                        blocked[ta] = True
            continue

    return blocked


def discover_sitemaps_from_robots(robots_text: str) -> List[str]:
    sitemaps: List[str] = []
    for line in robots_text.splitlines():
        if line.strip().lower().startswith("sitemap:"):
            sm = line.split(":", 1)[1].strip()
            if sm:
                sitemaps.append(sm)
    return sitemaps


def fetch_robots(origin: str, timeout: int) -> Tuple[Optional[str], Optional[str]]:
    try:
        _, robots = fetch_text(urljoin(origin, "/robots.txt"), timeout=timeout)
        return robots, None
    except Exception as e:
        return None, str(e)


# ----------------------------
# llms.txt detection
# ----------------------------
def fetch_llms_txt(origin: str, timeout: int) -> Tuple[bool, Optional[str]]:
    try:
        _, txt = fetch_text(urljoin(origin, "/llms.txt"), timeout=timeout)
        # Consider accessible if it returns content and isn't a HTML error page masquerading
        if txt and len(txt.strip()) > 10 and "<html" not in txt.lower():
            return True, None
        # If it exists but empty, treat as missing (high impact)
        return False, "llms.txt returned empty or non-text content."
    except Exception as e:
        return False, str(e)


# ----------------------------
# SITEMAP (RECURSIVE)
# ----------------------------
def extract_loc_urls_from_xml(xml_text: str) -> List[str]:
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml_text, flags=re.IGNORECASE)
    out: List[str] = []
    for u in locs:
        u = (u or "").strip()
        if u:
            out.append(normalize_url(u))
    return out


def crawl_sitemaps_for_products(
    starting_sitemap_url: str,
    origin: str,
    timeout: int,
    max_product_urls: int = 3,
    max_sitemaps_to_visit: int = 60,
) -> Tuple[List[str], List[str]]:
    notes: List[str] = []
    visited: Set[str] = set()
    queue: List[str] = [normalize_url(starting_sitemap_url)]
    found_products: List[str] = []
    found_set: Set[str] = set()

    while queue and len(visited) < max_sitemaps_to_visit and len(found_products) < max_product_urls:
        sm = normalize_url(queue.pop(0))
        if sm in visited:
            continue
        visited.add(sm)

        try:
            _, xml_text = fetch_text(sm, timeout=timeout)
        except Exception as e:
            notes.append(f"⚠️ Sitemap fetch failed: {sm} ({e})")
            continue

        locs = extract_loc_urls_from_xml(xml_text)

        for loc in locs:
            if len(found_products) >= max_product_urls:
                break

            if looks_like_sitemap_url(loc, origin):
                if loc not in visited and len(visited) + len(queue) < max_sitemaps_to_visit:
                    queue.append(loc)
                continue

            if looks_like_product_url(loc, origin) and loc not in found_set:
                found_set.add(loc)
                found_products.append(loc)

    if found_products:
        notes.append(f"✅ Found {len(found_products)} product URL(s) via sitemap crawl")
    else:
        notes.append("⚠️ No product URLs found in sitemap(s)")

    return found_products, notes


# ----------------------------
# HYBRID CRAWLER ENGINE (Shopify Turbo -> Universal -> Robots -> Scrape)
# ----------------------------
def discover_home_and_products(origin: str, timeout: int) -> Tuple[str, List[str], List[str], bool]:
    notes: List[str] = []
    sitemap_found_products = False

    home_final, home_html = fetch_text(urljoin(origin, "/"), timeout=timeout)
    homepage_url = normalize_url(home_final)

    # Step 1: Shopify Turbo
    try:
        turbo_url = urljoin(origin, SHOPIFY_SITEMAP_PRODUCTS_PATH)
        _, xml_text = fetch_text(turbo_url, timeout=timeout)
        locs = extract_loc_urls_from_xml(xml_text)

        picked: List[str] = []
        seen: Set[str] = set()
        for u in locs:
            if len(picked) >= 3:
                break
            if u in seen:
                continue
            seen.add(u)
            if looks_like_product_url(u, origin):
                picked.append(u)

        if picked:
            notes.append(f"✅ Shopify Turbo: found {len(picked)} product URL(s) via /sitemap_products_1.xml")
            sitemap_found_products = True
            return homepage_url, picked, notes, sitemap_found_products

        notes.append("⚠️ Shopify Turbo: sitemap exists but produced 0 valid product URLs")
    except Exception as e:
        notes.append(f"⚠️ Shopify Turbo failed: {e}")

    # Step 2: Universal sitemap.xml recursive
    try:
        sm_url = urljoin(origin, UNIVERSAL_SITEMAP_PATH)
        products, sm_notes = crawl_sitemaps_for_products(sm_url, origin=origin, timeout=timeout, max_product_urls=3)
        notes.extend(sm_notes)
        if products:
            sitemap_found_products = True
            return homepage_url, products, notes, sitemap_found_products
    except Exception as e:
        notes.append(f"⚠️ Universal sitemap failed: {e}")

    # Step 3: robots.txt sitemap fallbacks
    try:
        robots_text, _ = fetch_robots(origin, timeout=timeout)
        if robots_text:
            sm_urls = discover_sitemaps_from_robots(robots_text)
            if sm_urls:
                notes.append(f"ℹ️ robots.txt lists {len(sm_urls)} sitemap URL(s)")
                for sm in sm_urls:
                    products, sm_notes = crawl_sitemaps_for_products(sm, origin=origin, timeout=timeout, max_product_urls=3)
                    notes.extend(sm_notes)
                    if products:
                        notes.append("✅ Product URLs discovered via robots.txt sitemap")
                        sitemap_found_products = True
                        return homepage_url, products, notes, sitemap_found_products
            else:
                notes.append("⚠️ robots.txt did not list sitemap URLs")
        else:
            notes.append("⚠️ robots.txt not available for sitemap discovery")
    except Exception as e:
        notes.append(f"⚠️ robots.txt fallback failed: {e}")

    # Step 4: scrape homepage fallback
    soup = BeautifulSoup(home_html, "lxml")
    candidates: List[str] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_u = normalize_url(urljoin(origin, href))
        if looks_like_product_url(abs_u, origin):
            candidates.append(abs_u)

    picked: List[str] = []
    seen: Set[str] = set()
    for u in candidates:
        if u in seen:
            continue
        seen.add(u)
        picked.append(u)
        if len(picked) >= 3:
            break

    if picked:
        notes.append(f"✅ Homepage scrape: found {len(picked)} product URL(s)")
    else:
        notes.append("❌ Homepage scrape: no product-like links found")

    return homepage_url, picked, notes, sitemap_found_products


# ----------------------------
# JSON-LD PARSING
# ----------------------------
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
        for k in ("@type", "type", "name"):
            if k in t:
                out.extend(normalize_schema_type(t.get(k)))
    return [x for x in out if x]


def try_parse_json(raw: str) -> Optional[Any]:
    try:
        return json.loads(raw)
    except Exception:
        pass
    no_comments = re.sub(r"//.*?$|/\*.*?\*/", "", raw, flags=re.MULTILINE | re.DOTALL).strip()
    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", no_comments)
    try:
        return json.loads(no_trailing_commas)
    except Exception:
        return None


def split_possible_json_blocks(raw: str) -> List[str]:
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
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == open_ch:
                    depth += 1
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
        if not raw:
            continue
        parsed = try_parse_json(raw)
        if parsed is not None:
            payloads.append(parsed)
            continue
        for block in split_possible_json_blocks(raw):
            p = try_parse_json(block)
            if p is not None:
                payloads.append(p)
    return payloads, len(scripts)


def flatten_jsonld_objects(payloads: List[Any]) -> List[Dict[str, Any]]:
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


def find_first(objs: List[Dict[str, Any]], target: str) -> Optional[Dict[str, Any]]:
    for o in objs:
        if has_type(o, target):
            return o
    return None


def find_all(objs: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    return [o for o in objs if has_type(o, target)]


# ----------------------------
# SCORING RULES
# ----------------------------
def identity_ok(org_obj: Dict[str, Any]) -> bool:
    dis = org_obj.get("disambiguatingDescription")
    if isinstance(dis, str) and dis.strip():
        return True
    same = org_obj.get("sameAs")
    if isinstance(same, str) and same.strip():
        return True
    if isinstance(same, list) and any(isinstance(x, str) and x.strip() for x in same):
        return True
    return False


def offer_dict_has_price(d: Dict[str, Any]) -> bool:
    if "price" in d and str(d.get("price", "")).strip() != "":
        return True
    ps = d.get("priceSpecification")
    if isinstance(ps, dict) and "price" in ps and str(ps.get("price", "")).strip() != "":
        return True
    if isinstance(ps, list):
        for item in ps:
            if isinstance(item, dict) and "price" in item and str(item.get("price", "")).strip() != "":
                return True
    return False


def offers_has_price(offers: Any) -> bool:
    if isinstance(offers, dict):
        return offer_dict_has_price(offers)
    if isinstance(offers, list):
        for item in offers:
            if isinstance(item, dict) and offer_dict_has_price(item):
                return True
    return False


def commerce_ok(product_obj: Dict[str, Any]) -> bool:
    offers = product_obj.get("offers")
    if offers is None:
        return False
    return offers_has_price(offers)


def compute_score(org_found: bool, id_verified: bool, faq_found: bool, prod_found: bool, comm_ready: bool) -> int:
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


def authority_tier1_present(org_obj: Dict[str, Any]) -> bool:
    same_as = org_obj.get("sameAs")
    links: List[str] = []
    if isinstance(same_as, str) and same_as.strip():
        links = [same_as.strip()]
    elif isinstance(same_as, list):
        links = [x.strip() for x in same_as if isinstance(x, str) and x.strip()]
    if not links:
        return False
    for l in links:
        if not re.match(r"^https?://", l, flags=re.I):
            continue
        h = host_of_url(l)
        if any(dom in h for dom in AUTH_TIER1_DOMAINS):
            return True
    return False


# ----------------------------
# SEMANTIC CHECKS
# ----------------------------
def infer_brand_name_from_domain(origin: str) -> str:
    host = domain_host(origin)
    label = host.split(":")[0].split(".")[0]
    label = re.sub(r"[^a-z0-9\-]+", "", label)
    parts = [p for p in label.split("-") if p]
    if not parts:
        return "Your Brand"
    return " ".join(p.capitalize() for p in parts)


def extract_title_h1_from_html(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    title_text = (title_tag.get_text(strip=True) if title_tag else "") or ""
    h1_tag = soup.find("h1")
    h1_text = (h1_tag.get_text(" ", strip=True) if h1_tag else "") or ""
    return title_text, h1_text


def extract_heading_counts(html: str) -> Tuple[int, int, int]:
    soup = BeautifulSoup(html, "lxml")
    h1 = 1 if soup.find("h1") else 0
    h2 = len(soup.find_all("h2"))
    h3 = len(soup.find_all("h3"))
    return h1, h2, h3


def recency_ok(title_text: str, h1_text: str) -> bool:
    hay = f"{title_text} {h1_text}"
    return ("2025" in hay) or ("2026" in hay)


def brand_in_h1(brand: str, h1_text: str) -> bool:
    if not brand or not h1_text:
        return False
    tokens = [t.lower() for t in re.split(r"\s+", brand.strip()) if t.strip()]
    h = h1_text.lower()
    return all(t in h for t in tokens)


def semantic_density(text_len: int, html_len: int) -> float:
    if html_len <= 0:
        return 0.0
    return (float(text_len) / float(html_len)) * 100.0


def scan_images_for_alt(soup: BeautifulSoup) -> Tuple[int, int, List[str]]:
    imgs = soup.find_all("img")
    total = len(imgs)
    missing = 0
    examples: List[str] = []
    for img in imgs:
        alt = img.get("alt")
        if alt is None or str(alt).strip() == "":
            missing += 1
            src = (img.get("src") or "").strip()
            fname = extract_filename_from_src(src) if src else "(no-src)"
            examples.append(fname)
    # top 3 examples
    examples_out = []
    seen = set()
    for f in examples:
        if f in seen:
            continue
        seen.add(f)
        examples_out.append(f)
        if len(examples_out) >= 3:
            break
    return total, missing, examples_out


# ----------------------------
# PAGE AUDIT (Ghost code: <600 visible chars => 0) + Deep Tech Metrics
# ----------------------------
def audit_page(url: str, brand: str, timeout: int) -> PageAudit:
    try:
        final_url, html = fetch_text(url, timeout=timeout)
    except Exception as e:
        return PageAudit(
            requested_url=url,
            final_url=url,
            ok_fetch=False,
            fetch_error=str(e),
            org_found=False,
            identity_verified=False,
            faq_found=False,
            product_found=False,
            commerce_ready=False,
            score=0,
            warnings=[f"⚠️ Crawl Failed: Cannot access this page reliably. ({e})"],
            raw_kb=0.0,
            text_len=0,
            ghost=False,
            h1_present=False,
            h1_has_brand=False,
            semantic_density_pct=0.0,
            h2_count=0,
            h3_count=0,
            img_count=0,
            img_missing_alt_count=0,
            img_missing_alt_examples=[],
        )

    raw_bytes = len(html.encode("utf-8", errors="ignore"))
    raw_kb = raw_bytes / 1024.0
    html_len = len(html)

    soup = BeautifulSoup(html, "lxml")
    visible_text = soup.get_text(" ", strip=True)
    text_len = len(visible_text)

    # H1 / heading counts
    _, page_h1 = extract_title_h1_from_html(html)
    h1_present = bool(page_h1.strip())
    h1_has_brand = brand_in_h1(brand, page_h1)
    _, h2_count, h3_count = extract_heading_counts(html)

    # Images / alt
    img_count, img_missing_alt_count, img_missing_alt_examples = scan_images_for_alt(soup)

    # Semantic density
    sem_density = semantic_density(text_len, html_len)

    ghost = text_len < 600

    payloads, script_count = extract_jsonld_payloads(html)
    objs = flatten_jsonld_objects(payloads)

    org = find_first(objs, "Organization")
    products = find_all(objs, "Product")
    faqs = find_all(objs, "FAQPage")

    org_found = org is not None
    prod_found = len(products) > 0
    faq_found = len(faqs) > 0

    id_verified = identity_ok(org) if org else False
    comm_ready = any(commerce_ok(p) for p in products) if prod_found else False

    base_score = compute_score(org_found, id_verified, faq_found, prod_found, comm_ready)

    warnings: List[str] = []
    if script_count > 0 and len(payloads) == 0:
        warnings.append("⚠️ Schema Parsing Error: JSON-LD tags exist but appear malformed. AI may ignore them.")

    # Deep tech flags
    if sem_density < 10.0:
        warnings.append("⚠️ Poor Semantic Ratio: Bloated Code (high code-to-text noise). Too much noise can confuse AI context windows.")

    if img_count > 0 and img_missing_alt_count > 0:
        warnings.append("⚠️ Visual Context Gap: Some images are missing alt text. AI loses visual semantics.")

    if ghost:
        warnings.append("⚠️ Render Blocking: Client-Side JavaScript detected.")
        warnings.append("Impact: Severe. AI Agents see a blank page.")
        score = 0
    else:
        score = base_score

    return PageAudit(
        requested_url=url,
        final_url=final_url,
        ok_fetch=True,
        fetch_error=None,
        org_found=org_found,
        identity_verified=id_verified,
        faq_found=faq_found,
        product_found=prod_found,
        commerce_ready=comm_ready,
        score=score,
        warnings=warnings,
        raw_kb=raw_kb,
        text_len=text_len,
        ghost=ghost,
        h1_present=h1_present,
        h1_has_brand=h1_has_brand,
        semantic_density_pct=sem_density,
        h2_count=h2_count,
        h3_count=h3_count,
        img_count=img_count,
        img_missing_alt_count=img_missing_alt_count,
        img_missing_alt_examples=img_missing_alt_examples,
    )


# ----------------------------
# PATCH SNIPPETS (Phase 1)
# ----------------------------
def organization_jsonld_template(domain: str, brand: str) -> str:
    return json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "Organization",
            "name": brand,
            "url": domain,
            "disambiguatingDescription": f"{brand} is the official brand website at {domain}.",
            "sameAs": [
                f"{domain}/about",
                f"{domain}/contact",
            ],
        },
        indent=2,
        ensure_ascii=False,
    )


# ----------------------------
# RISK LABELING
# ----------------------------
def revenue_risk_from_score(score: int) -> Tuple[str, str]:
    # label, severity for UI
    if score < 50:
        return "High", "error"
    if score < 75:
        return "Medium", "warning"
    return "Low", "success"


def impact_badge(impact: str) -> str:
    impact = impact.upper().strip()
    if impact == "HIGH":
        return "[Impact: HIGH]"
    if impact == "MED":
        return "[Impact: MED]"
    return "[Impact: LOW]"


def effort_badge(effort: str) -> str:
    effort = effort.upper().strip()
    if effort == "LOW":
        return "[Effort: LOW]"
    if effort == "MED":
        return "[Effort: MED]"
    return "[Effort: HIGH]"


# ----------------------------
# SITE AUDIT RUNNER (Deep Tech)
# ----------------------------
def run_site_audit(input_url: str, timeout: int) -> SiteAuditResult:
    origin = origin_from_url(input_url)
    brand = infer_brand_name_from_domain(origin)

    # robots / AI blocker
    robots_text, robots_err = fetch_robots(origin, timeout=timeout)
    robots_access = robots_text is not None
    per_bot_blocked: Dict[str, bool] = {a: False for a in AI_BOTS_EXTENDED}
    any_ai_blocked = False
    sitemaps_from_robots: List[str] = []
    if robots_text:
        per_bot_blocked = parse_robots_for_blocks(robots_text, AI_BOTS_EXTENDED)
        any_ai_blocked = any(per_bot_blocked.values())
        sitemaps_from_robots = discover_sitemaps_from_robots(robots_text)

    # llms.txt
    llms_ok, llms_err = fetch_llms_txt(origin, timeout=timeout)

    # fetch homepage for semantic checks + (crawler needs it)
    home_final, home_html = fetch_text(urljoin(origin, "/"), timeout=timeout)
    homepage_url = normalize_url(home_final)

    # discover products (hybrid)
    discovered_home, product_urls, crawl_notes, sitemap_found_products = discover_home_and_products(origin, timeout=timeout)
    homepage_url = normalize_url(discovered_home) if discovered_home else homepage_url

    scan_urls: List[str] = [homepage_url]
    for u in product_urls:
        nu = normalize_url(u)
        if nu not in scan_urls and (not is_disallowed_asset(nu)) and (not nu.lower().endswith(".xml")):
            scan_urls.append(nu)
    scan_urls = scan_urls[:4]

    audits: List[PageAudit] = []
    for u in scan_urls:
        audits.append(audit_page(u, brand=brand, timeout=timeout))

    # homepage title/h1 semantic
    title_text, h1_text = extract_title_h1_from_html(home_html)
    recency_pass = recency_ok(title_text, h1_text)
    entity_pass_home = brand_in_h1(brand, h1_text)

    # homepage heading structure counts
    _, home_h2_count, home_h3_count = extract_heading_counts(home_html)

    # authority check (Tier 1) across scanned pages
    authority_pass_any = False
    org_seen_any = False
    for a in audits:
        if not a.ok_fetch:
            continue
        try:
            _, html = fetch_text(a.final_url, timeout=timeout)
        except Exception:
            continue
        payloads, _ = extract_jsonld_payloads(html)
        objs = flatten_jsonld_objects(payloads)
        org = find_first(objs, "Organization")
        if org:
            org_seen_any = True
            if authority_tier1_present(org):
                authority_pass_any = True
                break

    # derived flags
    ghost_driver = any(a.ghost for a in audits if a.ok_fetch)

    org_present_any = any(a.org_found for a in audits if a.ok_fetch)
    product_present_any = any(a.product_found for a in audits if a.ok_fetch)
    faq_present_any = any(a.faq_found for a in audits if a.ok_fetch)
    commerce_ready_any = any(a.commerce_ready for a in audits if a.ok_fetch)

    trust_driver_fail = (not authority_pass_any) or (not org_seen_any)

    # deep tech aggregates
    densities = [a.semantic_density_pct for a in audits if a.ok_fetch]
    avg_density = sum(densities) / len(densities) if densities else 0.0
    total_imgs = sum(a.img_count for a in audits if a.ok_fetch)
    total_missing_alt = sum(a.img_missing_alt_count for a in audits if a.ok_fetch)

    # aggregate top 3 filenames across pages
    all_missing_names: List[str] = []
    for a in audits:
        all_missing_names.extend(a.img_missing_alt_examples)
    top_missing: List[str] = []
    seen = set()
    for n in all_missing_names:
        if n in seen:
            continue
        seen.add(n)
        top_missing.append(n)
        if len(top_missing) >= 3:
            break

    # scoring
    health_score = round(sum(a.score for a in audits) / len(audits)) if audits else 0
    leakage_pct = max(0, min(100, 100 - health_score))

    deep = SiteDeepTech(
        llms_txt_accessible=llms_ok,
        llms_txt_error=llms_err,
        robots_access=robots_access,
        robots_error=robots_err,
        any_ai_blocked=any_ai_blocked,
        per_bot_blocked=per_bot_blocked,
        sitemap_found_products=sitemap_found_products,
        sitemap_notes=crawl_notes,
        title_text=title_text,
        h1_text=h1_text,
        recency_pass=recency_pass,
        entity_pass_home=entity_pass_home,
        authority_pass_any=authority_pass_any,
        org_seen_any=org_seen_any,
        ghost_driver=ghost_driver,
        trust_driver_fail=trust_driver_fail,
        avg_semantic_density_pct=avg_density,
        total_imgs=total_imgs,
        total_missing_alt=total_missing_alt,
        top_missing_alt_filenames=top_missing,
        home_h2_count=home_h2_count,
        home_h3_count=home_h3_count,
        pages_scanned=len(audits),
    )

    return SiteAuditResult(
        origin=origin,
        brand=brand,
        homepage_url=homepage_url,
        scan_urls=scan_urls,
        audits=audits,
        deep=deep,
        org_present_any=org_present_any,
        product_present_any=product_present_any,
        faq_present_any=faq_present_any,
        commerce_ready_any=commerce_ready_any,
        health_score=health_score,
        leakage_pct=leakage_pct,
    )


# ----------------------------
# UI HELPERS
# ----------------------------
def executive_summary_text(result: SiteAuditResult) -> str:
    d = result.deep
    issues: List[str] = []
    proofs: List[str] = []

    if not d.llms_txt_accessible:
        issues.append("missing an llms.txt file")
        proofs.append("llms.txt: not found")
    if d.total_missing_alt > 0:
        issues.append(f"has {d.total_missing_alt} images unreadable to AI (missing alt)")
        if d.top_missing_alt_filenames:
            proofs.append("examples: " + ", ".join(d.top_missing_alt_filenames))
    if d.ghost_driver:
        issues.append("contains pages that render blank to fast AI crawlers (Ghost Code)")
        proofs.append("render: <600 visible characters")
    if d.avg_semantic_density_pct < 10.0:
        issues.append("has a poor semantic ratio (<10%)")
        proofs.append(f"semantic density: {d.avg_semantic_density_pct:.1f}%")
    if d.trust_driver_fail:
        issues.append("lacks Tier-1 authority nodes (Wikidata/Wikipedia/Crunchbase)")
        proofs.append("authority: no Tier-1 sameAs link found")
    if not d.recency_pass:
        issues.append("lacks a freshness signal (2025/2026)")
        proofs.append("recency: not detected in Title/H1")
    if not d.entity_pass_home:
        issues.append("has entity ambiguity (brand not in homepage H1)")
        proofs.append("entity: H1 mismatch")

    if not issues:
        return "✅ Strong baseline detected: No critical deficiencies surfaced in core AI visibility layers."

    headline = "Critical Deficiencies Detected: " + ", ".join(issues[:3])
    if len(issues) > 3:
        headline += f", +{len(issues) - 3} more."

    proofline = ""
    if proofs:
        proofline = " Proof: " + " | ".join(proofs[:3])

    return headline + proofline


def render_badge(text: str) -> str:
    # lightweight badge formatting using markdown
    return f"`{text}`"


# ----------------------------
# APP UI
# ----------------------------
st.title("Agentic Infrastructure Audit")
st.caption(
    "WordLift-style deep technical audit for AI agents. "
    "Competitor benchmarking + insight cards + per-page proofs."
)

with st.container(border=True):
    col_a, col_b = st.columns(2)
    with col_a:
        your_url = st.text_input("Your Website URL", placeholder="https://yourbrand.com")
    with col_b:
        comp_url = st.text_input("Competitor Website URL (Optional)", placeholder="https://competitor.com")

    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        timeout = st.slider("Request timeout (seconds)", min_value=5, max_value=60, value=DEFAULT_TIMEOUT, step=5)
    with c2:
        st.write("")
        run = st.button("Run Competitive Audit", type="primary", use_container_width=True)

if not run:
    st.info("Enter your site (and optionally a competitor) to generate a deep AEO/AI visibility audit.")
    st.stop()

your_origin = origin_from_url(your_url)
if not your_origin:
    st.error("Please enter a valid 'Your Website URL' (e.g., https://example.com).")
    st.stop()

comp_origin = origin_from_url(comp_url) if comp_url.strip() else ""

with st.status("Running deep audit…", expanded=False) as status:
    status.update(label="Auditing your website…", state="running")
    your_result = run_site_audit(your_origin, timeout=timeout)
    time.sleep(0.05)

    comp_result: Optional[SiteAuditResult] = None
    if comp_origin:
        status.update(label="Auditing competitor…", state="running")
        try:
            comp_result = run_site_audit(comp_origin, timeout=timeout)
        except Exception as e:
            comp_result = None
            st.warning(f"Competitor audit failed: {e}")

    status.update(label="Complete.", state="complete")

# ----------------------------
# HEAD-TO-HEAD SCORECARD (Retain competitor feature)
# ----------------------------
st.header("Head-to-Head Scorecard")

if comp_result:
    left, right = st.columns(2)

    with left:
        st.subheader("Your Site")
        st.metric("Agentic Health Score", f"{your_result.health_score}/100", f"Industry Avg: {INDUSTRY_AVERAGE_SCORE}/100")
        risk_label, severity = revenue_risk_from_score(your_result.health_score)
        if severity == "error":
            st.error(f"Revenue Risk: {risk_label} — AI Traffic Leakage: {your_result.leakage_pct}%")
        elif severity == "warning":
            st.warning(f"Revenue Risk: {risk_label} — AI Traffic Leakage: {your_result.leakage_pct}%")
        else:
            st.success(f"Revenue Risk: {risk_label} — AI Traffic Leakage: {your_result.leakage_pct}%")

    with right:
        st.subheader("Competitor")
        st.metric("Agentic Health Score", f"{comp_result.health_score}/100", "Benchmark")
        comp_risk, comp_sev = revenue_risk_from_score(comp_result.health_score)
        if comp_sev == "error":
            st.error(f"Revenue Risk: {comp_risk} — AI Traffic Leakage: {comp_result.leakage_pct}%")
        elif comp_sev == "warning":
            st.warning(f"Revenue Risk: {comp_risk} — AI Traffic Leakage: {comp_result.leakage_pct}%")
        else:
            st.success(f"Revenue Risk: {comp_risk} — AI Traffic Leakage: {comp_result.leakage_pct}%")

    if comp_result.health_score > your_result.health_score:
        comp_host = domain_host(comp_result.origin)
        st.warning(f"⚠️ Alert: {comp_host} is outranking you on Authority Signals.")
    elif comp_result.health_score < your_result.health_score:
        st.success("✅ You are ahead on AI visibility signals.")
    else:
        st.info("ℹ️ You are tied on AI visibility signals.")
else:
    st.metric("Agentic Health Score", f"{your_result.health_score}/100", f"Industry Avg: {INDUSTRY_AVERAGE_SCORE}/100")
    risk_label, sev = revenue_risk_from_score(your_result.health_score)
    if sev == "error":
        st.error(f"Revenue Risk: {risk_label} — AI Traffic Leakage: {your_result.leakage_pct}%")
    elif sev == "warning":
        st.warning(f"Revenue Risk: {risk_label} — AI Traffic Leakage: {your_result.leakage_pct}%")
    else:
        st.success(f"Revenue Risk: {risk_label} — AI Traffic Leakage: {your_result.leakage_pct}%")

st.divider()

# ----------------------------
# EXECUTIVE SUMMARY
# ----------------------------
st.subheader("Executive Summary")
st.write(executive_summary_text(your_result))

st.divider()

# ----------------------------
# INSIGHT CARD GRID (2x2)
# ----------------------------
st.subheader("Deep Technical Insights")

c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

d = your_result.deep

# Card 1: AI Access & Control
with c1:
    with st.container():
        st.markdown("### AI Access & Control")
        # Robots
        if not d.robots_access:
            st.error(f"{impact_badge('HIGH')} Robots.txt unreachable — crawlers may downgrade trust. ({d.robots_error})")
        else:
            st.success("Robots.txt reachable.")

        if d.any_ai_blocked:
            blocked = [k for k, v in d.per_bot_blocked.items() if v]
            st.error(f"{impact_badge('HIGH')} AI agents blocked in robots.txt: {', '.join(blocked)}")
            st.caption("Impact: AI agents may not crawl your site at all.")
        else:
            st.success("No explicit AI-agent blocks detected (GPTBot/CCBot/Google-Extended/Anthropic/Claude).")

        # llms.txt
        if not d.llms_txt_accessible:
            st.error(f"{impact_badge('HIGH')} {effort_badge('LOW')} llms.txt missing — critical for explicit AI permissioning.")
            st.caption("Why it matters: explicit AI policy reduces ambiguity and improves agent trust decisions.")
        else:
            st.success("llms.txt detected (explicit AI permissioning present).")

        # Sitemap
        if d.sitemap_found_products:
            st.success("Sitemap discovery: product URLs found (crawl efficiency improved).")
        else:
            st.warning(f"{impact_badge('MED')} Sitemap discovery weak — product pages may be under-crawled.")

# Card 2: Semantic Structure
with c2:
    with st.container():
        st.markdown("### Semantic Structure")
        if d.ghost_driver:
            st.error(f"{impact_badge('HIGH')} Render Blocking (Ghost Code) detected.")
            st.caption("Severe: AI agents that read raw HTML see a blank page.")
        else:
            st.success("Render accessibility: readable HTML present.")

        # Heading hierarchy
        if d.home_h2_count == 0 and d.home_h3_count == 0:
            st.warning(f"{impact_badge('MED')} H1-H3 hierarchy is thin on homepage (weak topical structure).")
        else:
            st.success(f"Heading depth: H2={d.home_h2_count}, H3={d.home_h3_count} on homepage.")

        # Semantic density
        if d.avg_semantic_density_pct < 10.0:
            st.error(f"{impact_badge('HIGH')} Poor Semantic Ratio: {d.avg_semantic_density_pct:.1f}%")
            st.caption("Too much code noise can confuse AI context windows and dilute relevance.")
        elif d.avg_semantic_density_pct < 18.0:
            st.warning(f"{impact_badge('MED')} Semantic Ratio is moderate: {d.avg_semantic_density_pct:.1f}%")
        else:
            st.success(f"Semantic Ratio: {d.avg_semantic_density_pct:.1f}% (healthy signal-to-noise).")

# Card 3: Visual Context
with c3:
    with st.container():
        st.markdown("### Visual Context")
        st.write(f"Images scanned: **{d.total_imgs}**")
        if d.total_imgs == 0:
            st.info("No images detected on scanned pages.")
        else:
            if d.total_missing_alt > 0:
                st.warning(f"{impact_badge('MED')} {effort_badge('LOW')} Missing alt text on **{d.total_missing_alt}** images.")
                if d.top_missing_alt_filenames:
                    st.caption("Proof (top filenames): " + ", ".join(d.top_missing_alt_filenames))
                st.caption("Why it matters: alt text anchors visual content to AI-readable meaning.")
            else:
                st.success("All scanned images include alt text (strong visual semantics).")

# Card 4: Entity & Trust
with c4:
    with st.container():
        st.markdown("### Entity & Trust")
        # Schema coverage
        if not your_result.org_present_any:
            st.error(f"{impact_badge('HIGH')} Organization schema missing — brand entity is ambiguous.")
        else:
            st.success("Organization schema detected (brand entity present).")

        if not your_result.faq_present_any:
            st.warning(f"{impact_badge('MED')} {effort_badge('LOW')} FAQPage schema missing — weaker answer engine visibility.")
        else:
            st.success("FAQPage schema detected (answer engine active).")

        if not your_result.product_present_any:
            st.warning(f"{impact_badge('MED')} Product schema not detected on scanned pages.")
        else:
            st.success("Product schema detected (commerce entity present).")

        if your_result.product_present_any and not your_result.commerce_ready_any:
            st.error(f"{impact_badge('HIGH')} Commerce blocked — offers/price missing. AI cannot safely transact.")
        elif your_result.commerce_ready_any:
            st.success("Commerce-ready offers detected (price/stock signals present).")

        # Authority + Recency + Entity Clarity (homepage)
        if d.trust_driver_fail:
            st.warning(f"{impact_badge('HIGH')} Trust Vacuum — no Tier-1 authority nodes (Wikidata/Wikipedia/Crunchbase).")
            st.caption("Tip: Knowledge Graph links reduce hallucination and improve entity confidence.")
        else:
            st.success("Tier-1 authority node detected (Knowledge Graph connected).")

        if not d.recency_pass:
            st.warning(f"{impact_badge('MED')} Recency signal missing (no 2025/2026 in Title/H1).")
        else:
            st.success("Recency signal detected (2025/2026 present).")

        if not d.entity_pass_home:
            st.warning(f"{impact_badge('MED')} Entity clarity risk — brand name not clearly present in homepage H1.")
        else:
            st.success("Homepage H1 includes brand name (entity clarity strong).")

st.divider()

# ----------------------------
# DETAILED PAGE FINDINGS (Restored Per-Page Expanders)
# ----------------------------
st.subheader("Detailed Page Findings (Proof)")
st.caption("Home + discovered product pages. Each page shows exactly what is missing.")

for page in your_result.audits:
    title = f"{page.final_url} — {page.score}/100"
    with st.expander(title, expanded=False):
        if not page.ok_fetch:
            st.error(page.fetch_error or "Fetch failed.")
            continue

        # Quick metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Raw Size", f"{page.raw_kb:.1f} KB")
        with m2:
            st.metric("Visible Text", f"{page.text_len}")
        with m3:
            st.metric("Semantic Density", f"{page.semantic_density_pct:.1f}%")
        with m4:
            st.metric("Missing Alt", f"{page.img_missing_alt_count}/{page.img_count}")

        # Page-level diagnostics
        if page.ghost:
            st.error(f"{impact_badge('HIGH')} Render Blocking (Ghost Code): page is effectively blank to fast AI crawlers.")
        if page.semantic_density_pct < 10.0:
            st.warning(f"{impact_badge('HIGH')} Poor Semantic Ratio (bloated code): {page.semantic_density_pct:.1f}%")
        if page.img_missing_alt_count > 0:
            st.warning(f"{impact_badge('MED')} {effort_badge('LOW')} Missing alt on {page.img_missing_alt_count} image(s).")
            if page.img_missing_alt_examples:
                st.caption("Examples: " + ", ".join(page.img_missing_alt_examples))

        st.markdown("**Schema Checklist**")
        st.write(f"- Organization Schema: {'✅' if page.org_found else '❌'}")
        st.write(f"- Identity Verified (sameAs/disambiguatingDescription): {'✅' if page.identity_verified else '❌'}")
        st.write(f"- Product Schema: {'✅' if page.product_found else '❌'}")
        st.write(f"- Commerce Ready (offers/price): {'✅' if page.commerce_ready else '❌'}")
        st.write(f"- FAQPage Schema: {'✅' if page.faq_found else '❌'}")

        st.markdown("**On-Page Semantic Checklist**")
        st.write(f"- H1 Present: {'✅' if page.h1_present else '❌'}")
        st.write(f"- H1 Contains Brand Name: {'✅' if page.h1_has_brand else '❌'}")
        st.write(f"- Heading Depth: H2={page.h2_count}, H3={page.h3_count}")

        if page.warnings:
            st.markdown("**Alerts**")
            for w in page.warnings:
                st.info(w)

with st.expander("Crawl Notes", expanded=False):
    for n in your_result.deep.sitemap_notes:
        st.write(n)
    if your_result.deep.robots_access:
        st.write("Robots.txt: accessible.")
        if your_result.deep.any_ai_blocked:
            blocked = [k for k, v in your_result.deep.per_bot_blocked.items() if v]
            st.write("Blocked agents: " + ", ".join(blocked))
        else:
            st.write("No explicit AI-agent blocks detected in robots.txt.")
    else:
        st.write(f"Robots.txt: not accessible ({your_result.deep.robots_error}).")
    st.write("Pages scanned:")
    for u in your_result.scan_urls:
        st.write(f"- {u}")

st.divider()

# ----------------------------
# PHASE 1 / PHASE 2 UPSELL
# ----------------------------
st.subheader("Phase 1 vs Phase 2")

col_left, col_right = st.columns([0.55, 0.45])

with col_left:
    st.markdown("### Phase 1 (Defense): Basic Patch")
    st.caption("Minimum 'Hello' tag to anchor brand identity and reduce AI confusion.")
    st.code(organization_jsonld_template(your_result.origin, your_result.brand), language="json")
    if not your_result.deep.llms_txt_accessible:
        st.caption("Quick win: add /llms.txt to explicitly declare AI access policies.")

with col_right:
    st.markdown("### 🚀 Phase 2 (Offense): The AEO Engine")
    if comp_result:
        comp_host = domain_host(comp_result.origin)
        st.info(f"To beat {comp_host}, you need Programmatic Verification and Sentiment Injection.")
    else:
        st.info("To win, you need Programmatic Verification and Sentiment Injection.")
    st.caption("This includes Knowledge Graph bootstrapping, structured content maps, and competitive entity displacement.")
    st.link_button("👉 Book Your Strategy Call", "https://calendly.com", use_container_width=True)
