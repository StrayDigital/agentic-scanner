# app.py — Premium Agentic Infrastructure Audit (Vercel-style Dark Mode)
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
# Premium UI (Vercel-style Dark Mode)
# ----------------------------
st.set_page_config(page_title="Agentic Infrastructure Audit", page_icon="🧠", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
}

.stApp {
  background: #0E1117;
  color: #E6EDF3;
}

/* Center + max width container */
.block-container {
  max-width: 900px;
  padding-top: 2rem !important;
  padding-bottom: 3rem !important;
}

/* Inputs */
div[data-testid="stTextInput"] input {
  background: transparent !important;
  border: 1px solid #30363D !important;
  border-radius: 10px !important;
  color: #E6EDF3 !important;
  padding: 0.7rem 0.9rem !important;
}

div[data-testid="stTextInput"] label {
  color: #E6EDF3 !important;
  opacity: 0.9;
}

/* Button */
div.stButton > button {
  width: 100% !important;
  background: linear-gradient(45deg, #2ecc71, #3498db) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
  transition: all 0.3s ease !important;
  padding: 0.8rem 1rem !important;
}

div.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

/* Metric cards */
div[data-testid="stMetric"] {
  background: #161b22 !important;
  border: 1px solid #30363D !important;
  border-radius: 12px !important;
  padding: 14px 14px !important;
}

div[data-testid="stMetricLabel"] {
  color: #9DA7B3 !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
}

div[data-testid="stMetricValue"] {
  color: #E6EDF3 !important;
  font-weight: 800 !important;
}

/* Expanders */
details {
  background: #161b22 !important;
  border: 1px solid #30363D !important;
  border-radius: 12px !important;
  padding: 8px 12px !important;
}

details summary {
  color: #E6EDF3 !important;
  font-weight: 700 !important;
}

hr {
  border-color: rgba(48,54,61,0.55) !important;
}

/* Custom thick gauge */
.gauge-wrap {
  background: #161b22;
  border: 1px solid #30363D;
  border-radius: 12px;
  padding: 14px;
}
.gauge-title {
  color: #9DA7B3;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.gauge-row {
  display:flex;
  align-items: baseline;
  justify-content: space-between;
  margin-top: 8px;
}
.gauge-value {
  font-size: 28px;
  font-weight: 800;
  color: #E6EDF3;
}
.gauge-sub {
  font-size: 12px;
  color: #9DA7B3;
  margin-top: 6px;
}
.gauge-bar {
  margin-top: 10px;
  height: 14px;
  border-radius: 999px;
  background: rgba(230,237,243,0.08);
  overflow: hidden;
}
.gauge-fill {
  height: 100%;
  width: 0%;
  border-radius: 999px;
  background: linear-gradient(90deg, #ff4d4d 0%, #ff9f1a 55%, #f1c40f 100%);
}

/* Link button */
a[data-testid="stLinkButton"] {
  width: 100% !important;
  display: inline-flex !important;
  justify-content: center !important;
  border-radius: 10px !important;
  border: 1px solid #30363D !important;
  padding: 0.85rem 1rem !important;
  font-weight: 800 !important;
  color: #E6EDF3 !important;
  background: transparent !important;
  text-decoration: none !important;
}
a[data-testid="stLinkButton"]:hover {
  border-color: rgba(52,152,219,0.9) !important;
  box-shadow: 0 6px 20px rgba(52,152,219,0.15) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


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

AI_BOTS = ("GPTBot", "CCBot", "Google-Extended")

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


# ----------------------------
# NETWORKING
# ----------------------------
def fetch_text(url: str, timeout: int) -> Tuple[str, str]:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text


# ----------------------------
# ROBOTS / AI BLOCKING
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
# HYBRID CRAWLER ENGINE
# ----------------------------
def discover_home_and_products(origin: str, timeout: int) -> Tuple[str, List[str], List[str], bool]:
    notes: List[str] = []
    sitemap_reachable = False

    home_final, home_html = fetch_text(urljoin(origin, "/"), timeout=timeout)
    homepage_url = normalize_url(home_final)

    # Step 1: Shopify Turbo
    try:
        turbo_url = urljoin(origin, SHOPIFY_SITEMAP_PRODUCTS_PATH)
        _, xml_text = fetch_text(turbo_url, timeout=timeout)
        sitemap_reachable = True
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
            return homepage_url, picked, notes, sitemap_reachable

        notes.append("⚠️ Shopify Turbo: sitemap exists but produced 0 valid product URLs")
    except Exception as e:
        notes.append(f"⚠️ Shopify Turbo failed: {e}")

    # Step 2: Universal sitemap.xml recursive
    try:
        sm_url = urljoin(origin, UNIVERSAL_SITEMAP_PATH)
        products, sm_notes = crawl_sitemaps_for_products(sm_url, origin=origin, timeout=timeout, max_product_urls=3)
        notes.extend(sm_notes)
        if products:
            sitemap_reachable = True
            return homepage_url, products, notes, sitemap_reachable
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
                        sitemap_reachable = True
                        return homepage_url, products, notes, sitemap_reachable
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

    return homepage_url, picked, notes, sitemap_reachable


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
# PAGE AUDIT (with Ghost Code override)
# ----------------------------
def audit_page(url: str, timeout: int) -> PageAudit:
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
        )

    raw_bytes = len(html.encode("utf-8", errors="ignore"))
    raw_kb = raw_bytes / 1024.0

    soup = BeautifulSoup(html, "lxml")
    visible_text = soup.get_text(" ", strip=True)
    text_len = len(visible_text)

    # Ghost Code Detector (Critical) -> automatic 0
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
    )


# ----------------------------
# HOMEPAGE SEMANTIC CHECKS
# ----------------------------
def infer_brand_name_from_domain(origin: str) -> str:
    host = domain_host(origin)
    label = host.split(":")[0].split(".")[0]
    label = re.sub(r"[^a-z0-9\-]+", "", label)
    parts = [p for p in label.split("-") if p]
    if not parts:
        return "Your Brand"
    return " ".join(p.capitalize() for p in parts)


def extract_homepage_title_h1(home_html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(home_html, "lxml")
    title_tag = soup.find("title")
    title_text = (title_tag.get_text(strip=True) if title_tag else "") or ""
    h1_tag = soup.find("h1")
    h1_text = (h1_tag.get_text(" ", strip=True) if h1_tag else "") or ""
    return title_text, h1_text


def recency_ok(title_text: str, h1_text: str) -> bool:
    hay = f"{title_text} {h1_text}"
    return ("2025" in hay) or ("2026" in hay)


def brand_in_h1(brand: str, h1_text: str) -> bool:
    if not brand or not h1_text:
        return False
    tokens = [t.lower() for t in re.split(r"\s+", brand.strip()) if t.strip()]
    h = h1_text.lower()
    return all(t in h for t in tokens)


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


def faqpage_jsonld_template(domain: str, brand: str) -> str:
    return json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": f"What is {brand}?",
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": f"{brand} is available at {domain}. Replace this with your real FAQ answer.",
                    },
                }
            ],
        },
        indent=2,
        ensure_ascii=False,
    )


# ----------------------------
# UI HELPERS
# ----------------------------
def passfail(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"


def leakage_gauge_html(leakage_pct: int) -> str:
    leakage_pct = max(0, min(100, int(leakage_pct)))
    return f"""
<div class="gauge-wrap">
  <div class="gauge-title">📉 Estimated AI Traffic Leakage</div>
  <div class="gauge-row">
    <div class="gauge-value">{leakage_pct}%</div>
    <div style="color:#9DA7B3; font-size:12px; font-weight:600;">Market Readiness</div>
  </div>
  <div class="gauge-bar">
    <div class="gauge-fill" style="width:{leakage_pct}%"></div>
  </div>
  <div class="gauge-sub">Percentage of intent-based queries where your brand is ignored due to technical gaps.</div>
</div>
"""


# ----------------------------
# HEADER LAYOUT (70/30)
# ----------------------------
header_left, header_right = st.columns([0.7, 0.3], vertical_alignment="bottom")

with header_left:
    st.markdown("## Agentic Infrastructure Audit")
    st.markdown(
        "<div style='color:#9DA7B3; font-size:14px; margin-top:-4px;'>"
        "A premium AEO diagnostic: render accessibility, freshness signals, entity clarity, and trust authority."
        "</div>",
        unsafe_allow_html=True,
    )

with header_right:
    # Button already defined by Streamlit below; keep this column for a clean layout.
    # We'll render inputs and button just under the header, but align the button to the right column.
    pass

st.markdown("")

# Inputs row aligned like Vercel: input wide + button at right
in_left, in_right = st.columns([0.7, 0.3], vertical_alignment="bottom")
with in_left:
    site_url = st.text_input("Website URL", value="", placeholder="https://example.com")
with in_right:
    run = st.button("Run Scan", type="primary")

with st.expander("Advanced Settings", expanded=False):
    timeout = st.slider("Request timeout (seconds)", min_value=5, max_value=60, value=DEFAULT_TIMEOUT, step=5)

st.markdown("---")

if run:
    origin = origin_from_url(site_url)
    if not origin:
        st.error("Please enter a valid URL (e.g., https://example.com).")
        st.stop()

    brand = infer_brand_name_from_domain(origin)

    # Infrastructure: robots
    robots_text, robots_err = fetch_robots(origin, timeout=timeout)
    robots_access = robots_text is not None
    per_bot_blocked: Dict[str, bool] = {a: False for a in AI_BOTS}
    any_ai_blocked = False
    if robots_text:
        per_bot_blocked = parse_robots_for_blocks(robots_text, AI_BOTS)
        any_ai_blocked = any(per_bot_blocked.values())

    notes: List[str] = []
    audits: List[PageAudit] = []
    scan_urls: List[str] = []

    homepage_url = ""
    home_html = ""
    sitemap_reachable = False
    found_products = False

    with st.status("Running scan…", expanded=False) as status:
        status.update(label="Fetching homepage…", state="running")
        try:
            home_final, home_html = fetch_text(urljoin(origin, "/"), timeout=timeout)
            homepage_url = normalize_url(home_final)
        except Exception as e:
            status.update(label="Homepage fetch failed.", state="error")
            st.error(f"Homepage fetch failed: {e}")
            st.stop()

        status.update(label="Discovering product pages (Hybrid crawler)…", state="running")
        try:
            discovered_home, product_urls, crawl_notes, sitemap_reachable = discover_home_and_products(origin, timeout=timeout)
            notes.extend(crawl_notes)
            homepage_url = normalize_url(discovered_home) if discovered_home else homepage_url
        except Exception as e:
            notes.append(f"⚠️ Discovery engine error: {e}")
            product_urls = []

        scan_urls = [homepage_url]
        for u in product_urls:
            nu = normalize_url(u)
            if nu not in scan_urls and (not is_disallowed_asset(nu)) and (not nu.lower().endswith(".xml")):
                scan_urls.append(nu)
        scan_urls = scan_urls[:4]
        found_products = len(scan_urls) > 1

        status.update(label=f"Auditing {len(scan_urls)} page(s)…", state="running")
        for i, u in enumerate(scan_urls, start=1):
            status.update(label=f"Auditing {i}/{len(scan_urls)}…", state="running")
            audits.append(audit_page(u, timeout=timeout))
            time.sleep(0.05)

        status.update(label="Complete.", state="complete")

    # Semantic checks on homepage
    title_text, h1_text = extract_homepage_title_h1(home_html)
    recency_pass = recency_ok(title_text, h1_text)
    entity_pass = brand_in_h1(brand, h1_text)

    # Authority check across scanned pages (Tier 1 sources)
    authority_pass = False
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
                authority_pass = True
                break

    # Drivers
    ghost_driver = any(a.ghost for a in audits if a.ok_fetch)
    trust_driver_fail = (not authority_pass) or (not org_seen_any)

    # Schema layer presence
    org_present = any(a.org_found for a in audits if a.ok_fetch)
    product_present = any(a.product_found for a in audits if a.ok_fetch)
    faq_present = any(a.faq_found for a in audits if a.ok_fetch)
    commerce_ready = any(a.commerce_ready for a in audits if a.ok_fetch)

    # Health score
    health_score = round(sum(a.score for a in audits) / len(audits)) if audits else 0
    leakage = max(0, min(100, 100 - health_score))

    # ----------------------------
    # LAYER 1: SCORECARD
    # ----------------------------
    sc1, sc2 = st.columns([0.5, 0.5])
    with sc1:
        st.metric("Agentic Health Score", f"{health_score}/100", f"Industry Avg: {INDUSTRY_AVERAGE_SCORE}/100")
    with sc2:
        st.markdown(leakage_gauge_html(leakage), unsafe_allow_html=True)

    st.markdown("")

    if any_ai_blocked:
        blocked_list = [k for k, v in per_bot_blocked.items() if v]
        st.error(f"🚨 AI Access Blocked in robots.txt: {', '.join(blocked_list)}")
    if ghost_driver:
        st.warning("⚠️ Render Blocking: Client-Side JavaScript detected. Impact: Severe. AI Agents see a blank page.")

    # ----------------------------
    # LAYER 2: TOP INVISIBILITY DRIVERS (ONLY FAILS)
    # ----------------------------
    drivers: List[Tuple[str, str]] = []
    if ghost_driver:
        drivers.append(("Rendering Latency", "Render Blocking: Your page returns 200 OK but exposes almost no readable HTML content to fast crawlers."))
    if not recency_pass:
        drivers.append(("Stale Signal Risk", "Outdated Metadata: Homepage Title/H1 does not explicitly signal '2025' or '2026'."))
    if not entity_pass:
        drivers.append(("Entity Disconnect", "H1/Entity Mismatch: Brand name is not clearly present in the homepage H1."))
    if trust_driver_fail:
        drivers.append(("Trust Vacuum", "Low Trust Authority: No Tier-1 sameAs links found (Wikidata/Wikipedia/Crunchbase)."))

    if drivers:
        st.markdown("### 🛑 Top Invisibility Drivers")
        for title, desc in drivers:
            st.warning(f"**{title}** — {desc}")
        st.markdown("")

    # ----------------------------
    # LAYER 3: TECHNICAL WATERFALL (EXPANDABLE)
    # ----------------------------
    with st.expander("📋 View Full Technical Diagnostics (Waterfall)", expanded=False):
        st.markdown("#### Infrastructure")
        st.write(f"- **Robots.txt Access:** {passfail(robots_access)}" + ("" if robots_access else f" (error: {robots_err})"))
        st.write(f"- **AI Blocked (GPTBot/CCBot/Google-Extended):** {passfail(not any_ai_blocked)}")
        st.write(f"- **Sitemap Reachability (products discovered):** {passfail(found_products)}")
        st.write(f"- **Render Accessibility (Ghost Code):** {passfail(not ghost_driver)}")

        st.markdown("---")
        st.markdown("#### Schema Layer")
        st.write(f"- **Organization (Identity tag present):** {passfail(org_present)}")
        st.write(f"- **Product (Commerce tags present):** {passfail(product_present)}")
        st.write(f"- **FAQ (Knowledge tags present):** {passfail(faq_present)}")
        st.write(f"- **Commerce Readiness (offers/price):** {passfail(commerce_ready)}")

        st.markdown("---")
        st.markdown("#### Semantic Layer")
        st.write(f"- **Metadata Recency ('2025'/'2026' detected):** {passfail(recency_pass)}")
        st.write(f"- **Entity Clarity (Brand in H1):** {passfail(entity_pass)}")
        st.write(f"- **Authority Nodes (Tier-1 links present):** {passfail(authority_pass)}")

        st.markdown("---")
        st.markdown("#### Page-Level Detail")
        for a in audits:
            with st.expander(f"{a.final_url} — {a.score}/100", expanded=False):
                if not a.ok_fetch:
                    st.error(a.fetch_error or "Fetch failed.")
                    continue
                st.write(f"**Raw Page Size:** {a.raw_kb:.1f} KB")
                st.write(f"**Visible Text (raw HTML):** {a.text_len} characters")
                st.write(f"**Ghost Code:** {passfail(not a.ghost)}")
                st.write("**Schema:**")
                st.write(f"- Organization: {passfail(a.org_found)}")
                st.write(f"- Identity Verified: {passfail(a.identity_verified)}")
                st.write(f"- Product: {passfail(a.product_found)}")
                st.write(f"- Commerce Ready: {passfail(a.commerce_ready)}")
                st.write(f"- FAQ: {passfail(a.faq_found)}")
                if a.warnings:
                    st.markdown("**Alerts:**")
                    for w in a.warnings:
                        st.warning(w)

        with st.expander("Crawler Notes", expanded=False):
            for n in notes:
                st.write(n)
            st.write("**Pages scanned:**")
            for u in scan_urls:
                st.write(f"- {u}")
            st.write("**Homepage signals:**")
            st.write(f"- Title: {title_text[:220] + ('…' if len(title_text) > 220 else '')}")
            st.write(f"- H1: {h1_text[:220] + ('…' if len(h1_text) > 220 else '')}")

    # ----------------------------
    # PHASE 1 / PHASE 2
    # ----------------------------
    st.markdown("---")
    p1, p2 = st.columns([0.55, 0.45])

    with p1:
        st.markdown("### Phase 1 (Defense): Basic Patch")
        st.markdown(
            "<div style='color:#9DA7B3; font-size:13px;'>"
            "Minimum 'Hello' tags to prevent identity + answer failures."
            "</div>",
            unsafe_allow_html=True,
        )
        st.code(organization_jsonld_template(origin, brand), language="json")
        st.code(faqpage_jsonld_template(origin, brand), language="json")

    with p2:
        st.markdown("### 🚀 Phase 2: Vector Space Dominance")
        st.info(
            "To win, you need **Competitor Displacement**, **Sentiment Injection**, and **Programmatic Verification**. "
            "This requires custom engineering."
        )
        st.link_button("👉 Book Your Phase 2 Strategy Call", "https://calendly.com", use_container_width=True)

else:
    st.markdown(
        """
<div style="background:#161b22; border:1px solid #30363D; border-radius:12px; padding:16px;">
  <div style="font-weight:800; font-size:16px;">Run a scan to reveal your AI visibility bottlenecks</div>
  <div style="color:#9DA7B3; margin-top:6px; font-size:13px;">
    We automatically discover product pages (when possible) and diagnose why AI agents ignore your brand.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
