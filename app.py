# app.py — Agentic Visibility Scanner (Advanced AEO Audit Tool)
# Required libraries: streamlit, requests, beautifulsoup4, urllib.parse, re, json, time, datetime
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
# Config
# ----------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 15

SHOPIFY_SITEMAP_PRODUCTS_PATH = "/sitemap_products_1.xml"
UNIVERSAL_SITEMAP_PATH = "/sitemap.xml"

PRODUCT_HINT_TOKENS = ("/products/", "/product/", "/shop/", "/store/")
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
    "bloomberg.com",
)

SOCIAL_DOMAINS = (
    "instagram.com",
    "facebook.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
    "linkedin.com",
    "youtube.com",
    "pinterest.com",
)

INDUSTRY_AVERAGE_SCORE = 72


# ----------------------------
# Data structures
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

    # NEW: raw page size (KB) for detailed findings
    raw_size_kb: float


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
# Networking
# ----------------------------
def fetch_text(url: str, timeout: int) -> Tuple[str, str]:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text


# ----------------------------
# Robots.txt AI Blocker check
# ----------------------------
def parse_robots_for_blocks(robots_text: str, target_agents: Tuple[str, ...]) -> Dict[str, bool]:
    """
    Minimal group-aware robots parser:
      - Tracks user-agent(s) for current group
      - If group contains a target agent and has Disallow: / (or /*), mark as blocked
    """
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
# Sitemap parsing (recursive)
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
# Hybrid crawler engine
# ----------------------------
def discover_home_and_products(origin: str, timeout: int, status_cb=None) -> Tuple[str, List[str], List[str]]:
    notes: List[str] = []

    if status_cb:
        status_cb("Fetching homepage…")
    home_final, home_html = fetch_text(urljoin(origin, "/"), timeout=timeout)
    homepage_url = normalize_url(home_final)

    # Step 1: Shopify Turbo
    try:
        if status_cb:
            status_cb("Shopify Turbo: checking /sitemap_products_1.xml…")
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
            return homepage_url, picked, notes

        notes.append("⚠️ Shopify Turbo: sitemap exists but produced 0 valid product URLs")
    except Exception as e:
        notes.append(f"⚠️ Shopify Turbo failed: {e}")

    # Step 2: Universal sitemap.xml recursive
    try:
        if status_cb:
            status_cb("Universal: crawling /sitemap.xml recursively…")
        sm_url = urljoin(origin, UNIVERSAL_SITEMAP_PATH)
        products, sm_notes = crawl_sitemaps_for_products(sm_url, origin=origin, timeout=timeout, max_product_urls=3)
        notes.extend(sm_notes)
        if products:
            return homepage_url, products, notes
    except Exception as e:
        notes.append(f"⚠️ Universal sitemap failed: {e}")

    # Step 3: robots.txt sitemap fallbacks
    try:
        if status_cb:
            status_cb("Fallback: checking robots.txt for sitemap URLs…")
        robots_text, _ = fetch_robots(origin, timeout=timeout)
        if robots_text:
            sm_urls = discover_sitemaps_from_robots(robots_text)
            if sm_urls:
                notes.append(f"ℹ️ robots.txt lists {len(sm_urls)} sitemap URL(s)")
                for sm in sm_urls:
                    if status_cb:
                        status_cb("Fallback: crawling robots.txt sitemap…")
                    products, sm_notes = crawl_sitemaps_for_products(sm, origin=origin, timeout=timeout, max_product_urls=3)
                    notes.extend(sm_notes)
                    if products:
                        notes.append("✅ Product URLs discovered via robots.txt sitemap")
                        return homepage_url, products, notes
            else:
                notes.append("⚠️ robots.txt did not list sitemap URLs")
        else:
            notes.append("⚠️ robots.txt not available for sitemap discovery")
    except Exception as e:
        notes.append(f"⚠️ robots.txt fallback failed: {e}")

    # Step 4: scrape homepage fallback (products/shop/store)
    if status_cb:
        status_cb("Last resort: scanning homepage links for product pages…")
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

    return homepage_url, picked, notes


# ----------------------------
# JSON-LD parsing (robust)
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
# Audit logic (scoring + AEO checks)
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


def authority_tier_from_sameas(same_as: Any) -> Tuple[bool, bool, List[str]]:
    links: List[str] = []
    if isinstance(same_as, str) and same_as.strip():
        links = [same_as.strip()]
    elif isinstance(same_as, list):
        links = [x.strip() for x in same_as if isinstance(x, str) and x.strip()]

    norm_links: List[str] = []
    for l in links:
        if not re.match(r"^https?://", l, flags=re.I):
            continue
        norm_links.append(l)

    if not norm_links:
        return False, False, []

    tier1_present = any(any(dom in host_of_url(l) for dom in AUTH_TIER1_DOMAINS) for l in norm_links)

    all_social = True
    for l in norm_links:
        h = host_of_url(l)
        if not any(sd in h for sd in SOCIAL_DOMAINS):
            all_social = False
            break

    socials_only = all_social and not tier1_present
    return tier1_present, socials_only, norm_links


def build_warnings(
    org_found: bool,
    id_verified: bool,
    prod_found: bool,
    comm_ready: bool,
    faq_found: bool,
    authority_tier1: bool,
    authority_socials_only: bool,
) -> List[str]:
    warnings: List[str] = []
    if not org_found:
        warnings.append("⚠️ Invisible Brand Risk: AI agents cannot find a clear brand entity for your site. You risk being treated as unverified.")
    elif not id_verified:
        warnings.append("⚠️ Invisible Brand Risk: AI agents (like ChatGPT) cannot definitively prove you are a real business. You risk being ignored.")
    if prod_found and not comm_ready:
        warnings.append("❌ Revenue Blocked: AI cannot read your prices or stock levels. You are losing automated sales.")
    if not faq_found:
        warnings.append("⚠️ Silent Treatment: You have no structured answers. When users ask 'What is [Brand]?', AI stays silent or hallucinates.")
    if org_found and not authority_tier1:
        if authority_socials_only:
            warnings.append("⚠️ Weak Authority: Socials are not Knowledge Sources. You need Tier-1 citations (Wikidata/Wikipedia/Crunchbase) to win AEO.")
        else:
            warnings.append("⚠️ Authority Gap: No Tier-1 knowledge sources found in sameAs. AI trusts verified databases more than marketing pages.")
    return warnings


# ----------------------------
# Page audit (NEW: Ghost Code check + raw size KB)
# ----------------------------
def audit_page(url: str, timeout: int) -> PageAudit:
    try:
        final_url, html = fetch_text(url, timeout=timeout)
        ok_fetch = True
        fetch_error = None
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
            warnings=[f"⚠️ Crawl Failed: AI cannot consistently access this page. Visibility and trust signals can collapse. ({e})"],
            raw_size_kb=0.0,
        )

    raw_size_kb = len(html.encode("utf-8")) / 1024.0

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

    score = compute_score(org_found, id_verified, faq_found, prod_found, comm_ready)

    tier1_present = False
    socials_only = False
    if org and "sameAs" in org:
        tier1_present, socials_only, _ = authority_tier_from_sameas(org.get("sameAs"))

    warnings = build_warnings(
        org_found=org_found,
        id_verified=id_verified,
        prod_found=prod_found,
        comm_ready=comm_ready,
        faq_found=faq_found,
        authority_tier1=tier1_present,
        authority_socials_only=socials_only,
    )

    if script_count > 0 and len(payloads) == 0:
        warnings.append("⚠️ Executive Alert: Your site outputs schema, but it’s malformed—AI and Google may ignore it entirely.")

    # ----------------------------
    # NEW: "Ghost Code" check for JS-heavy/headless sites
    # Logic: If score == 0 AND text length < 500 (but fetch was OK), warn about CSR invisibility
    # ----------------------------
    try:
        soup = BeautifulSoup(html, "lxml")
        text_len = len(soup.get_text(" ", strip=True))
    except Exception:
        text_len = 0

    if ok_fetch and score == 0 and text_len < 500:
        warnings.append(
            "⚠️ Headless Architecture Detected: This page relies heavily on JavaScript. While Google renders this, many AI Agents "
            "(which consume raw HTML to save compute) see a blank page. You are invisible to 'Fast' crawlers."
        )

    return PageAudit(
        requested_url=url,
        final_url=final_url,
        ok_fetch=ok_fetch,
        fetch_error=fetch_error,
        org_found=org_found,
        identity_verified=id_verified,
        faq_found=faq_found,
        product_found=prod_found,
        commerce_ready=comm_ready,
        score=score,
        warnings=warnings,
        raw_size_kb=raw_size_kb,
    )


# ----------------------------
# AEO Special: Homepage signals
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


def contains_current_year(text: str) -> bool:
    if not text:
        return False
    year = datetime.now().year
    return str(year) in text


def brand_in_h1(brand: str, h1_text: str) -> bool:
    if not brand or not h1_text:
        return False
    brand_tokens = [t.lower() for t in re.split(r"\s+", brand.strip()) if t.strip()]
    h1_low = h1_text.lower()
    return all(t in h1_low for t in brand_tokens)


# ----------------------------
# UI helpers
# ----------------------------
def pct(n: int, d: int) -> int:
    if d <= 0:
        return 0
    return max(0, min(100, int(round(100 * n / d))))


def simulate_ai_response(
    health_score: int,
    authority_connected: bool,
    commerce_health: int,
    any_blocked: bool,
    stale_risk: bool,
    entity_confusion: bool,
    ghost_risk: bool,
) -> str:
    if any_blocked:
        return (
            "I can’t reliably access or cite this website because AI crawlers appear to be blocked. "
            "That means I cannot confidently verify your brand or products—so I will avoid recommending you."
        )
    if ghost_risk:
        return (
            "This site appears to rely heavily on client-side JavaScript rendering. Many fast AI crawlers see a near-empty page, "
            "so I may not be able to extract product or brand details reliably—meaning I won’t confidently recommend it."
        )
    if health_score >= 85 and authority_connected and commerce_health >= 80 and not stale_risk and not entity_confusion:
        return (
            "I can verify the brand identity through strong knowledge signals and I can read product pricing data. "
            "I would confidently recommend specific in-stock products and cite the brand clearly."
        )
    if commerce_health < 50:
        return (
            "I can see products, but I can’t reliably confirm price/stock. "
            "In an AI shopping moment, I will not push customers to buy because the sale details aren’t verifiable."
        )
    if not authority_connected:
        return (
            "I can’t connect this brand to trusted knowledge sources. "
            "Without Wikidata/Wikipedia-style verification, I will treat the brand as lower-confidence and avoid strong claims."
        )
    if stale_risk:
        return (
            "Your homepage looks stale compared to fresh, updated competitors. "
            "When answers require recency, I will prioritize sources that explicitly signal the current year and updated context."
        )
    if entity_confusion:
        return (
            "Your homepage doesn’t clearly state the brand name in the primary headline. "
            "That increases entity confusion and reduces how confidently I can cite or recommend you."
        )
    if health_score < 60:
        return (
            "I’m not confident I can verify your business and products consistently. "
            "In practice, I give generic answers—and you lose the recommendation."
        )
    return (
        "You have partial AI visibility, but your signals are inconsistent. "
        "With an authority upgrade and structured commerce signals, you can become the default brand AI recommends."
    )


def authority_label(connected: bool) -> str:
    return "✅ Knowledge Graph Connected" if connected else "❌ Knowledge Graph Connected"


# ----------------------------
# Phase 1 snippets (Defense)
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
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Agentic Visibility Scanner", page_icon="🧠", layout="centered")

st.title("🧠 Agentic Visibility Scanner")
st.caption(
    "Advanced AEO audit: checks whether AI agents (ChatGPT, Perplexity) can **verify your brand**, "
    "**answer questions about you**, and **read your product data**."
)

with st.sidebar:
    st.header("Settings")
    timeout = st.slider("Request timeout (seconds)", min_value=5, max_value=60, value=DEFAULT_TIMEOUT, step=5)

site_url = st.text_input("Enter your website URL", placeholder="https://example.com")
run = st.button("Run AEO Audit", type="primary", use_container_width
