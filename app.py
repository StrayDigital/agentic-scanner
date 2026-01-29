# app.py — Agentic Visibility Scanner (Universal + Turbo Hybrid Crawler)
# Required libraries: streamlit, requests, beautifulsoup4, urllib.parse, re, json, time
#
# Install:
#   pip install streamlit requests beautifulsoup4 lxml
# Run:
#   streamlit run app.py

import json
import re
import time
from dataclasses import dataclass
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

# Common product-like path tokens for non-Shopify structures
PRODUCT_HINT_TOKENS = ("/products/", "/product/", "/shop/", "/store/", "/item/", "/items/")

# Disallowed (non-HTML) extensions
DISALLOWED_EXTS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".pdf", ".zip", ".rar", ".7z",
    ".mp4", ".mov", ".avi", ".webm",
    ".css", ".js", ".json", ".xml",
)

AI_BOTS = ("GPTBot", "CCBot", "Google-Extended")


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

        # Remove inline comments
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

        m = re.match(r"(?i)user-agent\s*:\s*(.+)$", line)
        if m:
            agent = m.group(1).strip()
            # If we already saw rules in the current group and we see a new UA, start new group
            if saw_rule_in_group and current_agents:
                new_group()
            current_agents.append(agent)
            continue

        m = re.match(r"(?i)disallow\s*:\s*(.*)$", line)
        if m:
            path = (m.group(1) or "").strip()
            saw_rule_in_group = True

            # Empty path means allowed
            if path == "":
                continue

            blocks_all = path == "/" or path == "/*"
            if not blocks_all:
                continue

            # If any current agent matches target agent, mark blocked
            for ga in current_agents:
                for ta in target_agents:
                    if ga.lower() == ta.lower():
                        blocked[ta] = True
            continue

        # Ignore Allow/Sitemap/etc.

    return blocked


def check_ai_blockers(origin: str, timeout: int) -> Tuple[bool, Dict[str, bool], Optional[str], Optional[str]]:
    """
    Returns:
      any_blocked, per_bot_blocked, robots_text_or_none, error_or_none
    """
    try:
        _, robots = fetch_text(urljoin(origin, "/robots.txt"), timeout=timeout)
    except Exception as e:
        return False, {a: False for a in AI_BOTS}, None, str(e)

    per_bot = parse_robots_for_blocks(robots, AI_BOTS)
    any_blocked = any(per_bot.values())
    return any_blocked, per_bot, robots, None


def discover_sitemaps_from_robots(robots_text: str) -> List[str]:
    sitemaps: List[str] = []
    for line in robots_text.splitlines():
        if line.strip().lower().startswith("sitemap:"):
            sm = line.split(":", 1)[1].strip()
            if sm:
                sitemaps.append(sm)
    return sitemaps


# ----------------------------
# Sitemap parsing (recursive)
# ----------------------------
def extract_loc_urls_from_xml(xml_text: str) -> List[str]:
    # Simple <loc> extraction for both sitemapindex and urlset
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
    max_sitemaps_to_visit: int = 50,
) -> Tuple[List[str], List[str]]:
    """
    Recursively crawl sitemaps:
      - If loc ends with .xml and is internal => treat as child sitemap and fetch it
      - Never add .xml URLs to product list
      - Filter product-like URLs by tokens and exclude assets
    """
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
                # Child sitemap
                if loc not in visited and len(visited) + len(queue) < max_sitemaps_to_visit:
                    queue.append(loc)
                continue

            # HTML page candidate
            if looks_like_product_url(loc, origin) and loc not in found_set:
                found_set.add(loc)
                found_products.append(loc)

    if found_products:
        notes.append(f"✅ Found {len(found_products)} product URL(s) via recursive sitemap crawl")
    else:
        notes.append("⚠️ No product URLs found in sitemap(s)")

    return found_products, notes


# ----------------------------
# Hybrid crawler (Universal + Turbo)
# ----------------------------
def discover_home_and_products(origin: str, timeout: int, status_cb=None) -> Tuple[str, List[str], List[str]]:
    """
    Steps:
      1) Shopify Turbo: /sitemap_products_1.xml
      2) Universal Sitemap: /sitemap.xml recursive children
      3) Robots Fallback: /robots.txt Sitemap: URLs recursive
      4) Scrape Fallback: homepage <a> links containing product-like tokens
    Returns: (homepage_url, product_urls, notes)
    """
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

    # Step 2: Universal sitemap.xml (recursive)
    try:
        if status_cb:
            status_cb("Universal Sitemap: crawling /sitemap.xml recursively…")
        sm_url = urljoin(origin, UNIVERSAL_SITEMAP_PATH)
        products, sm_notes = crawl_sitemaps_for_products(sm_url, origin=origin, timeout=timeout, max_product_urls=3)
        notes.extend(sm_notes)
        if products:
            return homepage_url, products, notes
    except Exception as e:
        notes.append(f"⚠️ Universal Sitemap failed: {e}")

    # Step 3: Robots fallback for sitemap URLs
    try:
        if status_cb:
            status_cb("Robots Fallback: checking /robots.txt for Sitemap URLs…")
        _, robots = fetch_text(urljoin(origin, "/robots.txt"), timeout=timeout)
        sm_urls = discover_sitemaps_from_robots(robots)
        if sm_urls:
            notes.append(f"ℹ️ robots.txt listed {len(sm_urls)} sitemap URL(s)")
            for sm in sm_urls:
                if status_cb:
                    status_cb("Robots Fallback: crawling listed sitemap recursively…")
                products, sm_notes = crawl_sitemaps_for_products(sm, origin=origin, timeout=timeout, max_product_urls=3)
                notes.extend(sm_notes)
                if products:
                    notes.append("✅ Product URLs discovered via robots.txt sitemap")
                    return homepage_url, products, notes
            notes.append("⚠️ robots.txt sitemap(s) found, but none yielded product URLs")
        else:
            notes.append("⚠️ robots.txt did not list any sitemap URLs")
    except Exception as e:
        notes.append(f"⚠️ Robots fallback failed: {e}")

    # Step 4: Scrape fallback
    if status_cb:
        status_cb("Scrape Fallback: scanning homepage links for product-like URLs…")
    soup = BeautifulSoup(home_html, "lxml")
    candidates: List[str] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_u = normalize_url(urljoin(origin, href))
        if looks_like_product_url(abs_u, origin):
            candidates.append(abs_u)

    # De-dupe in order, take up to 3
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
        notes.append(f"✅ Scrape Fallback: found {len(picked)} product URL(s) on homepage")
    else:
        notes.append("❌ Scrape Fallback: no product-like links found on homepage")

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

    # Strip JS-style comments and trailing commas (best-effort)
    no_comments = re.sub(r"//.*?$|/\*.*?\*/", "", raw, flags=re.MULTILINE | re.DOTALL).strip()
    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", no_comments)

    try:
        return json.loads(no_trailing_commas)
    except Exception:
        return None


def split_possible_json_blocks(raw: str) -> List[str]:
    """
    Split multiple JSON objects embedded in a single script tag using bracket matching.
    """
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
                    # Expand @graph nodes
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
# Audit checks + scoring
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


def build_warnings(org_found: bool, id_verified: bool, prod_found: bool, comm_ready: bool, faq_found: bool) -> List[str]:
    warnings: List[str] = []

    # CEO-friendly scary warnings (requested)
    if org_found and not id_verified:
        warnings.append(
            "⚠️ Invisible Brand Risk: AI agents (like ChatGPT) cannot definitively prove you are a real business. You risk being ignored."
        )
    if prod_found and not comm_ready:
        warnings.append(
            "❌ Revenue Blocked: AI cannot read your prices or stock levels. You are losing automated sales."
        )
    if not faq_found:
        warnings.append(
            "⚠️ Silent Treatment: You have no structured answers. When users ask 'What is [Brand]?', AI stays silent or hallucinates."
        )

    # Keep this one (still valuable) if Organization missing entirely
    if not org_found:
        warnings.append(
            "⚠️ Invisible Brand Risk: AI agents cannot find a clear brand entity for your site. You risk being treated like an anonymous storefront."
        )

    return warnings


# ----------------------------
# Page audit
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
            warnings=[f"⚠️ Crawl Failed: If AI/search can’t fetch this page, your visibility and trust signals can collapse. ({e})"],
        )

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
    warnings = build_warnings(org_found, id_verified, prod_found, comm_ready, faq_found)

    if script_count > 0 and len(payloads) == 0:
        warnings.append(
            "⚠️ Executive Alert: Your site is outputting schema, but it’s malformed—AI and Google may ignore it entirely."
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
    )


# ----------------------------
# UI helpers
# ----------------------------
def pct(n: int, d: int) -> int:
    if d <= 0:
        return 0
    return max(0, min(100, int(round(100 * n / d))))


def infer_brand_name_from_domain(origin: str) -> str:
    """
    Simple heuristic: domain label -> title-ish brand.
    example: https://cool-store.com -> "Cool Store"
    """
    host = urlparse(origin).netloc.lower()
    host = host.replace("www.", "")
    label = host.split(":")[0].split(".")[0]
    label = re.sub(r"[^a-z0-9\-]+", "", label)
    parts = [p for p in label.split("-") if p]
    if not parts:
        return "Your Brand"
    return " ".join(p.capitalize() for p in parts)


def simulate_ai_response(health_score: int, identity_health: int, commerce_health: int, any_blocked: bool) -> Tuple[str, str]:
    if any_blocked:
        return (
            "What ChatGPT Will Do",
            "I can’t reliably access or cite this website because AI crawlers appear to be blocked. "
            "That means I cannot confidently verify your brand, your products, or your availability—so I will avoid recommending you.",
        )

    if health_score >= 85 and identity_health >= 80 and commerce_health >= 80:
        return (
            "What ChatGPT Will Do",
            "I can verify your brand entity and I can read product pricing signals. "
            "If a customer asks for recommendations, I can confidently surface specific products and describe them as available.",
        )

    if commerce_health < 50:
        return (
            "What ChatGPT Will Do",
            "I can see products, but I can’t reliably read your price/stock signals. "
            "In an AI shopping moment, that typically means I won’t push customers to buy—because I can’t verify the sale details.",
        )

    if identity_health < 50:
        return (
            "What ChatGPT Will Do",
            "I cannot definitively prove your brand’s identity. "
            "That increases the risk you get ignored, misattributed, or blended into generic results when customers ask about you.",
        )

    if health_score < 60:
        return (
            "What ChatGPT Will Do",
            "I’m not confident I can verify your business or products consistently. "
            "In practice, I give safer, generic answers—and you lose the recommendation.",
        )

    return (
        "What ChatGPT Will Do",
        "I can extract some information, but trust signals are inconsistent. "
        "With a focused upgrade, you can become the obvious, verifiable choice in AI answers and AI shopping flows.",
    )


# ----------------------------
# Dynamic Action Plan (templates)
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
                        "text": f"{brand} is a premium brand available at {domain}. Replace this with your real FAQ answer.",
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
    "An executive-grade audit for AI visibility. We scan your homepage + up to 3 product-like pages to see whether "
    "AI agents can **verify your brand**, **understand your products**, and **confidently recommend you**."
)

with st.sidebar:
    st.header("Settings")
    timeout = st.slider("Request timeout (seconds)", min_value=5, max_value=60, value=DEFAULT_TIMEOUT, step=5)
    st.write("")

site_url = st.text_input("Enter your website URL", placeholder="https://example.com")
run = st.button("Run Agentic Scan", type="primary", use_container_width=True)

if run:
    origin = origin_from_url(site_url)
    if not origin:
        st.error("Please enter a valid URL (e.g., https://example.com).")
        st.stop()

    brand_name = infer_brand_name_from_domain(origin)

    audits: List[PageAudit] = []
    notes: List[str] = []
    scan_urls: List[str] = []

    identity_health = 0
    commerce_health = 0

    with st.status("Initializing…", expanded=True) as status:

        def step(msg: str):
            status.update(label=msg, state="running")

        # AI Blocker check (continue scanning even if blocked)
        step("Checking AI access policies (/robots.txt)…")
        any_blocked, per_bot_blocked, robots_text, robots_err = check_ai_blockers(origin, timeout=timeout)

        if robots_err:
            notes.append(f"⚠️ robots.txt could not be fetched ({robots_err}) — continuing anyway.")
        else:
            blocked_list = [k for k, v in per_bot_blocked.items() if v]
            if blocked_list:
                notes.append(f"🚨 AI access blocked for: {', '.join(blocked_list)}")

        # Hybrid crawler
        step("Discovering high-value pages (homepage + product pages)…")
        try:
            homepage_url, product_urls, crawl_notes = discover_home_and_products(origin, timeout=timeout, status_cb=step)
            notes.extend(crawl_notes)
        except Exception as e:
            status.update(label="Discovery failed.", state="error")
            st.error(f"Discovery failed: {e}")
            st.stop()

        scan_urls = [normalize_url(homepage_url)]
        for u in product_urls:
            nu = normalize_url(u)
            if nu not in scan_urls and not is_disallowed_asset(nu) and not nu.lower().endswith(".xml"):
                scan_urls.append(nu)
        scan_urls = scan_urls[:4]

        if len(scan_urls) < 2:
            notes.append("⚠️ We could only scan your homepage. Product discovery returned 0 URLs.")

        # Audit pages
        step(f"Running AI visibility audit on {len(scan_urls)} page(s)…")
        for i, u in enumerate(scan_urls, start=1):
            step(f"Auditing page {i}/{len(scan_urls)}…")
            audits.append(audit_page(u, timeout=timeout))
            time.sleep(0.05)

        status.update(label="Scan complete.", state="complete")

    # ----------------------------
    # Results
    # ----------------------------
    st.subheader("Executive Summary")

    if any_blocked:
        st.error("🚨 **CRITICAL ERROR: AI BLOCKED** — You have explicitly blocked major AI agents. This is a direct visibility kill-switch.")
        blocked_list = [k for k, v in per_bot_blocked.items() if v]
        if blocked_list:
            st.warning(f"Blocked AI agents: {', '.join(blocked_list)}")
        st.info("We still ran the audit so you can see what your site is missing once access is restored.")

    with st.expander("Scan Details (Pages + Discovery Notes)", expanded=False):
        for n in notes:
            st.write(n)
        st.write("**Pages scanned:**")
        for u in scan_urls:
            st.write(f"- {u}")

    health_score = round(sum(a.score for a in audits) / len(audits)) if audits else 0
    st.markdown(f"### Agentic Health Score: `{health_score}/100`")

    # Aggregate warnings
    agg_warnings: List[str] = []
    seen_warns: Set[str] = set()
    for a in audits:
        for w in a.warnings:
            if w not in seen_warns:
                seen_warns.add(w)
                agg_warnings.append(w)

    if health_score < 80 and agg_warnings:
        st.markdown("#### What’s Costing You Visibility (Right Now)")
        for w in agg_warnings:
            if w.startswith("❌"):
                st.error(w)
            else:
                st.warning(w)
    elif health_score >= 80:
        st.success("✅ Strong position. AI agents can verify you more reliably than most competitors.")
    else:
        st.info("Your signals are inconsistent. This is exactly where competitors steal your AI visibility.")

    # Category health
    identity_pass = sum(1 for a in audits if a.org_found and a.identity_verified)
    identity_health = pct(identity_pass, len(audits))

    product_pages = [a for a in audits if a.product_found]
    commerce_pass = sum(1 for a in product_pages if a.commerce_ready)
    commerce_health = pct(commerce_pass, len(product_pages)) if product_pages else 0

    st.markdown("#### Signal Strength")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Identity Health (Can AI prove you’re real?)")
        st.progress(identity_health)
        st.write(f"**{identity_health}%**")
    with col2:
        st.caption("Commerce Readiness (Can AI confidently ‘sell’ you?)")
        st.progress(commerce_health)
        st.write(f"**{commerce_health}%**")

    # AI Simulation (dramatic)
    st.markdown("---")
    st.markdown("## 🔮 The 'ChatGPT Test'")
    st.caption("If a potential customer asks AI about your brand right now, here is exactly what happens:")
    sim_title, sim_msg = simulate_ai_response(health_score, identity_health, commerce_health, any_blocked=any_blocked)
    st.chat_message("assistant").write(f"**{sim_title}**\n\n{sim_msg}")

    # Detailed Findings per URL (CEO-friendly scorecard strings)
    st.markdown("---")
    st.markdown("## Detailed Scorecard by Page")

    for a in audits:
        with st.expander(f"{a.final_url} — {a.score}/100", expanded=False):
            if not a.ok_fetch:
                st.error(a.fetch_error or "Fetch failed.")
                continue

            c1, c2 = st.columns(2)
            with c1:
                st.write("✅ Brand Entity Signal: DETECTED" if a.org_found else "❌ Brand Entity Signal: MISSING")
                st.write("✅ Answer Engine: ACTIVE" if a.faq_found else "❌ Answer Engine: INACTIVE")
                st.write("✅ Product Intelligence: DETECTED" if a.product_found else "❌ Product Intelligence: MISSING")
            with c2:
                st.write("✅ Authority Verification: SECURE" if a.identity_verified else "❌ Authority Verification: UNVERIFIED")
                st.write("✅ AI Shopping Data: OPTIMIZED" if a.commerce_ready else "❌ AI Shopping Data: NOT READY")

            if a.warnings:
                st.markdown("**Executive Alerts:**")
                for w in a.warnings:
                    if w.startswith("❌"):
                        st.error(w)
                    else:
                        st.warning(w)
            else:
                st.success("No critical blockers detected on this page.")

    # ----------------------------
    # Dynamic Action Plan (CEO-friendly, “magic”)
    # ----------------------------
    st.markdown("---")
    st.markdown("## 🚀 Your Instant Authority Upgrade")

    domain = origin
    brand = brand_name

    identity_low = identity_health < 60
    commerce_low = (commerce_health < 60) if product_pages else True
    knowledge_low = any(not a.faq_found for a in audits)

    if identity_low:
        st.markdown("### 1) Authority Lock-In (Identity)")
        st.write(
            f"We generated this custom **'Authority Code'** specifically for **{brand}**. "
            "Copying this into your site is the fastest way to force AI to recognize you."
        )
        org_snippet = organization_jsonld_template(domain=domain, brand=brand)
        st.code(org_snippet, language="json")
        st.caption("Add this JSON-LD to your homepage (or site-wide). Replace sameAs links with your real social/press profiles for maximum authority.")
    else:
        st.success("✅ Authority looks secure. AI can verify your brand entity across the pages we scanned.")

    if knowledge_low:
        st.markdown("### 2) Become the Default Answer (Knowledge)")
        st.write(
            "Right now, when people ask AI basic questions about your brand, you’re not controlling the answer. "
            "This lightweight schema turns your site into a reliable ‘source of truth’."
        )
        faq_snippet = faqpage_jsonld_template(domain=domain, brand=brand)
        st.code(faq_snippet, language="json")
        st.caption("Replace the example Q&A with real questions customers ask. Even 3–5 FAQs can dramatically improve AI answer visibility.")
    else:
        st.success("✅ Answer Engine is active. We detected FAQPage schema on the pages scanned.")

    if commerce_low:
        st.markdown("### 3) Unlock AI-Powered Revenue (Commerce)")
        st.error("❌ Revenue Blocked: AI cannot read your prices or stock levels. You are losing automated sales.")
        st.info(
            "Commerce schema is inherently complex (variants, currency, availability, sale price). "
            "This usually requires a developer or a dedicated schema app to ensure **every product** publishes clean `offers` data. "
            "Once it’s fixed, AI shopping flows can confidently recommend and route buyers to checkout."
        )
    else:
        st.success("✅ AI Shopping Data looks optimized. Offers with price were detected on product pages.")

    # CTA
    st.markdown("---")
    st.markdown("### Want this fixed fast?")
    st.caption("If you want the shortest path to higher AI visibility (and fewer lost sales), we can implement the upgrade end-to-end.")
    st.link_button("👉 Book Your Fix (15 Min)", "https://calendly.com", use_container_width=True)
