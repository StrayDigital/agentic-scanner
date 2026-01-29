# app.py — Agentic Visibility Scanner (single-file Streamlit app)
# Dependencies:
#   pip install streamlit requests beautifulsoup4 lxml
# Run:
#   streamlit run app.py

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup


# ----------------------------
# Constants / Config
# ----------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 15

# Skip non-HTML assets
DISALLOWED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".pdf", ".zip", ".rar", ".7z",
    ".mp4", ".mov", ".avi", ".webm",
    ".css", ".js", ".json",
    ".xml",
)

PRODUCT_PATH_TOKEN = "/products/"


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class PageResult:
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
def ensure_url_scheme(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    return u


def get_origin(url: str) -> str:
    u = ensure_url_scheme(url)
    if not u:
        return ""
    p = urlparse(u)
    if not p.netloc:
        return ""
    return f"{p.scheme}://{p.netloc}"


def normalize_url(u: str) -> str:
    # drop fragment; keep query
    p = urlparse(u)
    return p._replace(fragment="").geturl()


def is_internal(u: str, origin: str) -> bool:
    try:
        return urlparse(u).netloc == urlparse(origin).netloc
    except Exception:
        return False


def looks_like_product_url(u: str, origin: str) -> bool:
    if not u:
        return False
    u = normalize_url(u)
    if not is_internal(u, origin):
        return False
    low = u.lower()
    if PRODUCT_PATH_TOKEN not in low:
        return False
    for ext in DISALLOWED_EXTENSIONS:
        if low.endswith(ext):
            return False
    return True


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
# Sitemap / Robots discovery (robust & recursive)
# ----------------------------
def extract_loc_urls(xml_text: str) -> List[str]:
    # Extract <loc> contents; works for urlset and sitemapindex
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml_text, flags=re.IGNORECASE)
    out: List[str] = []
    for loc in locs:
        loc = (loc or "").strip()
        if loc:
            out.append(normalize_url(loc))
    return out


def discover_sitemaps_from_robots(robots_text: str) -> List[str]:
    sitemaps: List[str] = []
    for line in robots_text.splitlines():
        if line.lower().startswith("sitemap:"):
            sm = line.split(":", 1)[1].strip()
            if sm:
                sitemaps.append(sm)
    return sitemaps


def crawl_sitemaps_for_products(
    starting_sitemap_url: str,
    origin: str,
    timeout: int,
    max_product_urls: int = 3,
    max_sitemaps_to_visit: int = 40,
) -> Tuple[List[str], List[str]]:
    """
    Recursively crawl sitemaps. If a <loc> ends with .xml, treat it as a child sitemap and fetch it.
    Return up to max_product_urls product page URLs (never .xml).
    """
    visited: Set[str] = set()
    queue: List[str] = [normalize_url(starting_sitemap_url)]
    product_urls: List[str] = []
    notes: List[str] = []

    while queue and len(visited) < max_sitemaps_to_visit and len(product_urls) < max_product_urls:
        sm_url = queue.pop(0)
        sm_url = normalize_url(sm_url)

        if sm_url in visited:
            continue
        visited.add(sm_url)

        try:
            _, xml_text = fetch_text(sm_url, timeout=timeout)
        except Exception as e:
            notes.append(f"⚠️ Sitemap fetch failed: {sm_url} ({e})")
            continue

        locs = extract_loc_urls(xml_text)

        for loc in locs:
            if len(product_urls) >= max_product_urls:
                break

            # Child sitemap recursion
            if looks_like_sitemap_url(loc, origin):
                if loc not in visited and len(visited) + len(queue) < max_sitemaps_to_visit:
                    queue.append(loc)
                continue

            # Only select likely HTML product pages
            if looks_like_product_url(loc, origin):
                if loc not in product_urls:
                    product_urls.append(loc)

    if product_urls:
        notes.append(f"✅ Found {len(product_urls)} product URL(s) via recursive sitemap crawl")
    else:
        notes.append("⚠️ No product URLs found in sitemap(s)")

    return product_urls, notes


def discover_product_urls(origin: str, timeout: int, status_cb=None) -> Tuple[str, List[str], List[str]]:
    """
    Find homepage + up to 3 product URLs using:
    1) /sitemap.xml (recursive children)
    2) robots.txt -> sitemap URLs (recursive)
    3) homepage link scraping
    """
    notes: List[str] = []
    homepage_url = origin + "/"
    home_final_url = homepage_url
    home_html = ""

    if status_cb:
        status_cb("Fetching homepage…")
    home_final_url, home_html = fetch_text(homepage_url, timeout=timeout)

    # Priority 1: /sitemap.xml (recursive)
    try:
        if status_cb:
            status_cb("Trying /sitemap.xml (recursive)…")
        sitemap_url = urljoin(origin, "/sitemap.xml")
        product_urls, sm_notes = crawl_sitemaps_for_products(
            starting_sitemap_url=sitemap_url,
            origin=origin,
            timeout=timeout,
            max_product_urls=3,
        )
        notes.extend(sm_notes)
        if product_urls:
            return normalize_url(home_final_url), product_urls, notes
    except Exception:
        notes.append("⚠️ /sitemap.xml not accessible")

    # Priority 2: robots.txt -> sitemap URLs (recursive)
    try:
        if status_cb:
            status_cb("Checking /robots.txt for sitemap…")
        _, robots_txt = fetch_text(urljoin(origin, "/robots.txt"), timeout=timeout)
        sitemap_urls = discover_sitemaps_from_robots(robots_txt)
        if sitemap_urls:
            # Try sitemaps from robots in order
            for sm in sitemap_urls:
                if status_cb:
                    status_cb("Crawling sitemap from robots.txt (recursive)…")
                product_urls, sm_notes = crawl_sitemaps_for_products(
                    starting_sitemap_url=sm,
                    origin=origin,
                    timeout=timeout,
                    max_product_urls=3,
                )
                notes.extend(sm_notes)
                if product_urls:
                    notes.append("✅ Using robots.txt sitemap source")
                    return normalize_url(home_final_url), product_urls, notes
            notes.append("⚠️ robots.txt had sitemap(s), but none yielded /products/ pages")
        else:
            notes.append("⚠️ robots.txt did not list any sitemap URLs")
    except Exception:
        notes.append("⚠️ /robots.txt not accessible")

    # Priority 3: Homepage scraping fallback
    if status_cb:
        status_cb("Fail-safe: scraping homepage links…")
    soup = BeautifulSoup(home_html, "lxml")
    candidates: List[str] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_u = normalize_url(urljoin(origin, href))
        if looks_like_product_url(abs_u, origin):
            candidates.append(abs_u)

    # De-dupe while preserving order
    seen: Set[str] = set()
    product_urls: List[str] = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            product_urls.append(u)
        if len(product_urls) >= 3:
            break

    if product_urls:
        notes.append(f"✅ Found {len(product_urls)} product URL(s) by scraping homepage links")
    else:
        notes.append("❌ Could not discover product URLs (site may not use /products/ paths or links are JS-rendered)")

    return normalize_url(home_final_url), product_urls, notes


# ----------------------------
# JSON-LD parsing + schema checks
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


def _try_json_parse(raw: str) -> Optional[Any]:
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Best-effort cleanup: remove comments + trailing commas
    no_comments = re.sub(r"//.*?$|/\*.*?\*/", "", raw, flags=re.MULTILINE | re.DOTALL).strip()
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
                    # Include @graph nodes
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


def find_any(objs: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    return [o for o in objs if has_type(o, target)]


def organization_identity_verified(org_obj: Dict[str, Any]) -> bool:
    dis = org_obj.get("disambiguatingDescription")
    if isinstance(dis, str) and dis.strip():
        return True
    same = org_obj.get("sameAs")
    if isinstance(same, str) and same.strip():
        return True
    if isinstance(same, list) and any(isinstance(x, str) and x.strip() for x in same):
        return True
    return False


def product_commerce_ready(product_obj: Dict[str, Any]) -> bool:
    offers = product_obj.get("offers")
    if isinstance(offers, dict) and offers:
        return True
    if isinstance(offers, list) and len(offers) > 0:
        return True
    return False


# ----------------------------
# Scoring / warnings
# ----------------------------
def compute_score(org_found: bool, identity_ok: bool, faq_found: bool, product_found: bool, commerce_ok: bool) -> int:
    score = 0
    if org_found:
        score += 10
        if identity_ok:
            score += 20
    if faq_found:
        score += 20
    if product_found:
        score += 20
        if commerce_ok:
            score += 30
    return score


def build_warnings(identity_ok: bool, commerce_ok: bool, faq_found: bool, product_found: bool, org_found: bool) -> List[str]:
    warns: List[str] = []
    if org_found and not identity_ok:
        warns.append("⚠️ Identity Risk: AI models may confuse your brand with generic terms.")
    if product_found and not commerce_ok:
        warns.append("❌ Commerce Blocked: Missing Price/Stock data. AI cannot sell your item.")
    if not faq_found:
        warns.append("⚠️ Knowledge Gap: Losing 'Answer' visibility.")
    # Optional extra nudge if org missing (not requested, but helpful and consistent with goal)
    if not org_found:
        warns.append("⚠️ Identity Risk: No Organization entity found. AI may not know who you are.")
    return warns


# ----------------------------
# Page audit
# ----------------------------
def audit_page(url: str, timeout: int) -> PageResult:
    try:
        final_url, html = fetch_text(url, timeout=timeout)
        ok_fetch = True
        fetch_error = None
    except Exception as e:
        return PageResult(
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
            warnings=[f"⚠️ Crawl Failed: If AI/search can’t fetch this page, visibility can drop. ({e})"],
        )

    payloads, script_count = extract_jsonld_payloads(html)
    objs = get_all_objects(payloads)

    org_obj = find_first(objs, "Organization")
    product_objs = find_any(objs, "Product")
    faq_objs = find_any(objs, "FAQPage")

    org_found = org_obj is not None
    product_found = len(product_objs) > 0
    faq_found = len(faq_objs) > 0

    identity_ok = organization_identity_verified(org_obj) if org_obj else False
    commerce_ok = any(product_commerce_ready(p) for p in product_objs) if product_found else False

    score = compute_score(org_found, identity_ok, faq_found, product_found, commerce_ok)
    warnings = build_warnings(identity_ok, commerce_ok, faq_found, product_found, org_found)

    # Parsing risk nudge if scripts exist but nothing parsed
    if script_count > 0 and len(payloads) == 0:
        warnings.append("⚠️ Parsing Risk: JSON-LD tags exist but aren’t valid JSON. Crawlers may ignore them.")

    return PageResult(
        requested_url=url,
        final_url=final_url,
        ok_fetch=ok_fetch,
        fetch_error=fetch_error,
        org_found=org_found,
        identity_verified=identity_ok,
        faq_found=faq_found,
        product_found=product_found,
        commerce_ready=commerce_ok,
        score=score,
        warnings=warnings,
    )


# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Agentic Visibility Scanner", page_icon="🧠", layout="centered")

st.title("🧠 Agentic Visibility Scanner")
st.caption(
    "Scan your Shopify site for **AI Agent readiness** (Identity • Commerce • Knowledge). "
    "We automatically analyze your homepage + 3 product pages."
)

with st.sidebar:
    st.header("Settings")
    timeout = st.slider("Request timeout (seconds)", min_value=5, max_value=60, value=DEFAULT_TIMEOUT, step=5)
    st.markdown("---")
    st.write("Dependencies: streamlit, requests, beautifulsoup4, lxml")

input_url = st.text_input("Shopify Domain or URL", placeholder="https://yourstore.com")
run = st.button("Run Scan", type="primary", use_container_width=True)

if run:
    origin = get_origin(input_url)
    if not origin:
        st.error("Please enter a valid site URL (e.g., https://yourstore.com).")
        st.stop()

    scanned_results: List[PageResult] = []
    discovery_notes: List[str] = []
    scan_urls: List[str] = []

    with st.status("Starting scan…", expanded=True) as status:

        def step(msg: str):
            status.update(label=msg, state="running")

        try:
            step("Discovering pages (homepage + 3 products)…")
            homepage, product_urls, discovery_notes = discover_product_urls(origin, timeout=timeout, status_cb=step)

            scan_urls = [normalize_url(homepage)]
            for pu in product_urls:
                nu = normalize_url(pu)
                if nu not in scan_urls and not nu.lower().endswith(".xml"):
                    scan_urls.append(nu)
            scan_urls = scan_urls[:4]  # homepage + 3

            step(f"Scan list ready: {len(scan_urls)} page(s)…")
            for i, u in enumerate(scan_urls, start=1):
                step(f"Auditing page {i}/{len(scan_urls)}…")
                scanned_results.append(audit_page(u, timeout=timeout))

            status.update(label="Scan complete.", state="complete")

        except Exception as e:
            status.update(label="Scan failed.", state="error")
            st.error(f"⚠️ Crawl Failed: If AI/search can’t reliably traverse your site, visibility can drop. ({e})")
            st.stop()

    st.subheader("Results")

    with st.expander("Crawler Notes"):
        for n in discovery_notes:
            st.write(n)
        st.write("**Pages scanned:**")
        for u in scan_urls:
            st.write(f"- {u}")

    if scanned_results:
        agentic_health_score = round(sum(r.score for r in scanned_results) / len(scanned_results))
    else:
        agentic_health_score = 0

    st.markdown(f"### Agentic Health Score: `{agentic_health_score}/100`")

    # Aggregate warnings (dedupe)
    agg_warns: List[str] = []
    seen_warns: Set[str] = set()
    for r in scanned_results:
        for w in r.warnings:
            if w not in seen_warns:
                seen_warns.add(w)
                agg_warns.append(w)

    # Display sales-trap warnings prominently if score is low or warnings exist
    if agentic_health_score < 80 and agg_warns:
        st.markdown("#### Critical Alerts")
        for w in agg_warns:
            if w.startswith("❌"):
                st.error(w)
            else:
                st.warning(w)
    elif agentic_health_score >= 80:
        st.success("✅ Strong baseline. You’re ahead of most stores on AI-readiness.")
    else:
        st.info("Partial coverage detected. Fixing schema gaps typically lifts AI visibility quickly.")

    st.markdown("---")

    # Per-page breakdown
    def check_line(label: str, ok: bool) -> str:
        return f"{'✅' if ok else '❌'} {label}"

    for r in scanned_results:
        with st.expander(f"{r.final_url} — {r.score}/100", expanded=False):
            if not r.ok_fetch:
                st.error(r.fetch_error or "Fetch failed.")
                continue

            col1, col2 = st.columns(2)
            with col1:
                st.write(check_line("Organization found (+10)", r.org_found))
                st.write(check_line("FAQPage found (+20)", r.faq_found))
                st.write(check_line("Product found (+20)", r.product_found))
            with col2:
                st.write(check_line("Identity Verified (+20)", r.identity_verified))
                st.write(check_line("Commerce Ready / Offers (+30)", r.commerce_ready))

            if r.warnings:
                st.markdown("**Page Alerts:**")
                for w in r.warnings:
                    if w.startswith("❌"):
                        st.error(w)
                    else:
                        st.warning(w)
            else:
                st.success("No critical issues detected on this page.")

    st.markdown("---")
    st.markdown("### Ready to fix this in 15 minutes?")
    st.caption("If your score is low, you’re likely losing AI attribution, ‘answer’ visibility, and automated shopping readiness.")
    st.link_button("👉 Book Your Fix (15 Min)", "https://calendly.com", use_container_width=True)
