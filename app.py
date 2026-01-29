# app.py — Agentic Infrastructure Audit (PageSpeed-style, clean white UI)
# Requirements:
#   pip install streamlit requests beautifulsoup4 lxml streamlit-extras
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
from streamlit_extras.metric_cards import style_metric_cards


# ----------------------------
# Page config (native, clean, high-contrast)
# ----------------------------
st.set_page_config(page_title="Agentic Infrastructure Audit", page_icon="📈", layout="centered")


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

PRODUCT_HINT_TOKENS = ("/products/", "/product/", "/shop/", "/store/", "/item/")
DISALLOWED_EXTS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".pdf", ".zip", ".rar", ".7z",
    ".mp4", ".mov", ".avi", ".webm",
    ".css", ".js", ".json", ".xml",
)

# For "Trust & Entity": socials are weak vs tier1, but requested is socials/sameAs detection
SOCIAL_DOMAINS = (
    "instagram.com",
    "facebook.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "linkedin.com",
)

# Ghost code threshold
GHOST_TEXT_MIN = 600


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


@dataclass
class SiteAudit:
    origin: str
    brand: str
    homepage_url: str
    scan_urls: List[str]
    notes: List[str]
    pages: List[PageAudit]

    # AI Access insight
    robots_accessible: bool
    robots_error: Optional[str]
    llms_txt_present: bool
    llms_txt_error: Optional[str]

    # Visual semantics insight (aggregated across scanned pages)
    img_total: int
    img_missing_alt: int
    img_missing_alt_examples: List[str]

    # Trust & Entity insight (aggregated across scanned pages)
    org_present_any: bool
    sameas_social_links: List[str]


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
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text


def safe_fetch_text(url: str, timeout: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        final_url, text = fetch_text(url, timeout)
        return final_url, text, None
    except Exception as e:
        return None, None, str(e)


# ----------------------------
# Sitemap helpers
# ----------------------------
def extract_loc_urls_from_xml(xml_text: str) -> List[str]:
    locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml_text, flags=re.IGNORECASE)
    return [normalize_url(u.strip()) for u in locs if (u or "").strip()]


def crawl_sitemap_for_products(sitemap_url: str, origin: str, timeout: int, limit: int = 3) -> Tuple[List[str], List[str]]:
    notes: List[str] = []
    found: List[str] = []
    seen: Set[str] = set()

    final, xml, err = safe_fetch_text(sitemap_url, timeout)
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


def discover_home_and_products(origin: str, timeout: int) -> Tuple[str, List[str], List[str]]:
    notes: List[str] = []

    home_final, home_html = fetch_text(urljoin(origin, "/"), timeout=timeout)
    homepage_url = normalize_url(home_final)

    # Step 1: Shopify Turbo
    turbo_url = urljoin(origin, SHOPIFY_SITEMAP_PRODUCTS_PATH)
    turbo_found, turbo_notes = crawl_sitemap_for_products(turbo_url, origin, timeout, limit=3)
    notes.extend([f"Shopify Turbo: {n}" for n in turbo_notes])
    if turbo_found:
        return homepage_url, turbo_found, notes

    # Step 2: Universal sitemap.xml
    sm_url = urljoin(origin, UNIVERSAL_SITEMAP_PATH)
    sm_found, sm_notes = crawl_sitemap_for_products(sm_url, origin, timeout, limit=3)
    notes.extend([f"Universal Sitemap: {n}" for n in sm_notes])
    if sm_found:
        return homepage_url, sm_found, notes

    # Step 3: Robots -> sitemap lines (lightweight)
    robots_final, robots_text, robots_err = safe_fetch_text(urljoin(origin, "/robots.txt"), timeout)
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

    # Step 4: Scrape homepage links
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
    # remove comments + trailing commas fallback
    no_comments = re.sub(r"//.*?$|/\*.*?\*/", "", raw, flags=re.MULTILINE | re.DOTALL).strip()
    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", no_comments)
    try:
        return json.loads(no_trailing_commas)
    except Exception:
        return None


def extract_jsonld_payloads(html: str) -> List[Any]:
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
                # explode @graph if present
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
    if isinstance(same, list) and any(isinstance(x, str) and x.strip() for x in same):
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
                if isinstance(item, dict) and "price" in item and str(item.get("price", "")).strip():
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


def compute_score(org_found: bool, id_verified: bool, faq_found: bool, prod_found: bool, comm_ready: bool, ghost: bool) -> int:
    # Ghost code forces 0 (render-blocking)
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
def check_robots(origin: str, timeout: int) -> Tuple[bool, Optional[str], Optional[str]]:
    final, txt, err = safe_fetch_text(urljoin(origin, "/robots.txt"), timeout)
    if err or txt is None:
        return False, None, err
    return True, txt, None


def check_llms_txt(origin: str, timeout: int) -> Tuple[bool, Optional[str]]:
    final, txt, err = safe_fetch_text(urljoin(origin, "/llms.txt"), timeout)
    if err or txt is None:
        return False, err
    # treat as present only if it isn't empty and not an HTML page
    if len(txt.strip()) > 10 and "<html" not in txt.lower():
        return True, None
    return False, "llms.txt exists but appears empty or non-text."


def semantic_density(text_len: int, html_len: int) -> float:
    if html_len <= 0:
        return 0.0
    return (float(text_len) / float(html_len)) * 100.0


def extract_h1_text(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    return (h1.get_text(" ", strip=True) if h1 else "").strip()


def schema_types_set(objs: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for o in objs:
        t = o.get("@type")
        types = normalize_schema_type(t)
        for x in types:
            out.add(x)
    # present as canonical-cased-ish
    return out


def extract_social_sameas(org_obj: Dict[str, Any]) -> List[str]:
    same_as = org_obj.get("sameAs")
    links: List[str] = []
    if isinstance(same_as, str) and same_as.strip():
        links = [same_as.strip()]
    elif isinstance(same_as, list):
        links = [x.strip() for x in same_as if isinstance(x, str) and x.strip()]
    socials: List[str] = []
    for l in links:
        h = urlparse(l).netloc.lower().replace("www.", "")
        if any(dom in h for dom in SOCIAL_DOMAINS):
            socials.append(l)
    # unique, keep order
    seen = set()
    out = []
    for s in socials:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def scan_images_missing_alt(soup: BeautifulSoup) -> Tuple[int, int, List[str]]:
    imgs = soup.find_all("img")
    total = len(imgs)
    missing = 0
    missing_names: List[str] = []
    for img in imgs:
        alt = img.get("alt")
        if alt is None or str(alt).strip() == "":
            missing += 1
            src = (img.get("src") or "").strip()
            missing_names.append(extract_filename_from_src(src))
    # top 3 unique
    top: List[str] = []
    seen = set()
    for n in missing_names:
        if n in seen:
            continue
        seen.add(n)
        top.append(n)
        if len(top) >= 3:
            break
    return total, missing, top


# ----------------------------
# Page audit
# ----------------------------
def audit_page(url: str, brand: str, timeout: int) -> PageAudit:
    final, html, err = safe_fetch_text(url, timeout)
    if err or html is None or final is None:
        return PageAudit(
            requested_url=url,
            final_url=url,
            ok_fetch=False,
            fetch_error=err or "Fetch failed.",
            score=0,
            org_found=False,
            identity_verified=False,
            faq_found=False,
            product_found=False,
            commerce_ready=False,
            ghost=False,
            text_len=0,
            html_len=0,
            semantic_density=0.0,
            h1_text="",
            h1_has_brand=False,
            schema_types_found=set(),
        )

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    text_len = len(text)
    html_len = len(html)
    ghost = text_len < GHOST_TEXT_MIN

    payloads = extract_jsonld_payloads(html)
    objs = flatten_jsonld_objects(payloads)

    org_obj = find_first(objs, "Organization")
    product_objs = find_all(objs, "Product")
    faq_objs = find_all(objs, "FAQPage")

    org_found = org_obj is not None
    identity_verified = identity_ok(org_obj) if org_obj else False
    product_found = len(product_objs) > 0
    commerce_ready = any(commerce_ok(p) for p in product_objs) if product_found else False
    faq_found = len(faq_objs) > 0

    score = compute_score(org_found, identity_verified, faq_found, product_found, commerce_ready, ghost)
    h1_text = extract_h1_text(soup)
    h1_has = brand_in_h1(brand, h1_text)
    types_found = schema_types_set(objs)

    return PageAudit(
        requested_url=url,
        final_url=final,
        ok_fetch=True,
        fetch_error=None,
        score=score,
        org_found=org_found,
        identity_verified=identity_verified,
        faq_found=faq_found,
        product_found=product_found,
        commerce_ready=commerce_ready,
        ghost=ghost,
        text_len=text_len,
        html_len=html_len,
        semantic_density=semantic_density(text_len, html_len),
        h1_text=h1_text,
        h1_has_brand=h1_has,
        schema_types_found=types_found,
    )


# ----------------------------
# Site audit runner
# ----------------------------
def run_site_audit(url: str, timeout: int) -> SiteAudit:
    origin = origin_from_url(url)
    brand = infer_brand_name(origin)

    notes: List[str] = []

    # Crawl: discover homepage + products
    home_url, product_urls, crawl_notes = discover_home_and_products(origin, timeout)
    notes.extend(crawl_notes)

    scan_urls = [home_url] + [u for u in product_urls if u and not is_disallowed_asset(u)]
    scan_urls = [normalize_url(u) for u in scan_urls]
    # unique, keep order, limit to 4 pages
    uniq: List[str] = []
    seen: Set[str] = set()
    for u in scan_urls:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
        if len(uniq) >= 4:
            break
    scan_urls = uniq

    # AI access checks
    robots_ok, robots_text, robots_err = check_robots(origin, timeout)
    llms_ok, llms_err = check_llms_txt(origin, timeout)

    # Page audits
    pages: List[PageAudit] = []
    for u in scan_urls:
        pages.append(audit_page(u, brand, timeout))

    # Visual semantics aggregate
    img_total = 0
    img_missing_alt = 0
    missing_examples: List[str] = []
    # Trust & Entity aggregate
    org_present_any = False
    social_sameas_links: List[str] = []

    # For image scan + sameAs socials, we need soups; reuse fetch lightly
    for p in pages:
        if not p.ok_fetch:
            continue
        final, html, err = safe_fetch_text(p.final_url, timeout)
        if err or html is None:
            continue
        soup = BeautifulSoup(html, "lxml")

        t, m, top = scan_images_missing_alt(soup)
        img_total += t
        img_missing_alt += m
        for x in top:
            if x not in missing_examples:
                missing_examples.append(x)
        missing_examples = missing_examples[:3]

        payloads = extract_jsonld_payloads(html)
        objs = flatten_jsonld_objects(payloads)
        org_obj = find_first(objs, "Organization")
        if org_obj:
            org_present_any = True
            for s in extract_social_sameas(org_obj):
                if s not in social_sameas_links:
                    social_sameas_links.append(s)

    return SiteAudit(
        origin=origin,
        brand=brand,
        homepage_url=home_url,
        scan_urls=scan_urls,
        notes=notes,
        pages=pages,
        robots_accessible=robots_ok,
        robots_error=robots_err,
        llms_txt_present=llms_ok,
        llms_txt_error=llms_err,
        img_total=img_total,
        img_missing_alt=img_missing_alt,
        img_missing_alt_examples=missing_examples,
        org_present_any=org_present_any,
        sameas_social_links=social_sameas_links,
    )


# ----------------------------
# Score + Risk logic
# ----------------------------
def site_score(site: SiteAudit) -> int:
    if not site.pages:
        return 0
    return int(round(sum(p.score for p in site.pages) / len(site.pages)))


def revenue_risk_from_gap(your_score: int, comp_score: int) -> str:
    gap = comp_score - your_score
    if gap >= 15:
        return "High"
    if gap >= 5:
        return "Medium"
    return "Low"


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
# UI
# ----------------------------
st.title("Agentic Infrastructure Audit")
st.caption(
    "A high-end AEO/AI visibility scanner with competitor benchmarking. "
    "Crawls your homepage + product pages and runs deep technical checks."
)

with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        target_url = st.text_input("Target URL", placeholder="https://yourbrand.com")
    with c2:
        competitor_url = st.text_input("Competitor URL (Optional)", placeholder="https://competitor.com")

    timeout = st.slider("Timeout (seconds)", min_value=5, max_value=60, value=DEFAULT_TIMEOUT, step=5)

    run = st.button("Run Competitive Audit", type="primary", use_container_width=True)

if not run:
    st.info("Enter a target URL (and optionally a competitor) to run a PageSpeed-style AI visibility audit.")
    st.stop()

target_origin = origin_from_url(target_url)
if not target_origin:
    st.error("Please enter a valid Target URL (e.g., https://example.com).")
    st.stop()

comp_origin = origin_from_url(competitor_url) if competitor_url.strip() else ""

with st.status("Running audit…", expanded=False) as status:
    status.update(label="Auditing target…", state="running")
    target_site = run_site_audit(target_origin, timeout)
    time.sleep(0.05)

    comp_site: Optional[SiteAudit] = None
    if comp_origin:
        status.update(label="Auditing competitor…", state="running")
        comp_site = run_site_audit(comp_origin, timeout)
        time.sleep(0.05)

    status.update(label="Complete.", state="complete")

your_score = site_score(target_site)
comp_score = site_score(comp_site) if comp_site else 0

# ----------------------------
# Metric Cards (streamlit-extras)
# ----------------------------
st.subheader("Competitive Scorecard")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Your Agentic Score", f"{your_score}/100")
with m2:
    if comp_site:
        delta = comp_score - your_score
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        st.metric("Competitor Score", f"{comp_score}/100", f"{arrow} {delta}")
    else:
        st.metric("Competitor Score", "—", "Enter competitor")
with m3:
    if comp_site:
        rr = revenue_risk_from_gap(your_score, comp_score)
        st.metric("Revenue Risk", rr)
    else:
        st.metric("Revenue Risk", "—")
style_metric_cards(border_left_color="#4285F4", border_color="#E6E8EB", box_shadow=True)

if comp_site and comp_score > your_score:
    comp_host = urlparse(comp_site.origin).netloc.replace("www.", "")
    st.warning(f"⚠️ Alert: {comp_host} is outranking you on Authority Signals.")

st.divider()

# ----------------------------
# WordLift-style Insight Grid (2x2)
# ----------------------------
st.subheader("Insight Grid (Deep Tech)")

g1, g2 = st.columns(2)
g3, g4 = st.columns(2)

# Card A: AI Access
with g1:
    with st.container(border=True):
        st.markdown("### AI Access")
        if target_site.robots_accessible:
            st.success("robots.txt reachable.")
        else:
            st.error("robots.txt unreachable. Crawlers may downgrade reliability.")
            if target_site.robots_error:
                st.caption(f"Error: {target_site.robots_error}")

        if target_site.llms_txt_present:
            st.success("llms.txt detected (explicit AI permissioning).")
        else:
            st.error("[Impact: HIGH] llms.txt missing — critical for explicit AI permissioning.")
            st.caption("Why it matters: It clarifies agent access/policy and reduces ambiguity in AI crawling decisions.")

        st.caption("Pages scanned: " + str(len(target_site.pages)))
        st.caption("Discovery notes (summary):")
        if target_site.notes:
            st.write("• " + " • ".join(target_site.notes[:3]))

# Card B: Visual Semantics
with g2:
    with st.container(border=True):
        st.markdown("### Visual Semantics")
        st.write(f"Images scanned: **{target_site.img_total}**")
        if target_site.img_total == 0:
            st.info("No images detected on scanned pages.")
        else:
            if target_site.img_missing_alt > 0:
                st.warning(f"Missing alt text: **{target_site.img_missing_alt}**")
                if target_site.img_missing_alt_examples:
                    st.caption("Proof (filenames): " + ", ".join(target_site.img_missing_alt_examples))
                st.caption("Why it matters: Missing alt reduces AI-readable meaning of visual content.")
            else:
                st.success("All scanned images include alt text.")

# Card C: Semantic Density
with g3:
    with st.container(border=True):
        st.markdown("### Semantic Density")
        densities = [p.semantic_density for p in target_site.pages if p.ok_fetch]
        avg_density = (sum(densities) / len(densities)) if densities else 0.0
        st.write(f"Average semantic density: **{avg_density:.2f}%**")

        if avg_density < 5.0:
            st.error("Bloated Code (<5%): AI struggles to read noise. Reduce markup/scripts and expose more meaningful text.")
        elif avg_density < 10.0:
            st.warning("Low density (<10%): Content-to-code ratio is weak. Consider simplifying templates and improving text signals.")
        else:
            st.success("Healthy semantic ratio (signal > noise).")

        st.caption("Definition: (Visible Text Length / HTML Length) × 100")

# Card D: Trust & Entity
with g4:
    with st.container(border=True):
        st.markdown("### Trust & Entity")
        org_any = any(p.org_found for p in target_site.pages if p.ok_fetch)
        id_any = any(p.identity_verified for p in target_site.pages if p.ok_fetch)
        if org_any:
            st.success("Organization schema detected.")
            if id_any:
                st.success("Identity verified (sameAs or disambiguatingDescription present).")
            else:
                st.warning("Organization present but identity verification is weak (missing sameAs/disambiguatingDescription).")
        else:
            st.error("Organization schema missing (brand entity is ambiguous).")

        if target_site.sameas_social_links:
            st.info("Social profiles detected in sameAs:")
            for s in target_site.sameas_social_links[:3]:
                st.write(f"• {s}")
        else:
            st.warning("No social sameAs links detected (weak entity corroboration).")

        st.caption("Note: Social links help, but Tier-1 sources are stronger for Knowledge Graph trust.")

st.divider()

# ----------------------------
# Detailed Findings (Waterfall)
# ----------------------------
st.subheader("Detailed Page Analysis")

with st.expander("Open Detailed Page Analysis", expanded=False):
    for p in target_site.pages:
        label = f"{p.final_url} — {p.score}/100"
        with st.expander(label, expanded=False):
            if not p.ok_fetch:
                st.error(p.fetch_error or "Fetch failed.")
                continue

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Ghost Code", "YES" if p.ghost else "NO")
            with c2:
                st.metric("Visible Text", str(p.text_len))
            with c3:
                st.metric("Semantic Density", f"{p.semantic_density:.2f}%")
            style_metric_cards(border_left_color="#34A853", border_color="#E6E8EB", box_shadow=False)

            st.markdown("**Checklist**")
            st.write(f"- H1 present: {'✅' if p.h1_text else '❌'}")
            st.write(f"- H1 contains brand: {'✅' if p.h1_has_brand else '❌'}")
            st.write(f"- Organization schema: {'✅' if p.org_found else '❌'}")
            st.write(f"- Identity verified: {'✅' if p.identity_verified else '❌'}")
            st.write(f"- Product schema: {'✅' if p.product_found else '❌'}")
            st.write(f"- Offers/price present: {'✅' if p.commerce_ready else '❌'}")
            st.write(f"- FAQPage schema: {'✅' if p.faq_found else '❌'}")

            if p.ghost:
                st.error("Render Blocking / Ghost Code: Page is 200 OK but exposes <600 readable characters. Fast AI crawlers may see blank HTML.")

            if p.schema_types_found:
                # show a short list of schema types
                types_preview = sorted(list(p.schema_types_found))[:12]
                st.caption("Schema types found (sample): " + ", ".join(types_preview))

st.divider()

# ----------------------------
# Phase 2 Upsell
# ----------------------------
st.subheader("Phase 1 vs Phase 2")

left, right = st.columns([0.55, 0.45])

with left:
    st.markdown("### Phase 1 (Defense): Identity Patch")
    st.caption("If your identity signal is weak, this is the minimum 'Hello' tag to anchor your brand entity.")
    st.code(organization_jsonld_template(target_site.origin, target_site.brand), language="json")

with right:
    st.markdown("### Phase 2 (Offense): Competitor Displacement")
    if comp_site:
        comp_host = urlparse(comp_site.origin).netloc.replace("www.", "")
        st.info(f"To beat {comp_host}, you need Programmatic Verification and Sentiment Injection.")
    else:
        st.info("To win, you need Programmatic Verification and Sentiment Injection.")
    st.caption("This requires a custom AEO engineering strategy (entity reinforcement, content mapping, and trust node acquisition).")
    st.link_button("👉 Book Your Strategy Call", "https://calendly.com", use_container_width=True)
