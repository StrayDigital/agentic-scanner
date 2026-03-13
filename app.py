# app.py — Agentic Infrastructure Audit
# pip install streamlit requests beautifulsoup4 lxml plotly anthropic
# streamlit run app.py

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

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

st.set_page_config(
    page_title="AEO Audit",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
  background: #f8f9fb !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
  color: #111827 !important;
  -webkit-font-smoothing: antialiased;
}

[data-testid="stAppViewContainer"] > .main { background: #f8f9fb !important; }

.main .block-container {
  max-width: 1200px !important;
  padding: 0 24px 80px !important;
  margin: 0 auto !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stTextInput"] label {
  font-family: 'Inter', sans-serif !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  color: #374151 !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
}
[data-testid="stTextInput"] input {
  background: #fff !important;
  border: 1.5px solid #D1D5DB !important;
  border-radius: 8px !important;
  color: #111827 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  padding: 10px 14px !important;
  transition: border-color 0.15s !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: #4F46E5 !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
  outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #9CA3AF !important; }

[data-testid="stButton"] > button[kind="primary"] {
  background: #4F46E5 !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 11px 24px !important;
  width: 100% !important;
  transition: background 0.15s !important;
  box-shadow: 0 1px 3px rgba(79,70,229,0.4) !important;
  letter-spacing: -0.01em !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
  background: #4338CA !important;
}

[data-testid="stExpander"] {
  background: #fff !important;
  border: 1.5px solid #E5E7EB !important;
  border-radius: 10px !important;
  margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  color: #111827 !important;
  padding: 14px 18px !important;
}

[data-testid="stCodeBlock"] {
  border: 1.5px solid #E5E7EB !important;
  border-radius: 8px !important;
}

[data-testid="stSpinner"] { color: #4F46E5 !important; }
[data-testid="column"] { padding: 0 6px !important; }

hr {
  border: none !important;
  border-top: 1.5px solid #E5E7EB !important;
  margin: 32px 0 !important;
}
</style>
"""

# ─────────────────────────────────────────
# HTML COMPONENTS
# ─────────────────────────────────────────

def page_header() -> str:
    return """
<div style="background:#fff;border-bottom:1.5px solid #E5E7EB;padding:16px 0;margin-bottom:32px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="width:30px;height:30px;background:#4F46E5;border-radius:7px;
      display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:15px;flex-shrink:0;">A</div>
    <span style="font-family:'Inter',sans-serif;font-size:15px;font-weight:700;color:#111827;letter-spacing:-0.02em;">AEO Audit</span>
    <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:500;color:#6B7280;
      background:#F3F4F6;border-radius:4px;padding:2px 8px;margin-left:2px;">by Agentic Infrastructure</span>
  </div>
</div>
"""

def section_title(title: str, subtitle: str = "") -> str:
    sub = f'<p style="font-family:\'Inter\',sans-serif;font-size:13px;color:#6B7280;margin-top:3px;">{subtitle}</p>' if subtitle else ""
    return f"""
<div style="margin:40px 0 16px;">
  <h2 style="font-family:'Inter',sans-serif;font-size:17px;font-weight:700;color:#111827;letter-spacing:-0.02em;">{title}</h2>
  {sub}
</div>
"""

def score_ring(score: int, label: str, brand: str) -> str:
    if score >= 70:   color, badge, badge_bg = "#059669","Healthy","#D1FAE5"
    elif score >= 40: color, badge, badge_bg = "#D97706","Needs Work","#FDE68A"
    else:             color, badge, badge_bg = "#DC2626","Critical","#FEE2E2"
    c = 2 * 3.14159 * 50
    d = (score / 100) * c
    return f"""
<div style="background:#fff;border:1.5px solid #E5E7EB;border-radius:12px;padding:24px;text-align:center;">
  <p style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;color:#9CA3AF;
     text-transform:uppercase;letter-spacing:0.06em;margin-bottom:18px;">{label}</p>
  <div style="position:relative;width:120px;height:120px;margin:0 auto 16px;">
    <svg viewBox="0 0 120 120" width="120" height="120">
      <circle cx="60" cy="60" r="50" fill="none" stroke="#F3F4F6" stroke-width="8"/>
      <circle cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="8"
        stroke-dasharray="{d:.1f} {c:.1f}" stroke-dashoffset="{c/4:.1f}" stroke-linecap="round"/>
    </svg>
    <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;">
      <span style="font-family:'Inter',sans-serif;font-size:28px;font-weight:800;color:#111827;line-height:1;">{score}</span>
      <span style="font-family:'Inter',sans-serif;font-size:11px;color:#9CA3AF;font-weight:500;">/100</span>
    </div>
  </div>
  <p style="font-family:'Inter',sans-serif;font-size:14px;font-weight:700;color:#111827;margin-bottom:6px;">{brand}</p>
  <span style="display:inline-block;font-family:'Inter',sans-serif;font-size:11px;font-weight:600;
    color:{color};background:{badge_bg};border-radius:20px;padding:3px 10px;">{badge}</span>
</div>
"""

def signal_table_row(label: str, earned: int, maximum: int) -> str:
    pct = int((earned / maximum) * 100) if maximum > 0 else 0
    ok = earned > 0
    dot_c = "#059669" if ok else "#D1D5DB"
    bar_c = "#4F46E5" if ok else "#E5E7EB"
    return f"""
<div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #F9FAFB;">
  <div style="width:7px;height:7px;border-radius:50%;background:{dot_c};flex-shrink:0;"></div>
  <span style="flex:1;font-family:'Inter',sans-serif;font-size:13px;color:#374151;font-weight:500;">{label}</span>
  <div style="width:80px;height:5px;background:#F3F4F6;border-radius:3px;overflow:hidden;">
    <div style="width:{pct}%;height:100%;background:{bar_c};border-radius:3px;"></div>
  </div>
  <span style="font-family:'Inter',sans-serif;font-size:12px;color:#6B7280;width:36px;text-align:right;font-weight:500;">{earned}/{maximum}</span>
</div>
"""

def stat_card(label: str, value: str, status: str, note: str = "") -> str:
    if status == "ok":     dot, vc = "#059669","#065F46"
    elif status == "warn": dot, vc = "#D97706","#92400E"
    elif status == "err":  dot, vc = "#DC2626","#991B1B"
    else:                  dot, vc = "#4F46E5","#1E1B4B"
    note_html = f'<p style="font-family:\'Inter\',sans-serif;font-size:11px;color:#9CA3AF;margin-top:2px;">{note}</p>' if note else ""
    return f"""
<div style="background:#fff;border:1.5px solid #E5E7EB;border-radius:10px;padding:16px;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
    <div style="width:7px;height:7px;border-radius:50%;background:{dot};flex-shrink:0;"></div>
    <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;color:#6B7280;
      text-transform:uppercase;letter-spacing:0.05em;">{label}</span>
  </div>
  <p style="font-family:'Inter',sans-serif;font-size:16px;font-weight:700;color:{vc};">{value}</p>
  {note_html}
</div>
"""

def crawler_badge(name: str, status: str) -> str:
    if status == "Accessible":
        bg, border, tc, dot = "#F0FDF4","#A7F3D0","#065F46","#059669"
    elif status == "Blocked":
        bg, border, tc, dot = "#FEF2F2","#FECACA","#991B1B","#DC2626"
    elif status == "Thin Render":
        bg, border, tc, dot = "#FFFBEB","#FDE68A","#92400E","#D97706"
    else:
        bg, border, tc, dot = "#F9FAFB","#E5E7EB","#6B7280","#9CA3AF"
    return f"""
<div style="background:{bg};border:1.5px solid {border};border-radius:8px;padding:12px 14px;">
  <p style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;color:#9CA3AF;margin-bottom:6px;">{name}</p>
  <div style="display:flex;align-items:center;gap:6px;">
    <div style="width:6px;height:6px;border-radius:50%;background:{dot};flex-shrink:0;"></div>
    <span style="font-family:'Inter',sans-serif;font-size:13px;font-weight:600;color:{tc};">{status}</span>
  </div>
</div>
"""

def authority_badge(name: str, found: bool) -> str:
    if found:
        return f'<span style="display:inline-flex;align-items:center;gap:5px;background:#F0FDF4;border:1.5px solid #A7F3D0;border-radius:6px;padding:5px 10px;font-family:\'Inter\',sans-serif;font-size:12px;font-weight:600;color:#065F46;">✓ {name}</span>'
    return f'<span style="display:inline-flex;align-items:center;gap:5px;background:#F9FAFB;border:1.5px solid #E5E7EB;border-radius:6px;padding:5px 10px;font-family:\'Inter\',sans-serif;font-size:12px;font-weight:500;color:#9CA3AF;">{name}</span>'

def check_row(label: str, ok: bool, detail: str = "") -> str:
    icon = "✓" if ok else "✕"
    icon_c = "#059669" if ok else "#DC2626"
    icon_bg = "#F0FDF4" if ok else "#FEF2F2"
    detail_html = f'<span style="font-family:\'Inter\',sans-serif;font-size:12px;color:#9CA3AF;margin-left:8px;">{detail[:60]}</span>' if detail else ""
    return f"""
<div style="display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid #F9FAFB;">
  <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:700;color:{icon_c};
    background:{icon_bg};width:20px;height:20px;border-radius:4px;
    display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;">{icon}</span>
  <span style="flex:1;font-family:'Inter',sans-serif;font-size:13px;color:#374151;font-weight:500;">{label}</span>
  {detail_html}
</div>
"""

def brief_box(content: str) -> str:
    return f"""
<div style="background:#fff;border:1.5px solid #E5E7EB;border-left:3px solid #4F46E5;
  border-radius:8px;padding:20px 24px;margin-top:12px;
  font-family:'Inter',sans-serif;font-size:14px;color:#374151;line-height:1.8;white-space:pre-wrap;">{content}</div>
"""

def locked_brief_box() -> str:
    return """
<div style="background:#fff;border:1.5px dashed #E5E7EB;border-radius:8px;
  padding:20px 24px;margin-top:12px;text-align:center;">
  <p style="font-family:'Inter',sans-serif;font-size:13px;color:#9CA3AF;margin-bottom:4px;">
    Add a Claude API key to unlock an AI-written diagnosis and fix plan.
  </p>
  <a href="https://console.anthropic.com" style="font-family:'Inter',sans-serif;font-size:13px;color:#4F46E5;font-weight:600;text-decoration:none;">
    Get a key at console.anthropic.com →
  </a>
</div>
"""

def cta_section() -> str:
    return """
<div style="background:#4F46E5;border-radius:12px;padding:40px;text-align:center;margin-top:8px;">
  <p style="font-family:'Inter',sans-serif;font-size:12px;font-weight:600;
     color:rgba(255,255,255,0.6);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">Ready to fix it?</p>
  <h3 style="font-family:'Inter',sans-serif;font-size:26px;font-weight:800;
     color:#fff;letter-spacing:-0.03em;margin-bottom:10px;">Get expert implementation</h3>
  <p style="font-family:'Inter',sans-serif;font-size:14px;color:rgba(255,255,255,0.7);
     max-width:400px;margin:0 auto;line-height:1.6;">
    We'll implement every signal, fix your schema, and deploy llms.txt — guaranteed to raise your score.
  </p>
</div>
"""

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
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
    "instagram.com", "facebook.com", "tiktok.com",
    "x.com", "twitter.com", "youtube.com", "linkedin.com",
)
AI_CRAWLERS = {
    "GPTBot":          "Mozilla/5.0 (compatible; GPTBot/1.0; +https://openai.com/gptbot)",
    "Claude-Web":      "Claude-Web/1.0",
    "CCBot":           "CCBot/2.0",
    "Google-Extended": "Google-Extended",
    "PerplexityBot":   "PerplexityBot/1.0",
}
AUTHORITY_DOMAINS = {
    "Wikipedia": "wikipedia.org",
    "Wikidata":  "wikidata.org",
    "Crunchbase":"crunchbase.com",
    "LinkedIn":  "linkedin.com",
    "YouTube":   "youtube.com",
    "X":         "x.com",
    "Twitter":   "twitter.com",
    "GitHub":    "github.com",
}
GHOST_TEXT_MIN = 600
INDUSTRY_AVERAGE_SCORE = 72


# ─────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# URL HELPERS
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# NETWORKING
# ─────────────────────────────────────────
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


def safe_fetch_text(
    url: str, timeout: int
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        final_url, text = fetch_text(url, timeout)
        return final_url, text, None
    except Exception as e:
        return None, None, str(e)


def fetch_with_ua(url: str, user_agent: str, timeout: int) -> Dict[str, str]:
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


# ─────────────────────────────────────────
# SITEMAP HELPERS
# ─────────────────────────────────────────
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
        notes.append(f"Sitemap fetch failed: {sitemap_url}")
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
        notes.append(f"Sitemap discovered {len(found)} product URL(s)")
    else:
        notes.append("Sitemap returned 0 product URLs")
    return found, notes


def discover_from_homepage_scrape(
    origin: str, home_html: str, limit: int = 3
) -> List[str]:
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
    notes.extend(turbo_notes)
    if turbo_found:
        return homepage_url, turbo_found, notes
    sm_url = urljoin(origin, UNIVERSAL_SITEMAP_PATH)
    sm_found, sm_notes = crawl_sitemap_for_products(sm_url, origin, timeout, limit=3)
    notes.extend(sm_notes)
    if sm_found:
        return homepage_url, sm_found, notes
    _, robots_text, _ = safe_fetch_text(urljoin(origin, "/robots.txt"), timeout)
    if robots_text:
        sitemap_lines = []
        for line in robots_text.splitlines():
            if line.strip().lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                if sm:
                    sitemap_lines.append(sm)
        if sitemap_lines:
            for sm in sitemap_lines:
                found, smn = crawl_sitemap_for_products(sm, origin, timeout, limit=3)
                notes.extend(smn)
                if found:
                    return homepage_url, found, notes
    scrape_found = discover_from_homepage_scrape(origin, home_html, limit=3)
    if scrape_found:
        notes.append(f"Homepage scrape: {len(scrape_found)} product-like URL(s)")
    return homepage_url, scrape_found, notes


# ─────────────────────────────────────────
# JSON-LD HELPERS
# ─────────────────────────────────────────
def try_parse_json(raw: str) -> Optional[Any]:
    try:
        return json.loads(raw)
    except Exception:
        pass
    no_comments = re.sub(r"//.*?$|/\*.*?\*/", "", raw, flags=re.MULTILINE | re.DOTALL).strip()
    no_trailing = re.sub(r",\s*([}\]])", r"\1", no_comments)
    try:
        return json.loads(no_trailing)
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


# ─────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────
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
    def _check(d: Dict[str, Any]) -> bool:
        if "price" in d and str(d.get("price", "")).strip():
            return True
        ps = d.get("priceSpecification")
        if isinstance(ps, dict) and "price" in ps:
            return True
        if isinstance(ps, list):
            return any(isinstance(i, dict) and "price" in i for i in ps)
        return False
    if isinstance(offers, dict):
        return _check(offers)
    if isinstance(offers, list):
        return any(isinstance(o, dict) and _check(o) for o in offers)
    return False


def commerce_ok(product_obj: Dict[str, Any]) -> bool:
    offers = product_obj.get("offers")
    if offers is None:
        return False
    return offers_has_price(offers)


def compute_score(
    org_found: bool, id_verified: bool, faq_found: bool,
    prod_found: bool, comm_ready: bool, ghost: bool
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


# ─────────────────────────────────────────
# TECHNICAL CHECKS
# ─────────────────────────────────────────
def check_robots(origin: str, timeout: int) -> Tuple[bool, Optional[str], Optional[str]]:
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
    return False, "Exists but appears empty or non-text."


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
        for x in normalize_schema_type(o.get("@type")):
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
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def compute_extractability(
    html: str, soup: BeautifulSoup, text: str
) -> Tuple[int, str, List[str]]:
    reasons: List[str] = []
    score = 0
    text_len = len(text.strip())
    if text_len >= 1000:
        score += 20
        reasons.append("Rich readable text (1000+ chars)")
    elif text_len >= 400:
        score += 10
        reasons.append("Moderate text length")
    else:
        reasons.append("Thin text content")
    if soup.find("title") and soup.find("title").get_text(strip=True):
        score += 10
        reasons.append("Title tag present")
    else:
        reasons.append("Missing title tag")
    if soup.find("h1") and soup.find("h1").get_text(strip=True):
        score += 10
        reasons.append("H1 tag present")
    else:
        reasons.append("Missing H1 tag")
    lower_text = text.lower()
    if any(kw in lower_text for kw in ["faq", "frequently asked", "question", "answer"]):
        score += 15
        reasons.append("FAQ-style content detected")
    else:
        reasons.append("No FAQ patterns found")
    if any(kw in lower_text for kw in ["privacy", "terms", "about us", "return policy", "shipping"]):
        score += 10
        reasons.append("Policy/trust content detected")
    else:
        reasons.append("No policy content found")
    if any(kw in lower_text for kw in ["price", "buy", "add to cart", "checkout", "shop", "order"]):
        score += 15
        reasons.append("Product/commerce signals found")
    else:
        reasons.append("No product signals detected")
    paras = soup.find_all("p")
    if len(paras) >= 5:
        score += 10
        reasons.append(f"{len(paras)} paragraph tags")
    else:
        reasons.append(f"Only {len(paras)} paragraph tag(s)")
    headings = soup.find_all(["h2", "h3"])
    if len(headings) >= 3:
        score += 10
        reasons.append(f"{len(headings)} sub-headings")
    else:
        reasons.append(f"Only {len(headings)} sub-heading(s)")
    score = min(100, score)
    if score >= 70:
        verdict = "High AI Extractability"
    elif score >= 40:
        verdict = "Moderate AI Extractability"
    else:
        verdict = "Low AI Extractability"
    return score, verdict, reasons


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
    score = int((hits / len(AUTHORITY_DOMAINS)) * 100)
    if score >= 60:
        verdict = "Strong knowledge graph presence"
    elif score >= 30:
        verdict = "Moderate knowledge graph presence"
    else:
        verdict = "Weak knowledge graph presence"
    return score, verdict, authority_results, authority_links


def scan_images(soup: BeautifulSoup) -> Tuple[int, int, List[str]]:
    imgs = soup.find_all("img")
    total = len(imgs)
    missing: List[str] = []
    for img in imgs:
        alt = img.get("alt")
        if alt is None or (isinstance(alt, str) and not alt.strip()):
            missing.append(extract_filename_from_src(img.get("src", "")))
    return total, len(missing), missing[:5]


def simulate_ai_crawlers(url: str, timeout: int) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    for name, ua in AI_CRAWLERS.items():
        results[name] = fetch_with_ua(url, ua, timeout)
        time.sleep(0.25)
    return results


# ─────────────────────────────────────────
# PAGE AUDIT
# ─────────────────────────────────────────
def audit_page(url: str, brand: str, timeout: int) -> PageAudit:
    final_url, html, err = safe_fetch_text(url, timeout)
    if err or not html:
        return PageAudit(
            requested_url=url, final_url=url, ok_fetch=False,
            fetch_error=err or "Empty response", score=0,
            org_found=False, identity_verified=False, faq_found=False,
            product_found=False, commerce_ready=False, ghost=True,
            text_len=0, html_len=0, semantic_density=0.0,
            h1_text="", h1_has_brand=False, schema_types_found=set(),
            extractability_score=0, extractability_verdict="Low AI Extractability",
            extractability_reasons=["Page could not be fetched"],
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
    h1_has_brand_val = brand_in_h1(brand, h1_text)
    types_found = schema_types_set(objs)
    density = calc_semantic_density(text_len, html_len)
    score = compute_score(org_found, id_verified, faq_found, product_found, commerce_ready, ghost)
    ext_score, ext_verdict, ext_reasons = compute_extractability(html, soup, text)
    return PageAudit(
        requested_url=url, final_url=final_url or url, ok_fetch=True,
        fetch_error=None, score=score, org_found=org_found,
        identity_verified=id_verified, faq_found=faq_found,
        product_found=product_found, commerce_ready=commerce_ready,
        ghost=ghost, text_len=text_len, html_len=html_len,
        semantic_density=density, h1_text=h1_text,
        h1_has_brand=h1_has_brand_val, schema_types_found=types_found,
        extractability_score=ext_score, extractability_verdict=ext_verdict,
        extractability_reasons=ext_reasons,
    )


# ─────────────────────────────────────────
# FULL SITE AUDIT
# ─────────────────────────────────────────
def run_site_audit(raw_url: str, timeout: int = DEFAULT_TIMEOUT) -> SiteAudit:
    origin = origin_from_url(raw_url)
    brand = infer_brand_name(origin)
    homepage_url, product_urls, discovery_notes = discover_home_and_products(origin, timeout)
    scan_urls = [homepage_url] + product_urls[:3]
    pages: List[PageAudit] = []
    for u in scan_urls:
        pages.append(audit_page(u, brand, timeout))
    robots_ok, _, robots_err = check_robots(origin, timeout)
    llms_ok, llms_err = check_llms_txt(origin, timeout)
    _, home_html = fetch_text(homepage_url, timeout)
    home_soup = BeautifulSoup(home_html, "lxml") if home_html else BeautifulSoup("", "lxml")
    img_total, img_missing_alt, img_examples = scan_images(home_soup)
    all_objs: List[Dict[str, Any]] = []
    for p in pages:
        if p.ok_fetch:
            _, html, _ = safe_fetch_text(p.final_url, timeout)
            if html:
                all_objs.extend(flatten_jsonld_objects(extract_jsonld_payloads(html)))
    org_present_any = any(
        has_type(o, "organization") or has_type(o, "localbusiness") for o in all_objs
    )
    sameas_social_links: List[str] = []
    seen_s: Set[str] = set()
    for o in all_objs:
        if has_type(o, "organization") or has_type(o, "localbusiness"):
            for s in extract_social_sameas(o):
                if s not in seen_s:
                    seen_s.add(s)
                    sameas_social_links.append(s)
    ai_crawler_results = simulate_ai_crawlers(homepage_url, timeout)
    authority_score, authority_verdict, authority_results, authority_links = compute_authority(all_objs)
    return SiteAudit(
        origin=origin, brand=brand, homepage_url=homepage_url,
        scan_urls=scan_urls, notes=discovery_notes, pages=pages,
        robots_accessible=robots_ok, robots_error=robots_err,
        llms_txt_present=llms_ok, llms_txt_error=llms_err,
        img_total=img_total, img_missing_alt=img_missing_alt,
        img_missing_alt_examples=img_examples,
        org_present_any=org_present_any, sameas_social_links=sameas_social_links,
        ai_crawler_results=ai_crawler_results,
        authority_score=authority_score, authority_verdict=authority_verdict,
        authority_results=authority_results, authority_links=authority_links,
    )


# ─────────────────────────────────────────
# CLAUDE API
# ─────────────────────────────────────────
def generate_ai_brief(audit: SiteAudit, competitor_audit: Optional[SiteAudit] = None) -> str:
    if not ANTHROPIC_AVAILABLE:
        return "Install the 'anthropic' package to enable AI-written fix briefs."
    api_key = st.session_state.get("anthropic_api_key", "")
    if not api_key:
        return ""
    hp = audit.pages[0] if audit.pages else None
    if not hp:
        return ""
    missing_signals = []
    if not hp.org_found:
        missing_signals.append("Organization schema (missing entirely)")
    elif not hp.identity_verified:
        missing_signals.append("Identity verification (no sameAs or disambiguatingDescription)")
    if not hp.faq_found:
        missing_signals.append("FAQPage schema")
    if not hp.product_found:
        missing_signals.append("Product schema")
    elif not hp.commerce_ready:
        missing_signals.append("Offers/Price in Product schema")
    if not audit.llms_txt_present:
        missing_signals.append("llms.txt file")
    if not audit.robots_accessible:
        missing_signals.append("robots.txt accessibility")
    if audit.img_missing_alt > 0:
        missing_signals.append(f"Alt text on {audit.img_missing_alt} images")
    competitor_context = ""
    if competitor_audit:
        comp_hp = competitor_audit.pages[0] if competitor_audit.pages else None
        if comp_hp:
            comp_score = max(p.score for p in competitor_audit.pages)
            target_score = max(p.score for p in audit.pages)
            gap = target_score - comp_score
            competitor_context = f"""
Competitor: {competitor_audit.brand} (score: {comp_score}/100)
Score gap: {'+' if gap >= 0 else ''}{gap} points vs competitor
Competitor has FAQPage: {comp_hp.faq_found}
Competitor has Product schema: {comp_hp.product_found}
Competitor has commerce-ready schema: {comp_hp.commerce_ready}
"""
    prompt = f"""You are an expert AEO (Answer Engine Optimization) consultant. Write a concise, actionable fix brief for this site audit.

Site: {audit.brand} ({audit.origin})
AEO Score: {max(p.score for p in audit.pages)}/100
Ghost code detected: {hp.ghost}
Semantic density: {hp.semantic_density:.1f}%
AI extractability: {hp.extractability_score}/100 ({hp.extractability_verdict})
Authority score: {audit.authority_score}/100 ({audit.authority_verdict})
Missing signals: {', '.join(missing_signals) if missing_signals else 'None'}
{competitor_context}

Write a 3-paragraph brief:
1. One-sentence overall diagnosis
2. Top 3 specific fixes in priority order with exact implementation steps
3. Expected score impact if fixes are implemented

Be direct, technical, and specific. No fluff. Use plain text, no markdown symbols."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"API error: {str(e)}"


def build_org_jsonld(brand: str, origin: str) -> str:
    return json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "Organization",
            "name": brand,
            "url": origin,
            "disambiguatingDescription": f"{brand} — describe your brand here.",
            "sameAs": [
                "https://www.linkedin.com/company/your-company",
                "https://www.instagram.com/your-handle",
                "https://en.wikipedia.org/wiki/Your_Page",
            ],
        },
        indent=2,
    )


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(page_header(), unsafe_allow_html=True)

    col_t, col_c, col_key, col_btn = st.columns([3, 3, 2, 1])
    with col_t:
        target_input = st.text_input("Your URL", placeholder="https://yoursite.com", key="target")
    with col_c:
        competitor_input = st.text_input("Competitor URL", placeholder="https://competitor.com (optional)", key="competitor")
    with col_key:
        api_key_input = st.text_input("Claude API Key", placeholder="sk-ant-...", type="password", key="api_key_field")
        if api_key_input:
            st.session_state["anthropic_api_key"] = api_key_input
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("Run Audit →", type="primary", key="run")

    if not run_btn:
        st.markdown("""
<div style="margin-top:48px;background:#fff;border:1.5px solid #E5E7EB;border-radius:12px;padding:48px;text-align:center;">
  <p style="font-family:'Inter',sans-serif;font-size:15px;font-weight:600;color:#374151;margin-bottom:6px;">Enter a URL above and run your audit</p>
  <p style="font-family:'Inter',sans-serif;font-size:13px;color:#9CA3AF;">
    Checks structured data, AI crawler access, entity signals, and knowledge graph presence.
  </p>
</div>
""", unsafe_allow_html=True)
        return

    target_url = ensure_scheme(target_input.strip())
    if not target_url:
        st.error("Please enter a valid URL.")
        return

    competitor_url = ensure_scheme(competitor_input.strip()) if competitor_input.strip() else None

    with st.spinner("Scanning…"):
        target_audit = run_site_audit(target_url)

    competitor_audit: Optional[SiteAudit] = None
    if competitor_url:
        with st.spinner("Scanning competitor…"):
            competitor_audit = run_site_audit(competitor_url)

    target_score = max(p.score for p in target_audit.pages) if target_audit.pages else 0
    comp_score   = max(p.score for p in competitor_audit.pages) if competitor_audit and competitor_audit.pages else None
    hp = target_audit.pages[0] if target_audit.pages else None

    # ── Scores ──
    st.markdown(section_title("AEO Score", "Overall AI search readiness"), unsafe_allow_html=True)

    if competitor_audit and comp_score is not None:
        sc1, sc2, sc3 = st.columns([1, 1, 2])
    else:
        sc1, sc3 = st.columns([1, 2])

    with sc1:
        st.markdown(score_ring(target_score, "Your Score", target_audit.brand), unsafe_allow_html=True)
    if competitor_audit and comp_score is not None:
        with sc2:
            st.markdown(score_ring(comp_score, "Competitor", competitor_audit.brand), unsafe_allow_html=True)

    with sc3:
        st.markdown("""
<div style="background:#fff;border:1.5px solid #E5E7EB;border-radius:12px;padding:22px 24px;">
  <p style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;color:#9CA3AF;
     text-transform:uppercase;letter-spacing:0.06em;margin-bottom:14px;">Signal Breakdown</p>
""", unsafe_allow_html=True)
        if hp:
            st.markdown(signal_table_row("Organization Schema",   10 if hp.org_found        else 0, 10), unsafe_allow_html=True)
            st.markdown(signal_table_row("Identity Verification", 20 if hp.identity_verified else 0, 20), unsafe_allow_html=True)
            st.markdown(signal_table_row("FAQPage Schema",        20 if hp.faq_found         else 0, 20), unsafe_allow_html=True)
            st.markdown(signal_table_row("Product Schema",        20 if hp.product_found     else 0, 20), unsafe_allow_html=True)
            st.markdown(signal_table_row("Offers / Price",        30 if hp.commerce_ready    else 0, 30), unsafe_allow_html=True)
        delta = target_score - INDUSTRY_AVERAGE_SCORE
        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        delta_color = "#059669" if delta >= 0 else "#DC2626"
        st.markdown(f"""
  <div style="margin-top:14px;padding-top:14px;border-top:1px solid #F3F4F6;
    display:flex;justify-content:space-between;align-items:center;">
    <span style="font-family:'Inter',sans-serif;font-size:12px;color:#9CA3AF;font-weight:500;">vs. industry avg ({INDUSTRY_AVERAGE_SCORE})</span>
    <span style="font-family:'Inter',sans-serif;font-size:14px;font-weight:700;color:{delta_color};">{delta_str} pts</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── AI Brief ──
    st.markdown(section_title("AI Fix Brief", "Claude-generated diagnosis and action plan"), unsafe_allow_html=True)
    if st.session_state.get("anthropic_api_key"):
        with st.spinner("Generating brief…"):
            brief = generate_ai_brief(target_audit, competitor_audit)
        if brief:
            st.markdown(brief_box(brief), unsafe_allow_html=True)
    else:
        st.markdown(locked_brief_box(), unsafe_allow_html=True)

    # ── Site Signals ──
    st.markdown(section_title("Site Signals", "Key technical checks"), unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        if target_audit.robots_accessible and target_audit.llms_txt_present:
            st.markdown(stat_card("AI Access", "Fully open", "ok", "robots.txt + llms.txt"), unsafe_allow_html=True)
        elif target_audit.robots_accessible:
            st.markdown(stat_card("AI Access", "Partial", "warn", "llms.txt missing"), unsafe_allow_html=True)
        else:
            st.markdown(stat_card("AI Access", "Blocked", "err", "robots.txt inaccessible"), unsafe_allow_html=True)
    with s2:
        if target_audit.img_missing_alt == 0 and target_audit.img_total > 0:
            st.markdown(stat_card("Image Alt Text", f"All {target_audit.img_total} tagged", "ok"), unsafe_allow_html=True)
        elif target_audit.img_missing_alt > 0:
            pct = int((target_audit.img_missing_alt / max(target_audit.img_total, 1)) * 100)
            st.markdown(stat_card("Image Alt Text", f"{pct}% missing", "warn", f"{target_audit.img_missing_alt} of {target_audit.img_total}"), unsafe_allow_html=True)
        else:
            st.markdown(stat_card("Image Alt Text", "No images found", "info"), unsafe_allow_html=True)
    with s3:
        if hp:
            d = hp.semantic_density
            s = "ok" if d >= 10 else "warn" if d >= 5 else "err"
            n = "Good ratio" if d >= 10 else "Weak ratio" if d >= 5 else "Bloated HTML"
            st.markdown(stat_card("Semantic Density", f"{d:.1f}%", s, n), unsafe_allow_html=True)
    with s4:
        if target_audit.org_present_any and len(target_audit.sameas_social_links) >= 2:
            st.markdown(stat_card("Entity", "Verified", "ok", f"{len(target_audit.sameas_social_links)} sameAs links"), unsafe_allow_html=True)
        elif target_audit.org_present_any:
            st.markdown(stat_card("Entity", "Partial", "warn", "No sameAs links"), unsafe_allow_html=True)
        else:
            st.markdown(stat_card("Entity", "Not found", "err", "Organization schema missing"), unsafe_allow_html=True)

    # ── Crawlers ──
    st.markdown(section_title("AI Crawler Access", f"Simulated test — {target_audit.homepage_url}"), unsafe_allow_html=True)
    cc = st.columns(5)
    for idx, (name, result) in enumerate(target_audit.ai_crawler_results.items()):
        with cc[idx]:
            st.markdown(crawler_badge(name, result.get("status", "Unknown")), unsafe_allow_html=True)

    # ── Authority ──
    st.markdown(section_title("Knowledge Graph Authority", target_audit.authority_verdict.capitalize()), unsafe_allow_html=True)
    al, ar = st.columns([1, 2])
    with al:
        st.markdown(score_ring(target_audit.authority_score, "Authority Score", "Knowledge Graph"), unsafe_allow_html=True)
    with ar:
        st.markdown('<div style="display:flex;flex-wrap:wrap;gap:8px;padding-top:4px;">', unsafe_allow_html=True)
        for name, found in target_audit.authority_results.items():
            st.markdown(authority_badge(name, found), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if target_audit.authority_links:
            st.markdown('<div style="margin-top:16px;">', unsafe_allow_html=True)
            for link in target_audit.authority_links[:4]:
                st.markdown(f'<a href="{link}" style="display:block;font-family:\'Inter\',sans-serif;font-size:12px;color:#4F46E5;text-decoration:none;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{link}</a>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Page Analysis ──
    st.markdown(section_title("Page Analysis", "Per-page audit results"), unsafe_allow_html=True)
    for i, page in enumerate(target_audit.pages):
        label = "Homepage" if i == 0 else f"Product Page {i}"
        url_display = page.requested_url[:70] + ("…" if len(page.requested_url) > 70 else "")
        with st.expander(f"{label} — {url_display}", expanded=(i == 0)):
            if not page.ok_fetch:
                st.markdown(f'<p style="font-family:\'Inter\',sans-serif;font-size:13px;color:#DC2626;">Fetch failed: {page.fetch_error}</p>', unsafe_allow_html=True)
                continue
            pa1, pa2 = st.columns(2)
            with pa1:
                sc = "#059669" if page.score >= 70 else "#D97706" if page.score >= 40 else "#DC2626"
                st.markdown(f'<p style="font-family:\'Inter\',sans-serif;font-size:32px;font-weight:800;color:{sc};margin-bottom:16px;letter-spacing:-0.03em;">{page.score}<span style="font-size:16px;color:#9CA3AF;font-weight:500;"> / 100</span></p>', unsafe_allow_html=True)
                for lbl, val, det in [
                    ("Ghost code clear",    not page.ghost,          "JS render blocking" if page.ghost else "Server-rendered"),
                    ("H1 tag present",      bool(page.h1_text),      page.h1_text[:50] if page.h1_text else ""),
                    ("Brand in H1",         page.h1_has_brand,       ""),
                    ("Organization schema", page.org_found,          ""),
                    ("Identity verified",   page.identity_verified,  ""),
                    ("FAQPage schema",      page.faq_found,          ""),
                    ("Product schema",      page.product_found,      ""),
                    ("Offers / Price data", page.commerce_ready,     ""),
                ]:
                    st.markdown(check_row(lbl, val, det), unsafe_allow_html=True)
            with pa2:
                st.markdown(f"""
<div style="background:#F9FAFB;border:1.5px solid #E5E7EB;border-radius:8px;padding:16px;margin-bottom:12px;">
  <p style="font-family:'Inter',sans-serif;font-size:11px;font-weight:600;color:#9CA3AF;
     text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px;">Technical Metrics</p>
  <table style="width:100%;border-collapse:collapse;">
    <tr><td style="font-family:'Inter',sans-serif;font-size:13px;color:#6B7280;padding:4px 0;">Semantic density</td>
        <td style="font-family:'Inter',sans-serif;font-size:13px;color:#111827;font-weight:600;text-align:right;">{page.semantic_density:.1f}%</td></tr>
    <tr><td style="font-family:'Inter',sans-serif;font-size:13px;color:#6B7280;padding:4px 0;">Text length</td>
        <td style="font-family:'Inter',sans-serif;font-size:13px;color:#111827;font-weight:600;text-align:right;">{page.text_len:,} chars</td></tr>
    <tr><td style="font-family:'Inter',sans-serif;font-size:13px;color:#6B7280;padding:4px 0;">HTML size</td>
        <td style="font-family:'Inter',sans-serif;font-size:13px;color:#111827;font-weight:600;text-align:right;">{page.html_len:,} chars</td></tr>
    <tr><td style="font-family:'Inter',sans-serif;font-size:13px;color:#6B7280;padding:4px 0;">Extractability</td>
        <td style="font-family:'Inter',sans-serif;font-size:13px;color:#111827;font-weight:600;text-align:right;">{page.extractability_score}/100</td></tr>
    <tr><td style="font-family:'Inter',sans-serif;font-size:13px;color:#6B7280;padding:4px 0;">Schema types</td>
        <td style="font-family:'Inter',sans-serif;font-size:12px;color:#4F46E5;font-weight:500;text-align:right;">{', '.join(sorted(page.schema_types_found)[:5]) or '—'}</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)
                for r in (page.extractability_reasons or [])[:6]:
                    ok_r = not any(w in r.lower() for w in ["missing", "thin", "no ", "only 0", "only 1"])
                    c_r, icon_r = ("#059669", "✓") if ok_r else ("#9CA3AF", "–")
                    st.markdown(f'<p style="font-family:\'Inter\',sans-serif;font-size:12px;color:{c_r};padding:2px 0;">{icon_r} {r}</p>', unsafe_allow_html=True)

    # ── Strategy ──
    st.markdown(section_title("Implementation Plan", "Schema snippet and access checklist"), unsafe_allow_html=True)
    st1, st2 = st.columns(2)
    with st1:
        st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:12px;font-weight:600;color:#374151;
   margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em;">Phase 1 — Entity Identity</p>
<p style="font-family:'Inter',sans-serif;font-size:13px;color:#6B7280;margin-bottom:12px;line-height:1.6;">
  Add this JSON-LD to the <code style="background:#F3F4F6;padding:1px 5px;border-radius:3px;">&lt;head&gt;</code> of every page.
</p>
""", unsafe_allow_html=True)
        st.code(build_org_jsonld(target_audit.brand, target_audit.origin), language="json")
    with st2:
        st.markdown("""
<p style="font-family:'Inter',sans-serif;font-size:12px;font-weight:600;color:#374151;
   margin-bottom:12px;text-transform:uppercase;letter-spacing:0.05em;">Phase 2 — AI Access Layer</p>
""", unsafe_allow_html=True)
        for lbl, ok, tip in [
            ("llms.txt",            target_audit.llms_txt_present,              "Create at /llms.txt — maps content for AI agents"),
            ("robots.txt",          target_audit.robots_accessible,             "Ensure AI crawlers are not blocked"),
            ("Image alt text",      target_audit.img_missing_alt == 0,          f"{target_audit.img_missing_alt} images need alt tags"),
            ("Organization sameAs", len(target_audit.sameas_social_links) >= 2, "Link to Wikipedia, LinkedIn, etc."),
        ]:
            st.markdown(check_row(lbl, ok, tip), unsafe_allow_html=True)

        if competitor_audit and comp_score is not None:
            gap = target_score - comp_score
            gap_str = f"+{gap}" if gap >= 0 else str(gap)
            gap_c = "#059669" if gap >= 0 else "#DC2626"
            gap_note = "You lead the competitor." if gap >= 0 else "Close the gap with the fixes above."
            st.markdown(f"""
<div style="margin-top:20px;background:#F9FAFB;border:1.5px solid #E5E7EB;border-radius:8px;padding:16px;">
  <p style="font-family:'Inter',sans-serif;font-size:11px;color:#9CA3AF;font-weight:600;
     text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">vs. {competitor_audit.brand}</p>
  <p style="font-family:'Inter',sans-serif;font-size:22px;font-weight:800;color:{gap_c};letter-spacing:-0.02em;">{gap_str} pts</p>
  <p style="font-family:'Inter',sans-serif;font-size:12px;color:#6B7280;margin-top:3px;">{gap_note}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(cta_section(), unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    col_cta, _ = st.columns([1, 2])
    with col_cta:
        st.link_button("Book a Strategy Call →", url="https://calendly.com/your-link", use_container_width=True)

    with st.expander("Crawler discovery log", expanded=False):
        for note in target_audit.notes:
            st.markdown(f'<p style="font-family:\'Inter\',sans-serif;font-size:12px;color:#9CA3AF;">{note}</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
