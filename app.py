# app.py — Agentic Infrastructure Audit (Premium Edition)
# Requirements:
#   pip install streamlit requests beautifulsoup4 lxml plotly anthropic
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

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ----------------------------
# Page config — must be first
# ----------------------------
st.set_page_config(
    page_title="AEO Audit — Agentic Infrastructure",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# Premium CSS injection
# ----------------------------

PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
  background: #f0f4f8 !important;
  color: #1e293b !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
  background: #f0f4f8 !important;
  padding: 0 !important;
}

.main .block-container {
  max-width: 1280px !important;
  padding: 0 2rem 4rem !important;
  margin: 0 auto !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  display: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: #0f172a !important;
  letter-spacing: -0.02em !important;
}

.stMarkdown p { color: #475569; line-height: 1.7; }

/* ── Text input ── */
[data-testid="stTextInput"] label {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: #64748b !important;
  font-weight: 500 !important;
}

[data-testid="stTextInput"] input {
  background: #ffffff !important;
  border: 1.5px solid #e2e8f0 !important;
  border-radius: 8px !important;
  color: #0f172a !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 14px !important;
  padding: 11px 16px !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}

[data-testid="stTextInput"] input:focus {
  border-color: #2563eb !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
  outline: none !important;
}

[data-testid="stTextInput"] input::placeholder { color: #94a3b8 !important; }

/* ── Primary button ── */
[data-testid="stButton"] > button[kind="primary"] {
  background: #2563eb !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  letter-spacing: 0.01em !important;
  padding: 12px 24px !important;
  transition: all 0.15s !important;
  width: 100% !important;
  box-shadow: 0 1px 3px rgba(37,99,235,0.3) !important;
}

[data-testid="stButton"] > button[kind="primary"]:hover {
  background: #1d4ed8 !important;
  box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
  transform: translateY(-1px) !important;
}

/* ── Secondary button ── */
[data-testid="stButton"] > button[kind="secondary"] {
  background: #ffffff !important;
  color: #475569 !important;
  border: 1.5px solid #e2e8f0 !important;
  border-radius: 8px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 13px !important;
  padding: 9px 18px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: #ffffff !important;
  border: 1.5px solid #e2e8f0 !important;
  border-radius: 10px !important;
  margin-bottom: 10px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

[data-testid="stExpander"] summary {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: #334155 !important;
  padding: 14px 18px !important;
}

[data-testid="stExpander"] summary:hover { color: #0f172a !important; }

/* ── Plotly charts ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: #ffffff !important;
  border: 1.5px solid #e2e8f0 !important;
  border-radius: 10px !important;
  padding: 18px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

[data-testid="stMetricLabel"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: 30px !important;
  color: #0f172a !important;
  font-weight: 800 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #2563eb !important; }

/* ── Code blocks ── */
[data-testid="stCodeBlock"] {
  background: #f8fafc !important;
  border: 1.5px solid #e2e8f0 !important;
  border-radius: 8px !important;
}

/* ── Columns gap ── */
[data-testid="column"] { padding: 0 8px !important; }

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1.5px solid #e2e8f0 !important;
  margin: 2.5rem 0 !important;
}
</style>
"""

# ----------------------------
# Custom HTML components
# ----------------------------
def hero_header() -> str:
    return """
<div style="
  background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 60%, #2563eb 100%);
  padding: 3rem 2.5rem 2.5rem;
  border-radius: 0 0 20px 20px;
  margin-bottom: 2.5rem;
  position: relative;
  overflow: hidden;
">
  <div style="position:absolute; top:-60px; right:-60px; width:300px; height:300px;
    background:rgba(255,255,255,0.04); border-radius:50%;"></div>
  <div style="position:absolute; bottom:-80px; right:100px; width:200px; height:200px;
    background:rgba(255,255,255,0.03); border-radius:50%;"></div>
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px; position:relative;">
    <div style="
      width:32px; height:32px; background:rgba(255,255,255,0.15);
      border-radius:8px; display:flex; align-items:center; justify-content:center;
      font-size:16px; border:1px solid rgba(255,255,255,0.2);
    ">⬡</div>
    <span style="
      font-family:'JetBrains Mono',monospace; font-size:11px;
      letter-spacing:0.14em; text-transform:uppercase; color:rgba(255,255,255,0.6);
    ">Agentic Infrastructure</span>
    <span style="
      font-family:'JetBrains Mono',monospace; font-size:10px;
      background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2);
      color:rgba(255,255,255,0.8); padding:2px 8px; border-radius:20px;
      letter-spacing:0.06em;
    ">LIVE</span>
  </div>
  <h1 style="
    font-family:'Plus Jakarta Sans',sans-serif; font-size:clamp(1.8rem,3.5vw,2.8rem);
    font-weight:800; color:#ffffff; letter-spacing:-0.03em;
    line-height:1.15; margin-bottom:12px; position:relative;
  ">AI Search Readiness<br>Audit Engine</h1>
  <p style="
    font-family:'Plus Jakarta Sans',sans-serif; font-size:15px;
    color:rgba(255,255,255,0.65); max-width:540px; line-height:1.6; position:relative;
  ">Score your site's readiness for ChatGPT, Perplexity, Claude & Google AI Overviews. Schema, entity signals, crawler access — instant results.</p>
</div>
"""

def section_header(title: str, subtitle: str = "", mono_tag: str = "") -> str:
    tag_html = f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:10px; letter-spacing:0.1em; text-transform:uppercase; color:#2563eb; background:#eff6ff; border:1px solid #bfdbfe; border-radius:4px; padding:3px 8px; margin-right:10px;">{mono_tag}</span>' if mono_tag else ""
    sub_html = f'<p style="font-family:\'Plus Jakarta Sans\',sans-serif; font-size:14px; color:#64748b; margin-top:5px;">{subtitle}</p>' if subtitle else ""
    return f"""
<div style="margin: 2.5rem 0 1.25rem;">
  <div style="display:flex; align-items:center; gap:6px; margin-bottom:5px;">
    {tag_html}
  </div>
  <h2 style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.35rem; font-weight:800; color:#0f172a; letter-spacing:-0.02em;">{title}</h2>
  {sub_html}
</div>
"""

def score_card(score: int, label: str, brand: str) -> str:
    if score >= 70:
        color = "#16a34a"
        bg_ring = "#dcfce7"
        grade = "Good"
        grade_bg = "#dcfce7"
        grade_c = "#16a34a"
    elif score >= 40:
        color = "#d97706"
        bg_ring = "#fef3c7"
        grade = "Fair"
        grade_bg = "#fef3c7"
        grade_c = "#d97706"
    else:
        color = "#dc2626"
        bg_ring = "#fee2e2"
        grade = "Poor"
        grade_bg = "#fee2e2"
        grade_c = "#dc2626"
    circumference = 2 * 3.14159 * 52
    dash = (score / 100) * circumference
    return f"""
<div style="
  background:#ffffff; border:1.5px solid #e2e8f0;
  border-radius:14px; padding:24px 20px; text-align:center;
  box-shadow:0 1px 4px rgba(0,0,0,0.06);
  position:relative; overflow:hidden;
">
  <div style="position:absolute; top:0; left:0; right:0; height:3px; background:{color};
    opacity:0.7; border-radius:14px 14px 0 0;"></div>
  <p style="font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.1em; text-transform:uppercase; color:#94a3b8; margin-bottom:16px;">{label}</p>
  <div style="position:relative; width:130px; height:130px; margin:0 auto 14px;">
    <svg width="130" height="130" viewBox="0 0 130 130">
      <circle cx="65" cy="65" r="52" fill="none" stroke="#f1f5f9" stroke-width="9"/>
      <circle cx="65" cy="65" r="52" fill="none" stroke="{color}" stroke-width="9"
        stroke-dasharray="{dash:.1f} {circumference:.1f}"
        stroke-dashoffset="{circumference/4:.1f}"
        stroke-linecap="round"/>
    </svg>
    <div style="
      position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
      text-align:center;
    ">
      <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:2rem; font-weight:800; color:{color}; line-height:1;">{score}</div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:9px; color:#94a3b8; letter-spacing:0.08em;">/100</div>
    </div>
  </div>
  <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:14px; font-weight:700; color:#0f172a; margin-bottom:8px;">{brand}</div>
  <span style="display:inline-block; font-family:'Plus Jakarta Sans',sans-serif; font-size:11px; font-weight:600; color:{grade_c}; background:{grade_bg}; padding:2px 10px; border-radius:20px;">{grade}</span>
</div>
"""

def signal_row(label: str, earned: int, maximum: int) -> str:
    pct = int((earned / maximum) * 100) if maximum > 0 else 0
    color = "#16a34a" if earned > 0 else "#e2e8f0"
    status = "✓" if earned > 0 else "—"
    status_color = "#16a34a" if earned > 0 else "#cbd5e1"
    return f"""
<div style="
  display:flex; align-items:center; gap:14px;
  padding:10px 0; border-bottom:1px solid #f1f5f9;
">
  <div style="width:18px; text-align:center; font-size:12px; color:{status_color}; font-weight:700;">{status}</div>
  <div style="flex:1; font-family:'Plus Jakarta Sans',sans-serif; font-size:13px; color:#475569; font-weight:500;">{label}</div>
  <div style="width:72px; height:4px; background:#f1f5f9; border-radius:4px; overflow:hidden;">
    <div style="width:{pct}%; height:100%; background:{color}; border-radius:4px;"></div>
  </div>
  <div style="width:44px; text-align:right; font-family:'JetBrains Mono',monospace; font-size:12px; color:#334155; font-weight:500;">{earned}/{maximum}</div>
</div>
"""

def insight_card(icon: str, title: str, value: str, status: str, detail: str = "") -> str:
    colors = {
        "ok":   ("#16a34a", "#f0fdf4", "#bbf7d0", "#166534"),
        "warn": ("#d97706", "#fffbeb", "#fde68a", "#92400e"),
        "err":  ("#dc2626", "#fef2f2", "#fecaca", "#991b1b"),
        "info": ("#2563eb", "#eff6ff", "#bfdbfe", "#1e40af"),
    }
    c, bg, border, text_c = colors.get(status, colors["info"])
    detail_html = f'<div style="font-family:\'JetBrains Mono\',monospace; font-size:10px; color:{c}; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{detail}</div>' if detail else ""
    return f"""
<div style="
  background:{bg}; border:1.5px solid {border};
  border-radius:10px; padding:16px;
  box-shadow:0 1px 2px rgba(0,0,0,0.03);
">
  <div style="display:flex; align-items:flex-start; gap:10px;">
    <div style="font-size:16px; margin-top:1px;">{icon}</div>
    <div style="flex:1; min-width:0;">
      <div style="font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.08em; text-transform:uppercase; color:{c}; margin-bottom:3px; opacity:0.8;">{title}</div>
      <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:15px; font-weight:700; color:{text_c};">{value}</div>
      {detail_html}
    </div>
  </div>
</div>
"""

def crawler_pill(name: str, status: str) -> str:
    if status == "Accessible":
        c, bg, border, dot = "#16a34a", "#f0fdf4", "#bbf7d0", "#16a34a"
    elif status == "Blocked":
        c, bg, border, dot = "#dc2626", "#fef2f2", "#fecaca", "#dc2626"
    elif status == "Thin Render":
        c, bg, border, dot = "#d97706", "#fffbeb", "#fde68a", "#d97706"
    else:
        c, bg, border, dot = "#64748b", "#f8fafc", "#e2e8f0", "#94a3b8"
    return f"""
<div style="
  background:{bg}; border:1.5px solid {border};
  border-radius:10px; padding:14px 16px;
  display:flex; flex-direction:column; gap:7px;
  box-shadow:0 1px 2px rgba(0,0,0,0.03);
">
  <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:#64748b; letter-spacing:0.07em;">{name}</div>
  <div style="display:flex; align-items:center; gap:6px;">
    <div style="width:7px; height:7px; border-radius:50%; background:{dot};
      box-shadow:0 0 5px {dot}80;"></div>
    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:12px; font-weight:700; color:{c};">{status}</div>
  </div>
</div>
"""

def authority_node(name: str, found: bool) -> str:
    if found:
        return f'<div style="display:flex; align-items:center; gap:7px; padding:7px 12px; background:#f0fdf4; border:1.5px solid #bbf7d0; border-radius:8px;"><span style="color:#16a34a; font-size:11px; font-weight:700;">✓</span><span style="font-family:\'JetBrains Mono\',monospace; font-size:11px; color:#166534; font-weight:500;">{name}</span></div>'
    else:
        return f'<div style="display:flex; align-items:center; gap:7px; padding:7px 12px; background:#f8fafc; border:1.5px solid #e2e8f0; border-radius:8px;"><span style="color:#cbd5e1; font-size:11px;">—</span><span style="font-family:\'JetBrains Mono\',monospace; font-size:11px; color:#94a3b8;">{name}</span></div>'

def page_audit_row(label: str, ok: bool, detail: str = "") -> str:
    icon = "✓" if ok else "✕"
    ic = "#16a34a" if ok else "#dc2626"
    icon_bg = "#f0fdf4" if ok else "#fef2f2"
    det = f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:10px; color:#94a3b8; margin-left:8px;">{detail[:60]}</span>' if detail else ""
    return f"""
<div style="
  display:flex; align-items:center; gap:10px;
  padding:8px 0; border-bottom:1px solid #f8fafc;
">
  <span style="font-family:'JetBrains Mono',monospace; font-size:11px; color:{ic};
    background:{icon_bg}; width:20px; height:20px; border-radius:4px;
    display:inline-flex; align-items:center; justify-content:center; flex-shrink:0;
    font-weight:700;">{icon}</span>
  <span style="font-family:'Plus Jakarta Sans',sans-serif; font-size:13px; color:#475569; flex:1; font-weight:500;">{label}</span>
  {det}
</div>
"""

def ai_brief_card(content: str) -> str:
    return f"""
<div style="
  background:#ffffff;
  border:1.5px solid #bfdbfe;
  border-left:4px solid #2563eb;
  border-radius:10px;
  padding:22px 26px;
  margin-top:14px;
  font-family:'Plus Jakarta Sans',sans-serif;
  font-size:14px;
  color:#334155;
  line-height:1.8;
  white-space:pre-wrap;
  box-shadow:0 1px 4px rgba(37,99,235,0.08);
">{content}</div>
"""

def cta_block() -> str:
    return """
<div style="
  background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
  border-radius:16px;
  padding:40px;
  text-align:center;
  margin-top:2rem;
  position:relative;
  overflow:hidden;
">
  <div style="position:absolute; top:-40px; right:-40px; width:200px; height:200px;
    background:rgba(255,255,255,0.04); border-radius:50%;"></div>
  <div style="font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.14em; text-transform:uppercase; color:rgba(255,255,255,0.6); margin-bottom:10px; position:relative;">Ready to fix it?</div>
  <h3 style="font-family:'Plus Jakarta Sans',sans-serif; font-size:1.75rem; font-weight:800; color:#ffffff; letter-spacing:-0.02em; margin-bottom:10px; position:relative;">Get expert implementation</h3>
  <p style="font-family:'Plus Jakarta Sans',sans-serif; font-size:14px; color:rgba(255,255,255,0.65); max-width:420px; margin:0 auto 24px; line-height:1.6; position:relative;">We'll implement every signal, fix your schema, and deploy llms.txt — guaranteed to raise your score.</p>
</div>
"""

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
    "instagram.com", "facebook.com", "tiktok.com",
    "x.com", "twitter.com", "youtube.com", "linkedin.com",
)
AI_CRAWLERS = {
    "GPTBot":         "Mozilla/5.0 (compatible; GPTBot/1.0; +https://openai.com/gptbot)",
    "Claude-Web":     "Claude-Web/1.0",
    "CCBot":          "CCBot/2.0",
    "Google-Extended":"Google-Extended",
    "PerplexityBot":  "PerplexityBot/1.0",
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
    turbo_found, turbo_notes = crawl_sitemap_for_products(
        turbo_url, origin, timeout, limit=3
    )
    notes.extend(turbo_notes)
    if turbo_found:
        return homepage_url, turbo_found, notes

    sm_url = urljoin(origin, UNIVERSAL_SITEMAP_PATH)
    sm_found, sm_notes = crawl_sitemap_for_products(sm_url, origin, timeout, limit=3)
    notes.extend(sm_notes)
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
            for sm in sitemap_lines:
                found, smn = crawl_sitemap_for_products(sm, origin, timeout, limit=3)
                notes.extend(smn)
                if found:
                    return homepage_url, found, notes

    scrape_found = discover_from_homepage_scrape(origin, home_html, limit=3)
    if scrape_found:
        notes.append(f"Homepage scrape: {len(scrape_found)} product-like URL(s)")
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
    no_trailing = re.sub(r",\s*([}\]])", r"\1", no_comments)
    try:
        return json.loads(no_trailing)
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


def find_first(
    objs: List[Dict[str, Any]], target: str
) -> Optional[Dict[str, Any]]:
    for o in objs:
        if has_type(o, target):
            return o
    return None


def find_all(objs: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    return [o for o in objs if has_type(o, target)]


# ----------------------------
# Scoring
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
    def _check(d: Dict[str, Any]) -> bool:
        if "price" in d and str(d.get("price", "")).strip():
            return True
        ps = d.get("priceSpecification")
        if isinstance(ps, dict) and "price" in ps:
            return True
        if isinstance(ps, list):
            return any(
                isinstance(i, dict) and "price" in i for i in ps
            )
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


# ----------------------------
# Technical checks
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


# ----------------------------
# Page audit
# ----------------------------
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


# ----------------------------
# Full site audit
# ----------------------------
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


# ----------------------------
# Claude API integration
# ----------------------------
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


# ----------------------------
# Streamlit App
# ----------------------------
def main() -> None:
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
    st.markdown(hero_header(), unsafe_allow_html=True)

    # ── Input row ──
    col_t, col_c, col_key = st.columns([2, 2, 1])
    with col_t:
        target_input = st.text_input(
            "YOUR URL",
            placeholder="https://yoursite.com",
            key="target",
        )
    with col_c:
        competitor_input = st.text_input(
            "COMPETITOR URL",
            placeholder="https://competitor.com (optional)",
            key="competitor",
        )
    with col_key:
        api_key_input = st.text_input(
            "CLAUDE API KEY",
            placeholder="sk-ant-...",
            type="password",
            key="api_key_field",
        )
        if api_key_input:
            st.session_state["anthropic_api_key"] = api_key_input

    run_btn = st.button("Run Audit →", type="primary", key="run")

    if not run_btn:
        st.markdown("""
<div style="
  margin-top:3rem; padding:3rem;
  background:#0d1117; border:1px dashed #1a2332;
  border-radius:16px; text-align:center;
">
  <div style="font-family:'DM Mono',monospace; font-size:11px; letter-spacing:0.14em; text-transform:uppercase; color:#2e3c4e; margin-bottom:12px;">awaiting input</div>
  <p style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#4a6080;">Enter a URL above and run your audit.</p>
</div>
""", unsafe_allow_html=True)
        return

    target_url = ensure_scheme(target_input.strip())
    if not target_url:
        st.error("Please enter a valid URL.")
        return

    competitor_url = ensure_scheme(competitor_input.strip()) if competitor_input.strip() else None

    # ── Run audits ──
    with st.spinner("Scanning your site…"):
        target_audit = run_site_audit(target_url)

    competitor_audit: Optional[SiteAudit] = None
    if competitor_url:
        with st.spinner("Scanning competitor…"):
            competitor_audit = run_site_audit(competitor_url)

    target_score = max(p.score for p in target_audit.pages) if target_audit.pages else 0
    comp_score = max(p.score for p in competitor_audit.pages) if competitor_audit and competitor_audit.pages else None
    hp = target_audit.pages[0] if target_audit.pages else None

    # ═══════════════════════════════════
    # SECTION 1 — Scorecard
    # ═══════════════════════════════════
    st.markdown(section_header("Agentic Score", "Overall AI search readiness rating", "01 / scorecard"), unsafe_allow_html=True)

    score_cols = st.columns([1, 1, 2] if competitor_audit else [1, 2])

    with score_cols[0]:
        st.markdown(score_card(target_score, "YOUR SCORE", target_audit.brand), unsafe_allow_html=True)

    if competitor_audit and comp_score is not None:
        with score_cols[1]:
            st.markdown(score_card(comp_score, "COMPETITOR", competitor_audit.brand), unsafe_allow_html=True)
        breakdown_col = score_cols[2]
    else:
        breakdown_col = score_cols[1]

    with breakdown_col:
        st.markdown("""
<div style="background:#0d1117; border:1px solid #1a2332; border-radius:16px; padding:24px 28px;">
  <div style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:0.14em; text-transform:uppercase; color:#4a6080; margin-bottom:16px;">Signal Breakdown</div>
""", unsafe_allow_html=True)
        if hp:
            st.markdown(signal_row("Organization Schema", 10 if hp.org_found else 0, 10), unsafe_allow_html=True)
            st.markdown(signal_row("Identity Verification", 20 if hp.identity_verified else 0, 20), unsafe_allow_html=True)
            st.markdown(signal_row("FAQPage Schema", 20 if hp.faq_found else 0, 20), unsafe_allow_html=True)
            st.markdown(signal_row("Product Schema", 20 if hp.product_found else 0, 20), unsafe_allow_html=True)
            st.markdown(signal_row("Offers / Price", 30 if hp.commerce_ready else 0, 30), unsafe_allow_html=True)
        delta = target_score - INDUSTRY_AVERAGE_SCORE
        delta_label = f"+{delta}" if delta >= 0 else str(delta)
        delta_c = "#22d47a" if delta >= 0 else "#f0504a"
        st.markdown(f"""
  <div style="margin-top:16px; padding-top:16px; border-top:1px solid #111922;
    display:flex; justify-content:space-between; align-items:center;">
    <span style="font-family:'DM Mono',monospace; font-size:10px; color:#4a6080; text-transform:uppercase; letter-spacing:0.1em;">vs Industry avg ({INDUSTRY_AVERAGE_SCORE})</span>
    <span style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700; color:{delta_c};">{delta_label} pts</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ═══════════════════════════════════
    # SECTION 2 — AI Brief
    # ═══════════════════════════════════
    if st.session_state.get("anthropic_api_key"):
        st.markdown(section_header("AI Fix Brief", "Claude-generated diagnosis and action plan", "02 / ai brief"), unsafe_allow_html=True)
        with st.spinner("Generating AI brief…"):
            brief = generate_ai_brief(target_audit, competitor_audit)
        if brief:
            st.markdown(ai_brief_card(brief), unsafe_allow_html=True)
    else:
        st.markdown(section_header("AI Fix Brief", "Add a Claude API key above to unlock AI-written diagnosis", "02 / ai brief"), unsafe_allow_html=True)
        st.markdown("""
<div style="background:#0d1117; border:1px dashed #1a2332; border-radius:12px; padding:20px 24px;">
  <span style="font-family:'DM Mono',monospace; font-size:12px; color:#2e3c4e;">
    Enter your Claude API key in the field above → instant AI-written fix brief per audit.<br>
    Get your key at <a href="https://console.anthropic.com" style="color:#4a9eff;">console.anthropic.com</a>
  </span>
</div>
""", unsafe_allow_html=True)

    # ═══════════════════════════════════
    # SECTION 3 — Insight Grid
    # ═══════════════════════════════════
    st.markdown(section_header("Insight Grid", "Technical signal overview", "03 / signals"), unsafe_allow_html=True)

    ig1, ig2, ig3, ig4 = st.columns(4)

    with ig1:
        if target_audit.robots_accessible and target_audit.llms_txt_present:
            st.markdown(insight_card("⬡", "AI ACCESS", "Fully open", "ok", "robots.txt + llms.txt"), unsafe_allow_html=True)
        elif target_audit.robots_accessible:
            st.markdown(insight_card("⬡", "AI ACCESS", "Partial", "warn", "llms.txt missing"), unsafe_allow_html=True)
        else:
            st.markdown(insight_card("⬡", "AI ACCESS", "Blocked", "err", "robots.txt inaccessible"), unsafe_allow_html=True)

    with ig2:
        if target_audit.img_missing_alt == 0 and target_audit.img_total > 0:
            st.markdown(insight_card("◈", "IMAGE ALT", f"All {target_audit.img_total} tagged", "ok"), unsafe_allow_html=True)
        elif target_audit.img_missing_alt > 0:
            pct = int((target_audit.img_missing_alt / max(target_audit.img_total, 1)) * 100)
            st.markdown(insight_card("◈", "IMAGE ALT", f"{pct}% missing", "warn", f"{target_audit.img_missing_alt}/{target_audit.img_total} images"), unsafe_allow_html=True)
        else:
            st.markdown(insight_card("◈", "IMAGE ALT", "No images", "info"), unsafe_allow_html=True)

    with ig3:
        if hp:
            d = hp.semantic_density
            if d >= 10:
                st.markdown(insight_card("≋", "DENSITY", f"{d:.1f}%", "ok", "Good signal ratio"), unsafe_allow_html=True)
            elif d >= 5:
                st.markdown(insight_card("≋", "DENSITY", f"{d:.1f}%", "warn", "Weak signal ratio"), unsafe_allow_html=True)
            else:
                st.markdown(insight_card("≋", "DENSITY", f"{d:.1f}%", "err", "Bloated code"), unsafe_allow_html=True)

    with ig4:
        if target_audit.org_present_any and len(target_audit.sameas_social_links) >= 2:
            st.markdown(insight_card("◎", "ENTITY", "Verified", "ok", f"{len(target_audit.sameas_social_links)} sameAs links"), unsafe_allow_html=True)
        elif target_audit.org_present_any:
            st.markdown(insight_card("◎", "ENTITY", "Partial", "warn", "No sameAs links"), unsafe_allow_html=True)
        else:
            st.markdown(insight_card("◎", "ENTITY", "Not found", "err", "Organization schema missing"), unsafe_allow_html=True)

    # ═══════════════════════════════════
    # SECTION 4 — AI Crawler Simulation
    # ═══════════════════════════════════
    st.markdown(section_header("AI Crawler Simulation", f"Access test against {target_audit.homepage_url}", "04 / crawlers"), unsafe_allow_html=True)

    crawler_cols = st.columns(5)
    for idx, (name, result) in enumerate(target_audit.ai_crawler_results.items()):
        with crawler_cols[idx]:
            st.markdown(crawler_pill(name, result.get("status", "Unknown")), unsafe_allow_html=True)

    # ═══════════════════════════════════
    # SECTION 5 — Authority
    # ═══════════════════════════════════
    st.markdown(section_header("Knowledge Graph Authority", target_audit.authority_verdict.capitalize(), "05 / authority"), unsafe_allow_html=True)

    auth_left, auth_right = st.columns([1, 2])
    with auth_left:
        st.markdown(score_card(target_audit.authority_score, "AUTHORITY SCORE", "Knowledge graph"), unsafe_allow_html=True)

    with auth_right:
        st.markdown('<div style="display:flex; flex-wrap:wrap; gap:8px; align-content:flex-start;">', unsafe_allow_html=True)
        for name, found in target_audit.authority_results.items():
            st.markdown(authority_node(name, found), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if target_audit.authority_links:
            st.markdown('<div style="margin-top:16px;">', unsafe_allow_html=True)
            for link in target_audit.authority_links[:4]:
                st.markdown(f'<a href="{link}" style="display:block; font-family:\'DM Mono\',monospace; font-size:11px; color:#4a9eff; text-decoration:none; margin-bottom:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{link}</a>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════
    # SECTION 6 — Page Analysis
    # ═══════════════════════════════════
    st.markdown(section_header("Page Analysis", "Per-page signal waterfall", "06 / pages"), unsafe_allow_html=True)

    for i, page in enumerate(target_audit.pages):
        label = "Homepage" if i == 0 else f"Product Page {i}"
        url_display = page.requested_url[:60] + ("…" if len(page.requested_url) > 60 else "")

        with st.expander(f"{label}  ·  {url_display}", expanded=(i == 0)):
            if not page.ok_fetch:
                st.markdown(f'<div style="font-family:\'DM Mono\',monospace; font-size:12px; color:#f0504a;">Fetch failed: {page.fetch_error}</div>', unsafe_allow_html=True)
                continue

            pa1, pa2 = st.columns(2)
            with pa1:
                score_color = "#22d47a" if page.score >= 70 else "#f59e0b" if page.score >= 40 else "#f0504a"
                st.markdown(f'<div style="font-family:\'Syne\',sans-serif; font-size:2rem; font-weight:800; color:{score_color}; margin-bottom:16px;">{page.score}<span style="font-size:1rem; color:#4a6080;">/100</span></div>', unsafe_allow_html=True)
                st.markdown(page_audit_row("Ghost code clear", not page.ghost, "JS render blocking" if page.ghost else "Server-rendered"), unsafe_allow_html=True)
                st.markdown(page_audit_row("H1 tag present", bool(page.h1_text), page.h1_text[:50] if page.h1_text else ""), unsafe_allow_html=True)
                st.markdown(page_audit_row("Brand in H1", page.h1_has_brand), unsafe_allow_html=True)
                st.markdown(page_audit_row("Organization schema", page.org_found), unsafe_allow_html=True)
                st.markdown(page_audit_row("Identity verified", page.identity_verified), unsafe_allow_html=True)
                st.markdown(page_audit_row("FAQPage schema", page.faq_found), unsafe_allow_html=True)
                st.markdown(page_audit_row("Product schema", page.product_found), unsafe_allow_html=True)
                st.markdown(page_audit_row("Offers / Price data", page.commerce_ready), unsafe_allow_html=True)

            with pa2:
                st.markdown(f"""
<div style="background:#060e1a; border:1px solid #1a2332; border-radius:10px; padding:18px; margin-bottom:12px;">
  <div style="font-family:'DM Mono',monospace; font-size:10px; color:#4a6080; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;">Technical Signals</div>
  <div style="font-family:'DM Sans',sans-serif; font-size:13px; color:#8a95a3; line-height:2;">
    Semantic density: <span style="color:#eef0f3;">{page.semantic_density:.1f}%</span><br>
    Text length: <span style="color:#eef0f3;">{page.text_len:,}</span> chars<br>
    HTML size: <span style="color:#eef0f3;">{page.html_len:,}</span> chars<br>
    Extractability: <span style="color:#eef0f3;">{page.extractability_score}/100</span> — {page.extractability_verdict}<br>
    Schema types: <span style="color:#eef0f3; font-family:'DM Mono',monospace; font-size:11px;">{', '.join(sorted(page.schema_types_found)[:6]) or 'none'}</span>
  </div>
</div>
""", unsafe_allow_html=True)
                if page.extractability_reasons:
                    st.markdown('<div style="font-family:\'DM Mono\',monospace; font-size:10px; color:#4a6080; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Extractability signals</div>', unsafe_allow_html=True)
                    for r in page.extractability_reasons[:6]:
                        has_pass = not any(w in r.lower() for w in ["missing", "thin", "no ", "only 0", "only 1"])
                        rc = "#22d47a" if has_pass else "#2e3c4e"
                        st.markdown(f'<div style="font-family:\'DM Sans\',sans-serif; font-size:12px; color:{rc}; padding:3px 0;">{r}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════
    # SECTION 7 — Strategy
    # ═══════════════════════════════════
    st.markdown(section_header("Implementation Strategy", "Schema snippets and fix roadmap", "07 / strategy"), unsafe_allow_html=True)

    strat1, strat2 = st.columns(2)

    with strat1:
        st.markdown("""
<div style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:0.12em; text-transform:uppercase; color:#4a9eff; margin-bottom:10px;">Phase 1 — Entity Identity</div>
<p style="font-family:'DM Sans',sans-serif; font-size:13px; color:#4a6080; margin-bottom:14px; line-height:1.6;">Add this JSON-LD block to the <code style="background:#111922; padding:1px 5px; border-radius:3px; font-size:11px;">&lt;head&gt;</code> of every page to establish your entity in the knowledge graph.</p>
""", unsafe_allow_html=True)
        st.code(build_org_jsonld(target_audit.brand, target_audit.origin), language="json")

    with strat2:
        st.markdown("""
<div style="font-family:'DM Mono',monospace; font-size:10px; letter-spacing:0.12em; text-transform:uppercase; color:#4a9eff; margin-bottom:10px;">Phase 2 — AI Access Layer</div>
""", unsafe_allow_html=True)
        checks = [
            ("llms.txt", target_audit.llms_txt_present, "Create at /llms.txt — maps your content for AI agents"),
            ("robots.txt", target_audit.robots_accessible, "Ensure crawlers are not blocked"),
            ("Image alt text", target_audit.img_missing_alt == 0, f"{target_audit.img_missing_alt} images need alt tags"),
            ("Organization sameAs", len(target_audit.sameas_social_links) >= 2, "Link to Wikipedia, LinkedIn, etc."),
        ]
        for label, ok, tip in checks:
            st.markdown(page_audit_row(label, ok, tip), unsafe_allow_html=True)

        if competitor_audit and comp_score is not None:
            gap = target_score - comp_score
            gap_c = "#22d47a" if gap >= 0 else "#f0504a"
            gap_str = f"+{gap}" if gap >= 0 else str(gap)
            st.markdown(f"""
<div style="margin-top:20px; padding:16px; background:#0d1117; border:1px solid #1a2332; border-radius:10px;">
  <div style="font-family:'DM Mono',monospace; font-size:10px; color:#4a6080; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">vs {competitor_audit.brand}</div>
  <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:{gap_c};">{gap_str} points</div>
  <div style="font-family:'DM Sans',sans-serif; font-size:12px; color:#4a6080; margin-top:4px;">{"You lead the competitor." if gap >= 0 else "Close the gap with the fixes above."}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(cta_block(), unsafe_allow_html=True)

    col_cta, _ = st.columns([1, 2])
    with col_cta:
        st.link_button(
            "Book a Strategy Call →",
            url="https://calendly.com/your-link",
            use_container_width=True,
        )

    with st.expander("Crawler discovery log", expanded=False):
        for note in target_audit.notes:
            st.markdown(f'<span style="font-family:\'DM Mono\',monospace; font-size:11px; color:#2e3c4e;">{note}</span>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
