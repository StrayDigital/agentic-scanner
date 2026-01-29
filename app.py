# app.py
# Whole-Site Schema Scanner (Homepage + 3 Product Pages)
# UPDATED: Fixes SyntaxError by ensuring all colons and brackets are present.

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
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.url, r.text
    except Exception as e:
        raise e


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


def extract_jsonld_payloads(html: str) ->
