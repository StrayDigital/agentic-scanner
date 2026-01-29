import json
import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# --- Config ---
DEFAULT_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# REPLACE THIS WITH YOUR CALENDLY LINK
BOOKING_URL = "https://calendly.com/your-handle/fix-it-call" 

# --- Helpers ---
def safe_base_url(url):
    u = url.strip()
    if not u: return ""
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    parsed = urlparse(u)
    return f"{parsed.scheme}://{parsed.netloc}"

def fetch_text(url, timeout):
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.url, r.text
    except Exception as e:
        raise e

def is_internal(url, origin):
    try:
        return urlparse(url).netloc == urlparse(origin).netloc
    except:
        return False

def normalize_url(u):
    return urlparse(u)._replace(fragment="").geturl()

# --- Parsing ---
def normalize_schema_type(t):
    out = []
    if not t: return out
    if isinstance(t, str): out.append(t.strip().lower())
    elif isinstance(t, list):
        for i in t: out.extend(normalize_schema_type(i))
    elif isinstance(t, dict):
        for k in ["@type", "type", "name"]:
            if k in t: out.extend(normalize_schema_type(t[k]))
    return [x for x in out if x]

def iter_json_objects(node):
    if isinstance(node, dict):
        yield node
        for v in node.values(): yield from iter_json_objects(v)
    elif isinstance(node, list):
        for i in node: yield from iter_json_objects(i)

def _try_json_parse(raw):
    try: return json.loads(raw)
    except: pass
    no_comments = re.sub(r"//.*?$|/\*.*?\*/", "", raw, flags=re.MULTILINE | re.DOTALL).strip()
    no_trailing = re.sub(r",\s*([}\]])", r"\1", no_comments)
    try: return json.loads(no_trailing)
    except: return None

def _split_json_blocks(raw):
    blocks = []
    i, n = 0, len(raw)
    while i < n:
        start = -1
        for j in range(i, n):
            if raw[j] in "{[":
                start = j
                break
        if start == -1: break
        
        open_ch = raw[start]
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        
        for k in range(start, n):
            ch = raw[k]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == open_ch: depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        blocks.append(raw[start:k+1].strip())
                        i = k + 1
                        break
        else: break
    return blocks or [raw]

def extract_payloads(html):
    soup = BeautifulSoup(html, "lxml")
    scripts = soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)})
    payloads = []
    for s in scripts:
        raw = s.get_text(strip=False).strip()
        if not raw: continue
        parsed = _try_json_parse(raw)
        if parsed:
            payloads.append(parsed)
            continue
        for block in _split_json_blocks(raw):
            p = _try_json_parse(block)
            if p: payloads.append(p)
    return payloads

def get_flat_objects(payloads):
    objs = []
    for p in payloads:
        roots = p if isinstance(p, list) else [p]
        for r in roots:
            for o in iter_json_objects(r):
                if isinstance(o, dict):
                    if "@graph" in o and isinstance(o["@graph"], list):
                        for g in o["@graph"]:
                            for go in iter_json_objects(g):
                                if isinstance(go, dict): objs.append(go)
                    objs.append(o)
    return objs

def find_obj(objs, type_name):
    for o in objs:
        if type_name.lower() in normalize_schema_type(o.get("@type")): return o
    return None

def find_objs(objs, type_name):
    return [o for o in objs if type_name.lower() in normalize_schema_type(o.get("@type"))]

# --- Auditing ---
def check_identity(org):
    if not org: return False
    d = org.get("disambiguatingDescription")
    if isinstance(d, str) and d.strip(): return True
    s = org.get("sameAs")
    if isinstance(s, str) and s.strip(): return True
    if isinstance(s, list) and any(isinstance(x, str) and x.strip() for x in s): return True
    return False

def check_offers(prod):
    o = prod.get("offers")
    if isinstance(o, (dict, list)) and len(o) > 0: return True
    return False

def audit_page(url, timeout):
    res = {
        "url": url, "final_url": url, "ok": False, "err": None,
        "org": False, "identity": False, "faq": False, "prod": False, "offers": False,
        "score": 0, "warn": []
    }
    try:
        final_url, html = fetch_text(url, timeout)
        res["final_url"] = final_url
        res["ok"] = True
    except Exception as e:
        res["err"] = str(e)
        res["warn"].append(f"⚠️ Crawl Failed: {e}")
        return res

    payloads = extract_payloads(html)
    objs = get_flat_objects(payloads)
    
    org_obj = find_obj(objs, "Organization")
    faq_objs = find_objs(objs, "FAQPage")
    prod_objs = find_objs(objs, "Product")
    
    res["org"] = bool(org_obj)
    res["faq"] = len(faq_objs) > 0
    res["prod"] = len(prod_objs) > 0
    
    if org_obj: res["identity"] = check_identity(org_obj)
    if prod_objs: res["offers"] = any(check_offers(p) for p in prod_objs)
    
    # Scoring
    if res["org"]:
        res["score"] += 10
        if res["identity"]: res["score"] += 20
    if res["faq"]: res["score"] += 20
    if res["prod"]:
        res["score"] += 20
        if res["offers"]: res["score"] += 30
        
    # Warnings
    if res["org"] and not res["identity"]:
        res["warn"].append("⚠️ **Identity Risk:** Add `disambiguatingDescription`.")
    if not res["org"]:
        res["warn"].append("⚠️ **Entity Risk:** No Organization schema found.")
    if res["prod"] and not res["offers"]:
        res["warn"].append("❌ **Commerce Blocked:** Missing Price/Stock data.")
    if not res["faq"]:
        res["warn"].append("⚠️ **Knowledge Gap:** No FAQPage schema found.")
        
    return res

# --- Crawler ---
def extract_xml_urls(xml):
    return [normalize_url(u.strip()) for u in re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, re.I) if u.strip()]

def get_product_urls_recursive(start_url, origin, timeout, status_cb=None, depth=0):
    if depth > 2: return []
    found = []
    try:
        _, xml = fetch_text(start_url, timeout)
        locs = extract_xml_urls(xml)
        
        child_sitemaps = [u for u in locs if u.lower().endswith('.xml')]
        pages = [u for u in locs if not u.lower().endswith('.
