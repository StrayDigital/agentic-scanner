# app.py
# Streamlit app: Site-Wide 'Smart Scan' (Sitemap based)
# Run:
#   pip install streamlit requests beautifulsoup4 lxml
#   streamlit run app.py

import json
import re
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin

import requests
import streamlit as st
from bs4 import BeautifulSoup

# ----------------------------
# 1. CRAWLER & SITEMAP LOGIC
# ----------------------------
def fetch_sitemap_urls(domain_url: str) -> List[str]:
    """
    Try to find sitemap.xml and extract product URLs.
    Returns a list of up to 5 prioritized URLs (Homepage + Products).
    """
    # Clean domain
    parsed = urlparse(domain_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    sitemap_url = urljoin(base_url, "sitemap.xml")
    
    urls_to_scan = [base_url] # Always scan homepage
    product_urls = []

    try:
        resp = requests.get(sitemap_url, timeout=10)
        if resp.status_code == 200:
            # Simple XML parsing for <loc> tags
            soup = BeautifulSoup(resp.content, "xml")
            all_locs = [loc.text.strip() for loc in soup.find_all("loc")]
            
            # Filter for Shopify product pages usually containing '/products/'
            product_urls = [u for u in all_locs if "/products/" in u]
            
            # If no /products/ found, just take random other pages
            if not product_urls:
                 product_urls = [u for u in all_locs if u != base_url][:5]
            
    except Exception:
        # If sitemap fails, just return homepage
        pass

    # Strategy: Scan Homepage + up to 3 random products
    if product_urls:
        sampled_products = random.sample(product_urls, min(len(product_urls), 3))
        urls_to_scan.extend(sampled_products)
    
    return urls_to_scan

# ----------------------------
# 2. CORE SCHEMA PARSING (Same as before)
# ----------------------------
def extract_jsonld(html: str) -> List[Any]:
    soup = BeautifulSoup(html, "lxml")
    scripts = soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)})
    payloads = []
    for s in scripts:
        txt = s.get_text(strip=False)
        # Basic cleanup
        txt = re.sub(r"//.*?$|/\*.*?\*/", "", txt, flags=re.MULTILINE | re.DOTALL).strip()
        txt = re.sub(r",\s*([}\]])", r"\1", txt)
        try:
            payloads.append(json.loads(txt))
        except:
            pass
    return payloads

def find_types(payloads: List[Any]) -> Set[str]:
    found = set()
    # Recursive search for @type
    def search(node):
        if isinstance(node, dict):
            t = node.get("@type")
            if t:
                if isinstance(t, list):
                    for x in t: found.add(str(x).lower())
                else:
                    found.add(str(t).lower())
            for v in node.values(): search(v)
        elif isinstance(node, list):
            for i in node: search(i)
    
    search(payloads)
    return found

def check_identity(payloads: List[Any]) -> bool:
    # Look for disambiguatingDescription or sameAs inside Organization
    def search(node):
        if isinstance(node, dict):
            if node.get("@type", "").lower() == "organization":
                if node.get("disambiguatingDescription") or node.get("sameAs"):
                    return True
            for v in node.values():
                if search(v): return True
        elif isinstance(node, list):
            for i in node:
                if search(i): return True
        return False
    return search(payloads)

def check_offers(payloads: List[Any]) -> bool:
    # Look for offers inside Product
    def search(node):
        if isinstance(node, dict):
            if node.get("@type", "").lower() == "product":
                if node.get("offers"):
                    return True
            for v in node.values():
                if search(v): return True
        elif isinstance(node, list):
            for i in node:
                if search(i): return True
        return False
    return search(payloads)

# ----------------------------
# 3. STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Agentic Site Scanner", page_icon="🤖")

st.title("🤖 Agentic Visibility Scanner")
st.markdown("Scans your **Homepage + Product Pages** to see if you are visible to AI Agents.")

url_input = st.text_input("Enter your Domain (e.g., https://getsmiley.com)")
start_scan = st.button("Run Full Site Scan", type="primary")

if start_scan and url_input:
    if not url_input.startswith("http"):
        st.error("Please include https://")
        st.stop()

    # Phase 1: Crawl
    with st.status("🕵️ Finding pages to scan...", expanded=True) as status:
        urls = fetch_sitemap_urls(url_input)
        st.write(f"Found {len(urls)} key pages to analyze.")
        time.sleep(1)
        status.update(label="Scanning pages...", state="running")
        
        # Phase 2: Analyze
        results = []
        progress_bar = st.progress(0)
        
        for idx, u in enumerate(urls):
            try:
                resp = requests.get(u, headers={"User-Agent": "AgenticScanner/1.0"}, timeout=10)
                payloads = extract_jsonld(resp.text)
                types = find_types(payloads)
                
                # Page-level Scoring
                page_score = 0
                notes = []
                
                # Identity Check (Organization)
                has_org = "organization" in types
                has_identity = check_identity(payloads)
                if has_org: page_score += 10
                if has_identity: page_score += 20
                
                # Commerce Check (Product) - Only if it's a product page
                is_product_page = "/products/" in u or "product" in types
                has_product = "product" in types
                has_offers = check_offers(payloads)
                
                if is_product_page:
                    if has_product: page_score += 20
                    if has_offers: page_score += 50
                else:
                    # If homepage, give points for just existing without errors
                    page_score += 70 

                results.append({
                    "url": u,
                    "score": min(page_score, 100),
                    "has_identity": has_identity,
                    "has_offers": has_offers if is_product_page else "N/A"
                })
                
            except Exception as e:
                st.warning(f"Failed to scan {u}")
            
            progress_bar.progress((idx + 1) / len(urls))

        status.update(label="Scan Complete!", state="complete")

    # Phase 3: The Report Card
    avg_score = int(sum(r["score"] for r in results) / len(results)) if results else 0
    
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.metric(label="Agentic Health Score", value=f"{avg_score}/100")
        if avg_score < 80:
            st.error("Critical Risk: Invisible")
        else:
            st.success("Agentic Ready")

    with c2:
        st.markdown("### ⚠️ Diagnosis")
        # Global Checks
        identity_safe = any(r["has_identity"] for r in results)
        commerce_safe = any(r["has_offers"] is True for r in results)
        
        if not identity_safe:
            st.warning("❌ **Identity Crisis:** No page linked your brand to a specific Entity. AI will confuse you with generic competitors.")
        else:
            st.success("✅ **Identity Safe:** AI knows who you are.")
            
        if not commerce_safe:
            st.warning("❌ **Commerce Blocked:** Your product feeds are missing Price/Stock data for AI agents.")
        else:
            st.success("✅ **Commerce Ready:** ChatGPT can sell your products.")

    # Detailed Table
    st.markdown("### Scanned Pages")
    st.dataframe(results)

    # The Sales Trap
    st.markdown("---")
    st.info("💡 **Want to fix the red 'X's automatically?** We can inject the missing 'Agentic Schema' into your site without touching your code.")
    st.link_button("👉 Book Your Fix (15 Min)", "https://calendly.com")
