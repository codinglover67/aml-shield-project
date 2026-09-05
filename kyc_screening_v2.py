"""
kyc_screening.py — PEP + Adverse Media module for AML Shield
=============================================================
Drop this file next to main.py, then add the routes below to main.py.

REQUIREMENTS:
    pip install httpx beautifulsoup4 lxml tldextract

USAGE in main.py:
    from kyc_screening import router as kyc_router
    app.include_router(kyc_router, prefix="/kyc")

PEP screening: scrapes https://www.opensanctions.org/search/ (no API key needed).

Adverse Media fetch order:
  1. Google Custom Search API  — best coverage, requires free API key + CX
  2. GDELT DOC 2.0             — no key needed, academic news index
  3. NewsAPI                   — optional fallback, requires free API key

To enable Google search (recommended):
  1. Get a free API key: https://console.cloud.google.com → "Custom Search API"
  2. Create a search engine: https://programmablesearchengine.google.com
     → tick "Search the entire web"  → copy the cx (Search Engine ID)
  3. Set env vars:
       export GOOGLE_CSE_KEY="AIza..."
       export GOOGLE_CSE_CX="12345:abcde..."
"""

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from media_whitelist import get_whitelist_entry, whitelist_summary, _WHITELIST_MAP
log = logging.getLogger("aml.kyc")
router = APIRouter(tags=["KYC"])

# ── Config ───────────────────────────────────────────────────────────────────

# OpenSanctions website search (no API key required)
OS_WEBSITE = "https://www.opensanctions.org"
OS_SEARCH_URL = f"{OS_WEBSITE}/search/"

# Tag badge text → normalised topic key mapping
# Derived from what the website actually renders in the result badges
_BADGE_TO_TOPIC: dict[str, str] = {
    "pep":                    "role.pep",
    "politically exposed":    "role.pep",
    "close associate":        "role.rca",
    "relative":               "role.rca",
    "foreign pep":            "role.pep.foreign",
    "domestic pep":           "role.pep.domestic",
    "diplomat":               "role.diplomat",
    "person of interest":     "poi",
    "counter-sanctioned":     "sanction",
    "sanctioned":             "sanction",
    "sanction":               "sanction",
    "sanction-linked":        "sanction.linked",
    "debarred":               "debarment",
    "crime":                  "crime",
    "terrorism":              "crime.terror",
    "financial crime":        "crime.fin",
    "trade risk":             "export",
}

# ── Adverse media keyword config ─────────────────────────────────────────────
# Analysts can PUT /kyc/media/config to update these at runtime

DEFAULT_KEYWORD_WEIGHTS: dict[str, float] = {
    # Tier 1 — Critical (immediate EDD)
    "terrorism":          1.00,
    "terrorist":          1.00,
    "terror":             1.00,
    "sanctions evasion":  0.95,
    "sanctions violation":0.95,
    "weapons":            0.90,
    "proliferation":      0.90,
    # Tier 2 — High
    "money laundering":   0.85,
    "laundering":         0.85,
    "corruption":         0.80,
    "bribery":            0.80,
    "bribe":              0.80,
    "embezzlement":       0.80,
    "organised crime":    0.80,
    "organized crime":    0.80,
    "drug trafficking":   0.80,
    "human trafficking":  0.80,
    # Tier 3 — Medium-High
    "fraud":              0.75,
    "scam":               0.70,
    "ponzi":              0.75,
    "tax evasion":        0.70,
    "tax fraud":          0.70,
    "insider trading":    0.72,
    "market manipulation":0.72,
    "forgery":            0.65,
    # Tier 4 — Medium
    "arrested":           0.55,
    "indicted":           0.60,
    "convicted":          0.65,
    "charged":            0.55,
    "investigated":       0.50,
    "bankruptcy":         0.40,
    "insolvent":          0.40,
    # Tier 5 — Low
    "lawsuit":            0.35,
    "litigation":         0.35,
    "regulatory action":  0.40,
    "fine":               0.30,
    "penalty":            0.30,
    "misconduct":         0.40,
}

_kw_weights: dict[str, float] = dict(DEFAULT_KEYWORD_WEIGHTS)  # mutable at runtime

# Risk tier thresholds
def _media_risk_tier(score: float) -> str:
    if score >= 0.80: return "CRITICAL"
    if score >= 0.55: return "HIGH"
    if score >= 0.30: return "MEDIUM"
    return "LOW"

# ── Schemas ──────────────────────────────────────────────────────────────────

class KYCRequest(BaseModel):
    name: str = Field(..., min_length=2)
    entity_type: str = Field(default="Person", description="Person or Company")
    country: Optional[str] = Field(default=None, description="ISO-2 country code hint")
    dob: Optional[str] = Field(default=None, description="Date of birth YYYY-MM-DD")
    search_media: bool = Field(default=True, description="Also run adverse media scan")
    media_sources: int = Field(default=5, ge=1, le=10, description="# top search results to scan")

class MediaConfigUpdate(BaseModel):
    weights: dict[str, float]


# ── OpenSanctions Website Scraper ─────────────────────────────────────────────

_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _badges_to_topics(badge_texts: list[str]) -> list[str]:
    """Map scraped badge label text → normalised topic keys."""
    topics = []
    for raw in badge_texts:
        key = raw.strip().lower()
        matched = _BADGE_TO_TOPIC.get(key)
        if matched and matched not in topics:
            topics.append(matched)
        # Partial-match fallback for novel labels
        elif not matched:
            for badge_key, topic_val in _BADGE_TO_TOPIC.items():
                if badge_key in key and topic_val not in topics:
                    topics.append(topic_val)
                    break
    return topics

def _topic_to_pep_flag(topic: str) -> dict | None:
    labels = {
        "role.pep":          ("PEP",               "Politically Exposed Person"),
        "role.rca":          ("RCA",               "Relative or Close Associate of PEP"),
        "role.pep.foreign":  ("Foreign PEP",       "Foreign PEP"),
        "role.pep.domestic": ("Domestic PEP",      "Domestic PEP"),
        "role.diplomat":     ("Diplomat",           "Diplomat / IO Official"),
        "poi":               ("Person of Interest","Person of Interest"),
        "sanction":          ("Sanctioned",        "On Sanctions List"),
        "sanction.linked":   ("Sanction-Linked",   "Linked to Sanctioned Entity"),
        "debarment":         ("Debarred",          "Debarred from public procurement"),
        "crime":             ("Crime-Linked",      "Linked to criminal activity"),
        "crime.terror":      ("Terrorism",         "Terrorism-linked"),
        "crime.fin":         ("Fin Crime",         "Financial crime"),
        "export":            ("Trade Risk",        "Export control / trade risk"),
    }
    if topic in labels:
        label, desc = labels[topic]
        return {"tag": topic, "label": label, "description": desc}
    return None

def _name_similarity(query: str, candidate: str) -> float:
    """Simple token-overlap similarity for name matching confidence."""
    q_tokens = set(query.lower().split())
    c_tokens = set(candidate.lower().split())
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens) / len(q_tokens | c_tokens)
    return round(overlap, 3)

async def _scrape_os_search(name: str, entity_type: str = "Person") -> list[dict]:
    """
    Scrape https://www.opensanctions.org/search/?q=<name>
    and parse result cards into structured dicts.

    HTML structure (from inspect element):
      ul.resultList  (class contains 'resultList')
        li.resultItem
          div.resultTitle  → entity name + href to entity page
          p.resultDetails  → "Person · PEP · Counter-sanctioned entity · United States"
    """
    # schema filter: opensanctions supports ?schema=Person or ?schema=LegalEntity
    schema = "LegalEntity" if entity_type.lower() in ("company", "legalentity") else "Person"
    params = {"q": name, "schema": schema}

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True,
                                     headers=_SCRAPE_HEADERS) as client:
            r = await client.get(OS_SEARCH_URL, params=params)
            r.raise_for_status()
    except Exception as e:
        log.warning("OpenSanctions scrape request failed: %s", e)
        return []

    soup = BeautifulSoup(r.text, "lxml")

    # The CSS module class names are hashed (e.g. __DJrSJW__) so we match
    # by partial class name substring — robust across deployments.
    def _find_by_partial(tag, partial: str):
        return soup.find_all(
            tag,
            class_=lambda c: c and any(partial in cls for cls in c.split())
        )

    result_items = _find_by_partial("li", "resultItem")
    if not result_items:
        # Fallback: try any <li> inside a <ul> that has "result" in its class
        result_list = soup.find("ul", class_=lambda c: c and "result" in c.lower())
        result_items = result_list.find_all("li") if result_list else []

    results = []
    for item in result_items[:10]:
        # ── Name ──────────────────────────────────────────────────────────────
        title_div = item.find(
            lambda t: t.name in ("div", "h3", "h4", "a")
            and t.get("class")
            and any("resultTitle" in cls or "Title" in cls for cls in t.get("class", []))
        )
        if not title_div:
            # last resort: first anchor in the item
            title_div = item.find("a")
        if not title_div:
            continue

        entity_name = title_div.get_text(strip=True)
        entity_href = title_div.get("href", "") or (
            title_div.find("a").get("href", "") if title_div.find("a") else ""
        )
        # href is like /entities/Q123456/ — extract the id
        entity_id = ""
        id_match = re.search(r"/entities/([^/]+)/?", entity_href)
        if id_match:
            entity_id = id_match.group(1)

        # ── Details line: "Person · PEP · Counter-sanctioned entity · United States"
        detail_el = item.find(
            lambda t: t.name in ("p", "div", "span")
            and t.get("class")
            and any("Detail" in cls or "detail" in cls or "Meta" in cls for cls in t.get("class", []))
        )
        details_text = detail_el.get_text(separator=" · ", strip=True) if detail_el else ""

        # Parse badges: each separated by " · "
        parts = [p.strip() for p in details_text.split("·") if p.strip()]
        # First part is usually the schema type (Person / Organization)
        schema_label = parts[0] if parts else ""
        badge_labels = parts[1:] if len(parts) > 1 else []
        # Last part is often a country — detect it (no badge styling, plain text)
        country_part = badge_labels[-1] if badge_labels else ""
        country = ""
        # Heuristic: country names won't match any badge key
        if country_part and country_part.lower() not in _BADGE_TO_TOPIC:
            country = country_part
            badge_labels = badge_labels[:-1]

        topics = _badges_to_topics(badge_labels)
        pep_flags = [f for f in (_topic_to_pep_flag(t) for t in topics) if f]
        is_pep = any(t.startswith("role.pep") or t in ("role.rca", "role.diplomat") for t in topics)
        is_sanctioned = "sanction" in topics
        score = _name_similarity(name, entity_name)

        results.append({
            "entity_id":      entity_id,
            "name":           entity_name,
            "schema":         schema_label or schema,
            "score":          score,
            "is_pep":         is_pep,
            "is_sanctioned":  is_sanctioned,
            "pep_categories": pep_flags,
            "topics":         topics,
            "badge_labels":   badge_labels,   # raw badge text for debugging
            "positions":      [],
            "countries":      [country] if country else [],
            "dob":            None,
            "datasets":       [],
            "os_url":         f"{OS_WEBSITE}/entities/{entity_id}/" if entity_id else
                              f"{OS_SEARCH_URL}?q={name}",
        })

    # Sort: PEP/sanctioned first, then by name similarity
    results.sort(key=lambda x: (-(1 if x["is_pep"] or x["is_sanctioned"] else 0), -x["score"]))
    return results

async def screen_pep(name: str, entity_type: str = "Person",
                     country: str = None, dob: str = None) -> dict:
    """
    Main PEP screening function.
    Scrapes https://www.opensanctions.org/search/ for results.
    """
    parsed = await _scrape_os_search(name, entity_type)

    if not parsed:
        return {
            "hit": False,
            "match": None,
            "all_results": [],
            "screened_at": datetime.now(timezone.utc).isoformat(),
        }

    top = parsed[0]
    # A "hit" requires: name similarity ≥ 0.3 AND at least one risk topic
    hit = top["score"] >= 0.30 and (top["is_pep"] or top["is_sanctioned"] or
                                      any(t in ["poi","crime","crime.terror","crime.fin","export"]
                                          for t in top["topics"]))

    # Note: website scraping doesn't give adjacent entity data.
    # If you need RCA mapping, you'd need to follow the entity's detail page.
    adjacent = []

    # Determine EDD risk level
    topics_hit = set(top["topics"])
    if "crime.terror" in topics_hit or "sanction" in topics_hit:
        edd_level = "HIGH — Immediate escalation required"
    elif top["is_pep"] or "role.rca" in topics_hit:
        edd_level = "MEDIUM — Enhanced Due Diligence required"
    elif hit:
        edd_level = "MEDIUM — Additional verification required"
    else:
        edd_level = "STANDARD — No elevated risk detected"

    return {
        "hit":          hit,
        "match":        top if hit else None,
        "edd_level":    edd_level,
        "adjacent":     adjacent,
        "all_results":  parsed[:5],
        "screened_at":  datetime.now(timezone.utc).isoformat(),
    }


# ── Adverse Media Screening ──────────────────────────────────────────────────
# Sources: GDELT DOC 2.0 (primary, no key) → NewsAPI (fallback, optional key)
# Credibility: computed per-article from 5 signals rather than a static allowlist

# ---------------------------------------------------------------------------
# Source config
# ---------------------------------------------------------------------------
# ── API keys (set as environment variables — never hardcode) ─────────────────
NEWSAPI_KEY:    str = os.getenv("NEWSAPI_KEY",    "")   # https://newsapi.org (free, 100 req/day)
GOOGLE_CSE_KEY: str = os.getenv("GOOGLE_CSE_KEY", "")  # Google Cloud Console → Custom Search API
GOOGLE_CSE_CX:  str = os.getenv("GOOGLE_CSE_CX",  "")  # programmablesearchengine.google.com → cx

_GDELT_URL    = "https://api.gdeltproject.org/api/v2/doc/doc"
_NEWSAPI_URL  = "https://newsapi.org/v2/everything"
_GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"

# AML-relevant GDELT themes — used to pre-filter results server-side.
# Full list: https://api.gdeltproject.org/api/v2/guides/GDELT_Global_Knowledge_Graph_Codebook_V2.1.pdf
_GDELT_THEMES = (
    "CRIMEORVIOLENCE,CRIME_FINANCIAL,CRIME_MONEY_LAUNDERING,"
    "SANCTIONS,TERROR,CORRUPTION,TAX_FRAUD,WB_1318_FINANCIAL_CRIME"
)

# Risk keyword search string appended to the name query
_RISK_TERMS = (
    '"money laundering" OR "financial crime" OR corruption OR bribery '
    'OR terrorism OR sanctions OR fraud OR "tax evasion" OR '
    '"organised crime" OR "drug trafficking" OR "human trafficking" OR '
    'indicted OR convicted OR arrested OR embezzlement OR "insider trading"'
)

# ---------------------------------------------------------------------------
# Tranco rank cache (loaded once at startup, optional)
# Tranco top-1M: https://tranco-list.eu/top-1m.csv.zip
# Download and place as tranco_top1m.csv alongside this file for best results.
# Format: rank,domain  (rank 1 = most popular)
# ---------------------------------------------------------------------------
_tranco_ranks: dict[str, int] = {}       # domain → rank (1-based, lower = more popular)
_TRANCO_MAX_RANK = 1_000_000             # anything beyond this is treated as unranked

_TRANCO_CSV_PATH = os.path.join(os.path.dirname(__file__), "tranco_top1m.csv")

def _load_tranco_ranks() -> None:
    """Load Tranco rank CSV into memory once at startup. Silently skips if not present."""
    global _tranco_ranks
    if not os.path.exists(_TRANCO_CSV_PATH):
        log.info("tranco_top1m.csv not found — Tranco rank signal disabled. "
                 "Download from https://tranco-list.eu and place next to kyc_screening.py")
        return
    try:
        loaded: dict[str, int] = {}
        with open(_TRANCO_CSV_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 1)
                if len(parts) == 2:
                    try:
                        rank   = int(parts[0])
                        domain = parts[1].strip().lower()
                        loaded[domain] = rank
                    except ValueError:
                        continue
        _tranco_ranks = loaded
        log.info("Tranco ranks loaded — %d domains", len(_tranco_ranks))
    except Exception as e:
        log.warning("Failed to load Tranco ranks: %s", e)

# Call at import time so it's ready before any requests
_load_tranco_ranks()

# ---------------------------------------------------------------------------
# TLD credibility classes
# ---------------------------------------------------------------------------
# Structurally trusted — only governments/IGOs/academia can register these
_TLD_TRUSTED: frozenset[str] = frozenset({
    "gov", "gov.sg", "gov.uk", "gov.au", "gov.my", "gov.hk",
    "edu", "ac.uk", "edu.sg", "edu.au",
    "mil", "int",   # intergovernmental organisations
})

# Common UGC / blog hosting platforms — hard block regardless of rank
_PLATFORM_BLOG_DOMAINS: frozenset[str] = frozenset({
    "medium.com", "substack.com", "wordpress.com", "blogspot.com",
    "blogger.com", "wix.com", "squarespace.com", "weebly.com",
    "ghost.io", "typepad.com", "tumblr.com",
})

# Social media — hard block
_SOCIAL_DOMAINS: frozenset[str] = frozenset({
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "linkedin.com", "tiktok.com", "reddit.com", "quora.com",
    "pinterest.com", "threads.net", "discord.com", "telegram.org",
})

# ---------------------------------------------------------------------------
# tldextract setup — offline snapshot, no network call needed
# ---------------------------------------------------------------------------
try:
    import tldextract as _tldextract
    _extractor = _tldextract.TLDExtract(suffix_list_urls=[], fallback_to_snapshot=True)
except ImportError:
    _extractor = None
    log.warning("tldextract not installed — pip install tldextract. "
                "Falling back to simple domain parsing.")

def _parse_domain(url: str) -> tuple[str, str, str]:
    """
    Return (registered_domain, domain_only, public_suffix) for a URL.
    e.g. 'https://uk.reuters.com/...' → ('reuters.com', 'reuters', 'com')
    Falls back to naive split if tldextract is unavailable.
    """
    if not url:
        return "", "", ""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        host = host.lower()
        if _extractor:
            r = _extractor(host)
            reg = r.top_domain_under_public_suffix or ""
            return reg, r.domain, r.suffix
        else:
            # Naive fallback: strip www., take last two parts
            if host.startswith("www."):
                host = host[4:]
            parts = host.rsplit(".", 2)
            if len(parts) >= 2:
                return ".".join(parts[-2:]), parts[-2], parts[-1]
            return host, host, ""
    except Exception:
        return "", "", ""

# ---------------------------------------------------------------------------
# Credibility scorer — returns 0.0–1.0
# ---------------------------------------------------------------------------

def _tranco_score(registered_domain: str) -> float:
    """
    Map Tranco rank to a 0–1 credibility score.
    Top-1k → 1.0, top-10k → 0.85, top-100k → 0.65, top-1M → 0.40, unranked → 0.0
    Curve is log-linear so rank differences at the top matter more.
    """
    if not _tranco_ranks or not registered_domain:
        return 0.0
    rank = _tranco_ranks.get(registered_domain)
    if rank is None:
        return 0.0
    import math
    # log10(1)=0, log10(1M)=6 → invert and normalise
    score = 1.0 - (math.log10(max(rank, 1)) / math.log10(_TRANCO_MAX_RANK))
    return round(max(0.0, min(score, 1.0)), 3)

def _tld_score(suffix: str) -> float:
    """
    Score based on TLD class.
    Trusted gov/edu/int → 1.0, generic .com/.org/.net → 0.5,
    known high-risk ccTLDs → 0.2, everything else → 0.4
    """
    if not suffix:
        return 0.3
    if suffix in _TLD_TRUSTED:
        return 1.0
    generic_trusted = {"com", "org", "net", "co", "io"}
    # Country codes with decent regulatory environments
    reputable_cc = {
        "co.uk", "com.au", "com.sg", "com.my", "com.hk",
        "co.nz", "co.jp", "de", "fr", "nl", "se", "no", "dk", "fi",
    }
    # High-risk / known spam ccTLDs
    risky_cc = {"ru", "cn", "kp", "ir", "sy", "cu", "ve", "xyz", "tk", "ml", "ga", "cf"}
    if suffix in generic_trusted:
        return 0.5
    if suffix in reputable_cc:
        return 0.6
    if suffix in risky_cc:
        return 0.2
    return 0.4

def _content_quality_score(soup: "BeautifulSoup | None", word_count: int) -> float:
    """
    Score article body quality from 0–1.
      - Word count: thin (<150) articles score low; rich (>400) score high
      - Ad/tracker ratio: count <script>/<ins>/<iframe> vs <p> tags
    Returns 0.0 if soup is None (title-only pass).
    """
    if soup is None:
        return 0.0  # unknown — will be excluded from weighted average

    # Word count signal
    if word_count < 100:
        wc_score = 0.1
    elif word_count < 200:
        wc_score = 0.4
    elif word_count < 400:
        wc_score = 0.65
    else:
        wc_score = min(1.0, 0.7 + (word_count - 400) / 2000)

    # Ad/noise ratio signal: scripts + iframes vs paragraphs
    n_scripts = len(soup.find_all(["script", "ins", "iframe"]))
    n_paras   = len(soup.find_all("p"))
    if n_paras == 0:
        noise_score = 0.2
    else:
        ratio = n_scripts / (n_paras + n_scripts)
        noise_score = round(max(0.0, 1.0 - ratio * 1.5), 3)

    return round(wc_score * 0.6 + noise_score * 0.4, 3)

def _byline_score(soup: "BeautifulSoup | None") -> float:
    """
    Detect journalist byline presence. Real news outlets almost always have one.
    Returns 1.0 if found, 0.0 if not, None if soup unavailable.
    """
    if soup is None:
        return 0.0
    # Schema.org structured data is the most reliable signal
    schema_author = soup.find("span", itemprop="author") or \
                    soup.find("a",    itemprop="author") or \
                    soup.find("meta", itemprop="author")
    if schema_author:
        return 1.0
    # Common CSS class / attribute patterns
    byline_signals = [
        soup.find(attrs={"class": re.compile(r"byline|author|reporter|journalist", re.I)}),
        soup.find(attrs={"rel": "author"}),
        soup.find("address"),   # HTML5 article author convention
    ]
    return 1.0 if any(byline_signals) else 0.3

def _source_url_match_score(rss_source_label: str, resolved_domain: str) -> float:
    """
    Check whether the RSS <source> label plausibly matches the resolved domain.
    If a feed claims to be 'Reuters' but resolves to 'sketchy-blog.com', that's a red flag.
    Returns 1.0 (match), 0.5 (uncertain / no label), 0.1 (clear mismatch).
    """
    if not rss_source_label or not resolved_domain:
        return 0.5
    # Normalise: lowercase, strip common suffixes
    label = re.sub(r"\s+", "", rss_source_label.lower())
    label = re.sub(r"(news|media|online|digital|times|post|daily|the|sg|com)$", "", label)
    domain_core = resolved_domain.split(".")[0]  # 'reuters' from 'reuters.com'
    # Simple containment check — not perfect but catches obvious mismatches
    if label in domain_core or domain_core in label:
        return 1.0
    # Both empty / very short → uncertain
    if len(label) < 3 or len(domain_core) < 3:
        return 0.5
    return 0.4   # label present but doesn't match domain — mild penalty

def compute_source_credibility(
    url: str,
    soup: "BeautifulSoup | None" = None,
    word_count: int = 0,
    rss_source_label: str = "",
    resolved_url: str = "",
) -> dict:
    """
    Compute composite source credibility (0.0–1.0).
 
    Priority order:
      1. Hard block  → score 0.0, tier "blocked",  multiplier 0.0
      2. Whitelist   → override score, tier = whitelist tier, multiplier 1.0
      3. Normal path → 5-signal weighted average (unchanged)
    """
    reg_domain, domain_only, suffix = _parse_domain(url)
    resolved_reg, _, _ = _parse_domain(resolved_url) if resolved_url else ("", "", "")
    effective_domain = resolved_reg or reg_domain
 
    # ── 1. Hard blocks ────────────────────────────────────────────────────
    for blocked_set in (_SOCIAL_DOMAINS, _PLATFORM_BLOG_DOMAINS):
        for bd in blocked_set:
            if effective_domain == bd or effective_domain.endswith("." + bd):
                return {
                    "score":      0.0,
                    "tier":       "blocked",
                    "signals":    {"hard_block": bd},
                    "multiplier": 0.0,
                }
 
    # ── 2. Whitelist override ─────────────────────────────────────────────
    wl_entry = get_whitelist_entry(effective_domain)
    if wl_entry:
        wl_tier, wl_score = wl_entry
        return {
            "score":      wl_score,
            "tier":       "high",           # always high for whitelisted sources
            "signals":    {
                "whitelist_tier":  wl_tier,
                "whitelist_score": wl_score,
                # Still compute these for auditability / future use
                "tranco_rank":     _tranco_score(effective_domain),
                "tld_class":       _tld_score(suffix),
                "content_quality": _content_quality_score(soup, word_count),
                "byline":          _byline_score(soup),
                "source_match":    _source_url_match_score(rss_source_label, effective_domain),
            },
            "multiplier": 1.0,              # never penalise whitelisted sources
            "whitelisted": True,            # flag for downstream logging
        }
 
    # ── 3. Normal 5-signal scoring (unchanged from original) ─────────────
    s_tranco  = _tranco_score(effective_domain)
    s_tld     = _tld_score(suffix)
    s_content = _content_quality_score(soup, word_count)
    s_byline  = _byline_score(soup)
    s_match   = _source_url_match_score(rss_source_label, effective_domain)
 
    signals = {
        "tranco_rank":     s_tranco,
        "tld_class":       s_tld,
        "content_quality": s_content,
        "byline":          s_byline,
        "source_match":    s_match,
    }
 
    if _tranco_ranks:
        score = (
            s_tranco  * 0.30 +
            s_tld     * 0.20 +
            s_content * 0.20 +
            s_byline  * 0.15 +
            s_match   * 0.15
        )
    else:
        score = (
            s_tld     * 0.35 +
            s_content * 0.30 +
            s_byline  * 0.20 +
            s_match   * 0.15
        )
 
    score = round(min(max(score, 0.0), 1.0), 3)
 
    if score >= 0.70:   tier = "high"
    elif score >= 0.40: tier = "medium"
    else:               tier = "low"
 
    multiplier = {"high": 1.0, "medium": 0.75, "low": 0.40}.get(tier, 0.40)
 
    return {
        "score":      score,
        "tier":       tier,
        "signals":    signals,
        "multiplier": multiplier,
        "whitelisted": False,
    }
 

# ---------------------------------------------------------------------------
# News fetchers — GDELT primary, NewsAPI fallback
# ---------------------------------------------------------------------------

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-SG,en;q=0.9",
}

async def _fetch_gdelt(name: str, max_items: int = 15) -> list[dict]:
    """
    Query GDELT DOC 2.0 API.
    No API key needed. Returns structured JSON with domain, tone, and URL.
    Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

    Key params used:
      query      — name + risk keywords
      mode       = artlist  (article list, not timeline or wordcloud)
      maxrecords = up to 250
      format     = json
      timespan   = timespan e.g. "3months"
      sort       = datedesc
    """
    query = f'"{name}" ({_RISK_TERMS})'
    params = {
        "query":      query,
        "mode":       "artlist",
        "maxrecords": str(max_items),
        "format":     "json",
        "timespan":   "3months",
        "sort":       "datedesc",
        "trans":      "googtrans",   # auto-translate non-English titles
    }
    try:
        async with httpx.AsyncClient(timeout=15, headers=_FETCH_HEADERS,
                                     follow_redirects=True) as client:
            r = await client.get(_GDELT_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning("GDELT fetch failed: %s", e)
        return []

    articles = []
    for art in data.get("articles", []):
        url = art.get("url", "")
        if not url:
            continue
        # GDELT tone: negative = bad sentiment; below -5 is notably negative
        tone = art.get("tone", 0.0)
        try:
            tone = float(tone)
        except (TypeError, ValueError):
            tone = 0.0

        articles.append({
            "title":       art.get("title", ""),
            "url":         url,
            "date":        art.get("seendate", "")[:16],    # YYYYMMDDTHHMMSS → trim
            "source":      art.get("domain", ""),           # GDELT gives domain directly
            "snippet":     art.get("socialimage", ""),      # reuse field; will be replaced by scrape
            "gdelt_tone":  round(tone, 2),
            "_gdelt":      True,
        })
    log.info("GDELT returned %d articles for '%s'", len(articles), name)
    return articles


async def _fetch_google_cse(name: str, max_items: int = 10) -> list[dict]:
    """
    Google Custom Search API — requires free API key + cx.
    Falls back silently if keys not set.
    Setup: https://console.cloud.google.com → Custom Search API
           https://programmablesearchengine.google.com → cx
    export GOOGLE_CSE_KEY="AIza..."  GOOGLE_CSE_CX="12345:abc..."
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return []

    query = f'"{name}" ({_RISK_TERMS})'
    articles = []

    for start in range(1, max_items + 1, 10):
        num = min(10, max_items - len(articles))
        params = {
            "key":    GOOGLE_CSE_KEY,
            "cx":     GOOGLE_CSE_CX,
            "q":      query,
            "num":    num,
            "start":  start,
            "lr":     "lang_en",
            "safe":   "off",
            "fields": "items(title,link,snippet,displayLink,pagemap/metatags)",
        }
        try:
            async with httpx.AsyncClient(timeout=12, headers=_FETCH_HEADERS,
                                         follow_redirects=True) as client:
                r = await client.get(_GOOGLE_CSE_URL, params=params)
                if r.status_code == 429:
                    log.warning("Google CSE: quota exceeded")
                    break
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            log.warning("Google CSE fetch failed: %s", e)
            break

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            url = item.get("link", "")
            if not url:
                continue
            metatags = (item.get("pagemap") or {}).get("metatags", [{}])
            date_raw = ""
            for tag in metatags:
                date_raw = (tag.get("article:published_time") or
                            tag.get("og:updated_time") or
                            tag.get("date") or "")
                if date_raw:
                    break
            articles.append({
                "title":   item.get("title", ""),
                "url":     url,
                "date":    date_raw[:16] if date_raw else "",
                "source":  item.get("displayLink", ""),
                "snippet": item.get("snippet", ""),
                "_gdelt":  False,
                "_google": True,
                "_source": "google_cse",
            })
        if len(articles) >= max_items:
            break

    log.info("Google CSE returned %d articles for '%s'", len(articles), name)
    return articles


async def _fetch_ddg(name: str, max_items: int = 10) -> list[dict]:
    """
    DuckDuckGo HTML scraper — no API key, no setup, works out of the box.
    Scrapes https://html.duckduckgo.com/html/ which returns clean result cards
    with no JavaScript and no CAPTCHA.

    DDG is more scraper-friendly than Google but results are slightly less
    comprehensive. Used as the primary no-key source.
    """
    # DDG HTML endpoint — add risk terms to pre-filter for adverse content
    risk_short = (
        "fraud OR laundering OR corruption OR terrorism OR sanctions OR "
        "bribery OR convicted OR arrested OR indicted OR embezzlement"
    )
    query = f'"{name}" ({risk_short})'

    _DDG_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://duckduckgo.com/",
    }

    try:
        async with httpx.AsyncClient(
            timeout=15, follow_redirects=True, headers=_DDG_HEADERS
        ) as client:
            r = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": "us-en", "kp": "-2"},  # kp=-2 = safe off
            )
            r.raise_for_status()
    except Exception as e:
        log.warning("DDG scrape failed: %s", e)
        return []

    soup = BeautifulSoup(r.text, "lxml")

    # DDG HTML result structure:
    # <div class="result results_links results_links_deep web-result">
    #   <a class="result__a" href="...">Title</a>
    #   <a class="result__url">domain.com</a>
    #   <div class="result__snippet">snippet text</div>
    # </div>
    result_divs = soup.find_all("div", class_=lambda c: c and "result" in c and "web-result" in c)

    # Fallback: find any result__a links if structure differs
    if not result_divs:
        result_divs = soup.find_all("div", class_="result")

    articles = []
    for div in result_divs[:max_items]:
        title_el   = div.find("a", class_="result__a")
        url_el     = div.find("a", class_="result__url")
        snippet_el = div.find("div", class_="result__snippet")

        if not title_el:
            continue

        title = title_el.get_text(strip=True)
        # DDG wraps URLs in a redirect — get the actual href
        raw_href = title_el.get("href", "")
        # Extract real URL from DDG redirect: //duckduckgo.com/l/?uddg=https%3A...
        url = raw_href
        if "uddg=" in raw_href:
            from urllib.parse import unquote, urlparse, parse_qs
            try:
                qs = parse_qs(urlparse(raw_href).query)
                url = unquote(qs.get("uddg", [raw_href])[0])
            except Exception:
                pass
        elif raw_href.startswith("//"):
            url = "https:" + raw_href

        if not url or url.startswith("javascript"):
            continue

        source = ""
        if url_el:
            source = url_el.get_text(strip=True)
        else:
            # Extract domain from URL
            try:
                from urllib.parse import urlparse as _up
                source = _up(url).hostname or ""
            except Exception:
                pass

        snippet = snippet_el.get_text(strip=True) if snippet_el else ""

        articles.append({
            "title":   title,
            "url":     url,
            "date":    "",
            "source":  source,
            "snippet": snippet,
            "_gdelt":  False,
            "_google": False,
            "_source": "ddg",
        })

    log.info("DDG returned %d articles for '%s'", len(articles), name)
    return articles


async def _fetch_google_scrape(name: str, max_items: int = 10) -> list[dict]:
    """
    Google Search HTML scraper — no API key needed.
    Less reliable than DDG (CAPTCHAs possible under heavy use) but broader
    coverage. Used as secondary no-key source after DDG.

    Rotates user agents to reduce detection. If Google returns a CAPTCHA
    page, returns empty list silently and lets DDG results stand.
    """
    import random
    risk_short = (
        "fraud OR laundering OR corruption OR terrorism OR sanctions OR "
        "bribery OR convicted OR arrested OR indicted OR embezzlement"
    )
    query = f'"{name}" ({risk_short})'

    _UA_POOL = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    ]
    headers = {
        "User-Agent":      random.choice(_UA_POOL),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT":             "1",
        "Connection":      "keep-alive",
    }
    params = {
        "q":    query,
        "num":  min(max_items, 10),
        "hl":   "en",
        "gl":   "sg",    # Singapore region for SEA relevance
        "safe": "off",
    }

    try:
        async with httpx.AsyncClient(
            timeout=15, follow_redirects=True, headers=headers
        ) as client:
            r = await client.get("https://www.google.com/search", params=params)
            r.raise_for_status()
    except Exception as e:
        log.warning("Google scrape failed: %s", e)
        return []

    # Check for CAPTCHA
    if "captcha" in r.text.lower() or "unusual traffic" in r.text.lower():
        log.warning("Google scrape: CAPTCHA detected — skipping Google scrape results")
        return []

    soup = BeautifulSoup(r.text, "lxml")

    # Google result structure (class names vary but these patterns are stable):
    # <div class="g"> or <div data-hveid="...">
    #   <h3>Title</h3>
    #   <a href="/url?q=https://...">
    #   <div>snippet</div>
    # </div>
    articles = []

    # Find all result containers — Google uses <div class="g"> consistently
    result_blocks = soup.find_all("div", class_="g")
    if not result_blocks:
        # Fallback: any div with an h3 and an anchor
        result_blocks = [d for d in soup.find_all("div") if d.find("h3") and d.find("a")]

    for block in result_blocks[:max_items]:
        h3 = block.find("h3")
        if not h3:
            continue
        title = h3.get_text(strip=True)
        if not title:
            continue

        # Extract URL from Google redirect wrapper
        anchor = block.find("a", href=True)
        raw_href = anchor["href"] if anchor else ""
        url = raw_href
        if raw_href.startswith("/url?"):
            from urllib.parse import unquote, urlparse, parse_qs
            try:
                qs = parse_qs(urlparse(raw_href).query)
                url = unquote(qs.get("q", [raw_href])[0])
            except Exception:
                pass

        if not url or url.startswith("/search") or url.startswith("javascript"):
            continue

        # Snippet: first non-empty <span> or <div> after the title
        snippet = ""
        for el in block.find_all(["span", "div"]):
            txt = el.get_text(strip=True)
            if len(txt) > 40 and txt != title:
                snippet = txt[:300]
                break

        # Source domain
        try:
            from urllib.parse import urlparse as _up
            source = _up(url).hostname or ""
            if source.startswith("www."):
                source = source[4:]
        except Exception:
            source = ""

        articles.append({
            "title":   title,
            "url":     url,
            "date":    "",
            "source":  source,
            "snippet": snippet,
            "_gdelt":  False,
            "_google": True,
            "_source": "google_scrape",
        })

    log.info("Google scrape returned %d articles for '%s'", len(articles), name)
    return articles


async def _fetch_newsapi(name: str, max_items: int = 10) -> list[dict]:
    """
    NewsAPI fallback — only runs if NEWSAPI_KEY env var is set.
    Free tier: 100 req/day, articles up to 1 month old.
    Sign up at https://newsapi.org (free).
    """
    if not NEWSAPI_KEY:
        return []
    query = f'"{name}" AND ({_RISK_TERMS})'
    params = {
        "q":        query,
        "language": "en",
        "sortBy":   "relevancy",
        "pageSize": str(min(max_items, 100)),
        "apiKey":   NEWSAPI_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=12, headers=_FETCH_HEADERS,
                                     follow_redirects=True) as client:
            r = await client.get(_NEWSAPI_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning("NewsAPI fetch failed: %s", e)
        return []

    articles = []
    for art in data.get("articles", []):
        url = art.get("url", "")
        if not url or url == "https://removed.com":
            continue
        src = art.get("source", {})
        articles.append({
            "title":   art.get("title", "") or "",
            "url":     url,
            "date":    (art.get("publishedAt", "") or "")[:16],
            "source":  src.get("name", "") if isinstance(src, dict) else str(src),
            "snippet": art.get("description", "") or "",
            "_gdelt":  False,
        })
    log.info("NewsAPI returned %d articles for '%s'", len(articles), name)
    return articles


async def _scrape_article(url: str, max_chars: int = 800) -> dict:
    """
    Fetch a news article, extract body text, and collect quality signals.
    Returns:
        {
          "text":       str,           # cleaned body text up to max_chars
          "word_count": int,
          "soup":       BeautifulSoup | None,
          "resolved_url": str,         # final URL after redirects
        }
    """
    empty = {"text": "", "word_count": 0, "soup": None, "resolved_url": url}
    if not url or url.startswith("javascript"):
        return empty
    try:
        async with httpx.AsyncClient(
            timeout=10, follow_redirects=True, headers=_FETCH_HEADERS
        ) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return {**empty, "resolved_url": str(r.url)}
            resolved_url = str(r.url)
            soup = BeautifulSoup(r.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "ins", "iframe", "noscript"]):
                tag.decompose()
            body = soup.find("article") or soup.find("main") or soup.body
            if not body:
                return {**empty, "resolved_url": resolved_url, "soup": soup}
            text = re.sub(r"\s+", " ", body.get_text(separator=" ")).strip()
            word_count = len(text.split())
            return {
                "text":         text[:max_chars],
                "word_count":   word_count,
                "soup":         soup,
                "resolved_url": resolved_url,
            }
    except Exception as e:
        log.debug("Article scrape failed for %s: %s", url[:60], e)
        return empty


# ---------------------------------------------------------------------------
# Keyword scorer (unchanged interface)
# ---------------------------------------------------------------------------

def _score_text(text: str, kw_weights: dict[str, float]) -> tuple[float, list[dict]]:
    """
    Scan text for risk keywords.
    Returns (max_weighted_score, list of keyword match dicts).
    """
    text_lower = text.lower()
    matches = []
    seen: set[str] = set()
    for kw, w in sorted(kw_weights.items(), key=lambda x: -x[1]):
        if kw in text_lower and kw not in seen:
            seen.add(kw)
            count   = text_lower.count(kw)
            boosted = min(w + (count - 1) * 0.03, 1.0)
            matches.append({"keyword": kw, "weight": w, "count": count,
                             "boosted_score": round(boosted, 3)})
    if not matches:
        return 0.0, []
    return round(max(m["boosted_score"] for m in matches), 3), matches


# ---------------------------------------------------------------------------
# Main adverse media pipeline
# ---------------------------------------------------------------------------

async def run_adverse_media(name: str, max_sources: int = 5) -> dict:
    """
    Adverse media screening pipeline:

    1. Fetch from GDELT (primary) → fallback to NewsAPI if GDELT returns nothing
    2. Title-pass keyword score on all results (fast, no extra network calls)
    3. Sort by title score; scrape body of top max_sources articles concurrently
    4. For each scraped article: compute source credibility from 5 signals,
       then final_score = keyword_score × credibility_multiplier
    5. Articles scoring 0.0 after credibility adjustment are excluded from flagged list
    6. Return ranked results with per-article credibility breakdown
    """
    # ── Step 1: Fetch from all sources, merge, deduplicate ───────────────────
    # Order: Google CSE (API, best) → DDG scrape (free, reliable) →
    #        Google scrape (free, broader) → GDELT (academic) → NewsAPI (fallback)
    # All no-key sources run always; results merged and deduped by URL.
    articles: list[dict] = []
    fetch_sources: list[str] = []
    seen_urls: set[str] = set()

    def _merge(new_arts: list[dict], source_label: str) -> int:
        added = 0
        for art in new_arts:
            url = art.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                articles.append(art)
                added += 1
        if added:
            fetch_sources.append(source_label)
        return added

    # 1. Google CSE (API key — best quality, if configured)
    if GOOGLE_CSE_KEY and GOOGLE_CSE_CX:
        cse_arts = await _fetch_google_cse(name, max_items=max_sources * 3)
        _merge(cse_arts, "google_cse")

    # 2. DuckDuckGo scrape (no key, reliable, runs always)
    ddg_arts = await _fetch_ddg(name, max_items=max_sources * 3)
    _merge(ddg_arts, "ddg")

    # 3. Google scrape (no key, broader coverage, may hit CAPTCHA)
    goog_arts = await _fetch_google_scrape(name, max_items=max_sources * 2)
    _merge(goog_arts, "google_scrape")

    # 4. GDELT (no key, academic index)
    gdelt_arts = await _fetch_gdelt(name, max_items=max_sources * 2)
    _merge(gdelt_arts, "gdelt")

    # 5. NewsAPI (optional API key fallback)
    if not articles or len(articles) < max_sources:
        newsapi_arts = await _fetch_newsapi(name, max_items=max_sources * 2)
        _merge(newsapi_arts, "newsapi")

    fetch_source = "+".join(fetch_sources) if fetch_sources else "none"
    log.info("Total articles after merge+dedup: %d from [%s]", len(articles), fetch_source)

    if not articles:
        return {
            "overall_score":   0.0,
            "risk_tier":       "LOW",
            "articles_scanned": 0,
            "articles_dropped": 0,
            "fetch_source":    "none",
            "flagged":         [],
            "all_articles":    [],
            "screened_at":     datetime.now(timezone.utc).isoformat(),
        }

    # ── Step 2: Title-pass credibility pre-filter + keyword score ─────────────
    # Hard-block social/blog domains before we waste any scrape budget
    pre_filtered = []
    dropped = 0
    for art in articles:
        cred = compute_source_credibility(url=art["url"], rss_source_label=art["source"])
        if cred["tier"] == "blocked":
            dropped += 1
            log.debug("Dropped blocked source: %s", art.get("source", art["url"][:40]))
            continue
        title_score, title_kws = _score_text(art["title"], _kw_weights)
        pre_filtered.append({
            **art,
            "title_score": title_score,
            "title_kws":   title_kws,
            "_cred_pre":   cred,         # title-pass credibility (no soup yet)
        })

    if not pre_filtered:
        return {
            "overall_score":    0.0,
            "risk_tier":        "LOW",
            "articles_scanned": 0,
            "articles_dropped": dropped,
            "fetch_source":     fetch_source,
            "flagged":          [],
            "all_articles":     [],
            "screened_at":      datetime.now(timezone.utc).isoformat(),
        }

    # Sort by title score; take top max_sources for full scrape
    pre_filtered.sort(key=lambda x: (
        -(1 if x["_cred_pre"].get("whitelisted") else 0),
        -x["title_score"]
    ))
    to_scrape = pre_filtered[:max_sources]
    title_only = pre_filtered[max_sources:]

    # ── Step 3: Concurrent article scrape ─────────────────────────────────────
    scrape_results = await asyncio.gather(
        *[_scrape_article(a["url"]) for a in to_scrape],
        return_exceptions=True,
    )

    # ── Step 4: Score scraped articles ────────────────────────────────────────
    results = []

    for art, scrape in zip(to_scrape, scrape_results):
        if isinstance(scrape, Exception):
            scrape = {"text": "", "word_count": 0, "soup": None, "resolved_url": art["url"]}

        full_text  = art["title"] + " " + scrape["text"]
        kw_score, kws = _score_text(full_text, _kw_weights)

        # Re-compute credibility now we have soup + resolved URL
        cred = compute_source_credibility(
            url=art["url"],
            soup=scrape["soup"],
            word_count=scrape["word_count"],
            rss_source_label=art["source"],
            resolved_url=scrape["resolved_url"],
        )

        final_score = round(kw_score * cred["multiplier"], 3)

        results.append({
            "title":              art["title"],
            "url":                art["url"],
            "source":             art["source"],
            "date":               art["date"],
            "snippet":            scrape["text"][:300],
            "score":              final_score,
            "keyword_score":      kw_score,
            "risk_tier":          _media_risk_tier(final_score),
            "keywords_hit":       sorted(kws, key=lambda x: -x["boosted_score"])[:5],
            "credibility":        cred,
            "gdelt_tone":         art.get("gdelt_tone"),
            "word_count":         scrape["word_count"],
        })

    # ── Step 5: Title-only articles (no scrape budget) ────────────────────────
    for art in title_only:
        cred = art["_cred_pre"]
        final_score = round(art["title_score"] * cred["multiplier"], 3)
        results.append({
            "title":         art["title"],
            "url":           art["url"],
            "source":        art["source"],
            "date":          art["date"],
            "snippet":       "",
            "score":         final_score,
            "keyword_score": art["title_score"],
            "risk_tier":     _media_risk_tier(final_score),
            "keywords_hit":  sorted(art["title_kws"], key=lambda x: -x["boosted_score"])[:5],
            "credibility":   cred,
            "gdelt_tone":    art.get("gdelt_tone"),
            "word_count":    0,
        })

    results.sort(key=lambda x: -x["score"])
    flagged = [
    r for r in results
        if r["score"] > 0 or r["credibility"].get("whitelisted")
    ]
    overall = max((r["score"] for r in flagged), default=0.0)

    return {
        "overall_score":    round(overall, 3),
        "risk_tier":        _media_risk_tier(overall),
        "articles_scanned": len(results),
        "articles_dropped": dropped,
        "fetch_source":     fetch_source,
        "flagged":          flagged,
        "all_articles":     results,
        "screened_at":      datetime.now(timezone.utc).isoformat(),
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/screen")
async def kyc_full_screen(req: KYCRequest):
    """
    Combined KYC screening:
    - PEP check via OpenSanctions
    - Adverse media via Google News RSS + keyword scoring
    Returns unified risk profile.
    """
    pep_task = screen_pep(req.name, req.entity_type, req.country, req.dob)
    media_task = run_adverse_media(req.name, req.media_sources) if req.search_media else None

    if media_task:
        pep_result, media_result = await asyncio.gather(pep_task, media_task)
    else:
        pep_result = await pep_task
        media_result = {"overall_score": 0, "risk_tier": "N/A", "flagged": [], "articles_scanned": 0}

    # Unified risk score: max(pep_score_component, media_score)
    pep_score_map = {"HIGH — Immediate escalation required": 0.9,
                     "MEDIUM — Enhanced Due Diligence required": 0.6,
                     "MEDIUM — Additional verification required": 0.5,
                     "STANDARD — No elevated risk detected": 0.0}
    pep_num = pep_score_map.get(pep_result.get("edd_level", ""), 0.0)
    media_num = media_result.get("overall_score", 0.0)
    combined = round(max(pep_num, media_num * 0.85 + pep_num * 0.15), 3)

    if combined >= 0.80:    combined_tier = "CRITICAL"
    elif combined >= 0.55:  combined_tier = "HIGH"
    elif combined >= 0.30:  combined_tier = "MEDIUM"
    else:                   combined_tier = "LOW"

    return {
        "name":              req.name,
        "combined_score":    combined,
        "combined_tier":     combined_tier,
        "pep":               pep_result,
        "adverse_media":     media_result,
        "screened_at":       datetime.now(timezone.utc).isoformat(),
        "scrape_mode":       True,   # using website scraping, no API key needed
    }

@router.post("/pep")
async def pep_only(req: KYCRequest):
    """PEP screening only (no media scan)."""
    return await screen_pep(req.name, req.entity_type, req.country, req.dob)

@router.post("/media")
async def media_only(req: KYCRequest):
    """Adverse media only."""
    return await run_adverse_media(req.name, req.media_sources)

@router.get("/media/config")
def get_media_config():
    """Return current keyword weights and whitelist summary."""
    categories = {}
    for kw, w in sorted(_kw_weights.items(), key=lambda x: -x[1]):
        tier = ("Critical" if w >= 0.90 else "High" if w >= 0.70
                else "Medium" if w >= 0.45 else "Low")
        categories.setdefault(tier, []).append({"keyword": kw, "weight": w})
 
    return {
        "weights":   _kw_weights,
        "by_tier":   categories,
        "whitelist": whitelist_summary(),   # ← new
    }
@router.get("/media/debug")
async def debug_media(name: str = "test"):
    """
    Diagnostic endpoint — shows raw fetch counts from all sources.
    Hit: GET /kyc/media/debug?name=YourTestName
    """
    ddg_arts, goog_arts, gdelt_arts, newsapi_arts, cse_arts = await asyncio.gather(
        _fetch_ddg(name, max_items=10),
        _fetch_google_scrape(name, max_items=10),
        _fetch_gdelt(name, max_items=10),
        _fetch_newsapi(name, max_items=10),
        _fetch_google_cse(name, max_items=10),
    )

    def _prefilter(arts: list[dict]) -> tuple[list, int]:
        passed, dropped = [], 0
        for art in arts:
            cred = compute_source_credibility(url=art["url"], rss_source_label=art["source"])
            if cred["tier"] == "blocked":
                dropped += 1
            else:
                title_score, _ = _score_text(art["title"], _kw_weights)
                passed.append({
                    "title":       art["title"],
                    "url":         art["url"],
                    "source":      art["source"],
                    "title_score": title_score,
                    "cred_tier":   cred["tier"],
                    "whitelisted": cred.get("whitelisted", False),
                    "_fetch_source": art.get("_source", "unknown"),
                })
        return passed, dropped

    ddg_f,   ddg_d   = _prefilter(ddg_arts)
    goog_f,  goog_d  = _prefilter(goog_arts)
    gdelt_f, gdelt_d = _prefilter(gdelt_arts)
    cse_f,   cse_d   = _prefilter(cse_arts)

    return {
        "google_cse_enabled":    bool(GOOGLE_CSE_KEY and GOOGLE_CSE_CX),
        "newsapi_enabled":       bool(NEWSAPI_KEY),
        "ddg_raw":               len(ddg_arts),
        "google_scrape_raw":     len(goog_arts),
        "gdelt_raw":             len(gdelt_arts),
        "newsapi_raw":           len(newsapi_arts),
        "google_cse_raw":        len(cse_arts),
        "ddg_after_filter":      len(ddg_f),
        "google_scrape_after_filter": len(goog_f),
        "gdelt_after_filter":    len(gdelt_f),
        "google_cse_after_filter": len(cse_f),
        "ddg_articles":          ddg_f,
        "google_scrape_articles": goog_f,
        "gdelt_articles":        gdelt_f,
        "google_cse_articles":   cse_f,
    }
@router.put("/media/config")
def update_media_config(body: MediaConfigUpdate):
    """
    Update keyword weights at runtime — no restart needed.
    Analysts can add new keywords or adjust weights.
    """
    global _kw_weights
    # Validate all weights are in [0,1]
    for kw, w in body.weights.items():
        if not (0.0 <= w <= 1.0):
            raise HTTPException(400, f"Weight for '{kw}' must be between 0 and 1")
    _kw_weights = {**DEFAULT_KEYWORD_WEIGHTS, **body.weights}
    log.info("Adverse media keyword config updated — %d keywords", len(_kw_weights))
    return {"status": "updated", "keyword_count": len(_kw_weights), "weights": _kw_weights}

@router.post("/media/config/reset")
def reset_media_config():
    """Reset keyword weights to factory defaults."""
    global _kw_weights
    _kw_weights = dict(DEFAULT_KEYWORD_WEIGHTS)
    return {"status": "reset", "keyword_count": len(_kw_weights)}
 
@router.get("/media/whitelist")
def get_whitelist():
    """
    Return all whitelisted domains grouped by tier.
    Useful for analysts to audit what sources are trusted.
    """
    by_tier: dict[str, list[str]] = {}
    for domain, (tier, score) in sorted(_WHITELIST_MAP.items()):
        by_tier.setdefault(tier, []).append({
            "domain": domain,
            "credibility_override": score,
        })
    return {
        "total":   len(_WHITELIST_MAP),
        "by_tier": by_tier,
    }