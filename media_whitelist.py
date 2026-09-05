# ---------------------------------------------------------------------------
# media_whitelist.py
# Trusted news source whitelist for adverse media screening.
#
# HOW IT WORKS IN THE PIPELINE:
#   TIER 1 (whitelist_boost):   domain is a known credible outlet
#                                → credibility score forced to 0.90+
#                                → multiplier = 1.0 always
#   TIER 2 (neutral):            domain not in whitelist, not blocked
#                                → normal credibility scoring applies
#   TIER 3 (blocked):            _SOCIAL_DOMAINS / _PLATFORM_BLOG_DOMAINS
#                                → score = 0.0, dropped immediately
#
# HOW TO ADD SITES:
#   Add the registered domain (no www, no https) to the appropriate category.
#   Use the format: "domain.tld"  e.g. "reuters.com"
#   For country-specific domains: "bbc.co.uk", "straitstimes.com"
# ---------------------------------------------------------------------------

from __future__ import annotations

# ---------------------------------------------------------------------------
# Whitelist tiers — edit these freely
# ---------------------------------------------------------------------------

# Tier A — Major international wire services & broadcasters
# Credibility override: 0.95
WIRE_SERVICES: frozenset[str] = frozenset({
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "afp.com",
    "ft.com",
    "economist.com",
    "wsj.com",
    "nytimes.com",
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "washingtonpost.com",
    "aljazeera.com",
    "dw.com",
    "france24.com",
})

# Tier B — Regional / national newspapers & broadcasters
# Credibility override: 0.82
REGIONAL_PRESS: frozenset[str] = frozenset({
    # Singapore & SEA
    "straitstimes.com",
    "channelnewsasia.com",
    "businesstimes.com.sg",
    "todayonline.com",
    "thestar.com.my",
    "malaymail.com",
    "bangkokpost.com",
    "philstar.com",
    "vnexpress.net",
    # Asia-Pacific
    "scmp.com",
    "japantimes.co.jp",
    "koreatimes.co.kr",
    "thehindu.com",
    "hindustantimes.com",
    "smh.com.au",
    "afr.com",
    # Europe
    "spiegel.de",
    "lemonde.fr",
    "elpais.com",
    # Middle East
    "arabnews.com",
    "gulfnews.com",
})

# Tier C — Financial / regulatory / legal news
# Credibility override: 0.88
FINANCIAL_LEGAL: frozenset[str] = frozenset({
    "law360.com",
    "globalwitness.org",
    "occrp.org",           # Organised Crime and Corruption Reporting Project
    "icij.org",            # International Consortium of Investigative Journalists
    "transparency.org",
    "fatf-gafi.org",
    "acams.org",
    "complianceweek.com",
    "risk.net",
    "regulationasia.com",
    "finews.com",
    "moneycontrol.com",
    "livemint.com",
})

# Tier D — Government & IGO sources (already caught by _TLD_TRUSTED but explicit here too)
# Credibility override: 1.0
GOVERNMENT_IGO: frozenset[str] = frozenset({
    "mas.gov.sg",
    "acra.gov.sg",
    "cpib.gov.sg",
    "interpol.int",
    "unodc.org",
    "worldbank.org",
    "imf.org",
    "un.org",
    "ofac.treas.gov",
    "treasury.gov",
    "ec.europa.eu",
    "europol.europa.eu",
})

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

# Flat map: domain → (tier_label, credibility_override)
_WHITELIST_MAP: dict[str, tuple[str, float]] = {}

for _domain in WIRE_SERVICES:
    _WHITELIST_MAP[_domain] = ("wire", 0.95)
for _domain in REGIONAL_PRESS:
    _WHITELIST_MAP[_domain] = ("regional", 0.82)
for _domain in FINANCIAL_LEGAL:
    _WHITELIST_MAP[_domain] = ("financial_legal", 0.88)
for _domain in GOVERNMENT_IGO:
    _WHITELIST_MAP[_domain] = ("government_igo", 1.00)

# Full combined set for fast membership test
ALL_WHITELISTED: frozenset[str] = frozenset(_WHITELIST_MAP.keys())


def get_whitelist_entry(registered_domain: str) -> tuple[str, float] | None:
    """
    Return (tier_label, credibility_score) if domain is whitelisted, else None.

    Checks:
      1. Exact match:    "reuters.com" → hit
      2. Subdomain:      "uk.reuters.com" → resolved to "reuters.com" upstream,
                         but as a safety net we also check suffix match here.

    Usage in compute_source_credibility():
        entry = get_whitelist_entry(effective_domain)
        if entry:
            tier_label, override_score = entry
            # skip normal scoring, use override directly
    """
    if not registered_domain:
        return None

    # Exact match first (fastest)
    if registered_domain in _WHITELIST_MAP:
        return _WHITELIST_MAP[registered_domain]

    # Subdomain safety net: "blogs.wsj.com" → check "wsj.com"
    parts = registered_domain.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in _WHITELIST_MAP:
            return _WHITELIST_MAP[parent]

    return None


def is_whitelisted(registered_domain: str) -> bool:
    """Quick boolean check."""
    return get_whitelist_entry(registered_domain) is not None


def whitelist_summary() -> dict:
    """Return counts per tier — useful for /media/config endpoint."""
    return {
        "wire_services":   len(WIRE_SERVICES),
        "regional_press":  len(REGIONAL_PRESS),
        "financial_legal": len(FINANCIAL_LEGAL),
        "government_igo":  len(GOVERNMENT_IGO),
        "total":           len(_WHITELIST_MAP),
    }
