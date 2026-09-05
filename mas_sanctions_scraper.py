"""
MAS Targeted Financial Sanctions Scraper
=========================================
Scrapes all sanction lists from:
  https://www.mas.gov.sg/regulation/anti-money-laundering/targeted-financial-sanctions/lists-of-designated-individuals-and-entities

For each list entry in the "List" column, it:
  1. Follows the link to the UN Sanctions List Materials page
  2. Downloads the XML file for that list
  3. Parses each individual/entity from the XML
  4. Outputs a single consolidated CSV + SQLite DB

Why XML over PDF/HTML?
  - PDF: requires layout-sensitive text parsing, brittle
  - HTML (the legacy file): single flat file, OK but less structured
  - XML: machine-readable, consistent schema, unambiguous field tags
    → best for downstream AML pipeline ingestion

Output columns (individuals & entities combined):
  source_list, ref_number, name, name_type (primary/alias),
  title, designation, dob, place_of_birth, nationality,
  passport_no, national_id, address, listed_on, last_updated,
  other_info, gender, entity_type (individual/entity)

Run:
    pip install requests beautifulsoup4 lxml
    python mas_sanctions_scraper.py

Or with a proxy / custom headers if MAS blocks scraping:
    python mas_sanctions_scraper.py --delay 2
"""

import argparse
import csv
import io
import logging
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAS_BASE = "https://www.mas.gov.sg"
MAS_SANCTIONS_URL = (
    "https://www.mas.gov.sg/regulation/anti-money-laundering/"
    "targeted-financial-sanctions/lists-of-designated-individuals-and-entities"
)
UN_BASE = "https://main.un.org"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

OUTPUT_CSV = Path("mas_sanctions.csv")
OUTPUT_DB = Path("mas_sanctions.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class SanctionRecord:
    source_list: str = ""           # e.g. "UN 1737 List"
    ref_number: str = ""            # e.g. "IRI.001"
    entity_type: str = ""           # "individual" or "entity"
    name: str = ""                  # primary name
    name_aliases: str = ""          # pipe-separated aliases
    title: str = ""
    designation: str = ""
    dob: str = ""
    place_of_birth: str = ""
    nationality: str = ""
    passport_no: str = ""
    national_id: str = ""
    address: str = ""
    listed_on: str = ""
    last_updated: str = ""
    other_info: str = ""
    gender: str = ""
    good_quality_aka: str = ""      # high-confidence aliases
    low_quality_aka: str = ""       # lower-confidence aliases


def record_to_dict(r: SanctionRecord) -> dict:
    return {f.name: getattr(r, f.name) for f in fields(r)}


# ---------------------------------------------------------------------------
# Step 1: Scrape MAS page → list of (label, url) for each sanction list
# ---------------------------------------------------------------------------
def get_mas_list_links(session: requests.Session) -> list[tuple[str, str, str]]:
    """
    Returns list of (list_label, url, last_updated) from the MAS sanctions page.
    The table has columns: Category | Regulations/Legislation | List | Last Updated.
    last_updated is scraped from the "Last Updated" column (e.g. "17 Sep 2024").
    """
    log.info("Fetching MAS sanctions index...")
    resp = session.get(MAS_SANCTIONS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # Find the table with the sanction lists
    # The table has columns: Category | Regulations/Legislation | List | Last Updated
    links: list[tuple[str, str, str]] = []
    seen_urls: set[str] = set()

    for table in soup.find_all("table"):
        headers_row = table.find("tr")
        if not headers_row:
            continue
        header_texts = [th.get_text(strip=True) for th in headers_row.find_all(["th", "td"])]

        # Check if this is the right table (has a "List" column)
        if "List" not in header_texts:
            continue

        list_col_idx = header_texts.index("List")

        # Locate "Last Updated" column — header may read "Last Updated" or "LastUpdated"
        last_updated_col_idx: Optional[int] = None
        for i, h in enumerate(header_texts):
            if "last" in h.lower() and "updated" in h.lower():
                last_updated_col_idx = i
                break

        # Track the most-recently-seen Last Updated value so rows that span
        # multiple <tr> entries (merged Category/Legislation cells) still get it.
        current_last_updated = ""

        for row in table.find_all("tr")[1:]:  # skip header row
            cells = row.find_all("td")
            if len(cells) <= list_col_idx:
                continue

            # Read Last Updated for this row (may be blank for continuation rows)
            if last_updated_col_idx is not None and len(cells) > last_updated_col_idx:
                raw = cells[last_updated_col_idx].get_text(strip=True)
                if raw:
                    current_last_updated = raw

            list_cell = cells[list_col_idx]
            for a in list_cell.find_all("a", href=True):
                href = a["href"].strip()
                label = a.get_text(strip=True)
                if not label:
                    continue

                # Make absolute URL
                if href.startswith("http"):
                    full_url = href
                else:
                    full_url = urljoin(MAS_BASE, href)

                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    links.append((label, full_url, current_last_updated))
                    log.info(
                        "  Found list: %s → %s  [last updated: %s]",
                        label, full_url, current_last_updated or "—",
                    )

    log.info("Total lists found: %d", len(links))
    return links


# ---------------------------------------------------------------------------
# Step 2: From each list page, find the XML download URL
# ---------------------------------------------------------------------------
def get_xml_url_from_list_page(
    session: requests.Session, list_label: str, list_url: str
) -> Optional[str]:
    """
    Fetches the sanctions list materials page and returns the XML download URL.
    The page has Pdf | Xml | Html links; we prefer Xml.
    Falls back to Html if Xml is unavailable.
    """
    log.info("  Fetching list page: %s", list_url)

    # Some URLs are direct downloads (PDF/XML), not materials pages
    parsed = urlparse(list_url)
    path_lower = parsed.path.lower()
    if path_lower.endswith(".xml"):
        return list_url
    if path_lower.endswith(".pdf"):
        log.warning("  List %s is a direct PDF link — skipping (use XML)", list_label)
        return None

    try:
        resp = session.get(list_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error("  Failed to fetch %s: %s", list_url, e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Look for links with text "Xml" or "XML" or href ending in .xml
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        href = a["href"]
        if text == "xml" or href.lower().endswith(".xml"):
            xml_url = href if href.startswith("http") else urljoin(list_url, href)
            log.info("  → XML found: %s", xml_url)
            return xml_url

    # Fallback: look for HTML legacy format
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        href = a["href"]
        if text == "html" or href.lower().endswith(".html"):
            html_url = href if href.startswith("http") else urljoin(list_url, href)
            log.info("  → No XML, falling back to HTML: %s", html_url)
            return html_url  # handled differently in parse step

    log.warning("  No XML or HTML found for %s", list_label)
    return None


# ---------------------------------------------------------------------------
# Step 3a: Parse UN XML sanctions list (standard UNSC format)
# ---------------------------------------------------------------------------
def parse_un_xml(xml_content: bytes, source_list: str, mas_last_updated: str = "") -> list[SanctionRecord]:
    """
    Parses UN Security Council XML sanctions list.
    The XML schema uses INDIVIDUALS and ENTITIES sections.
    Each INDIVIDUAL/ENTITY has sub-elements for name, DOB, aliases, etc.
    """
    records: list[SanctionRecord] = []

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        log.error("  XML parse error: %s", e)
        return records

    # Strip namespaces for easier querying
    def strip_ns(tag: str) -> str:
        return re.sub(r"\{[^}]+\}", "", tag)

    def find_all_stripped(elem, tag: str):
        return [c for c in elem.iter() if strip_ns(c.tag).upper() == tag.upper()]

    def text(elem, tag: str, default: str = "") -> str:
        results = find_all_stripped(elem, tag)
        if not results:
            return default
        return " | ".join(
            r.text.strip() for r in results if r.text and r.text.strip()
        ) or default

    def collect_names(elem) -> tuple[str, str, str, str]:
        """Returns (primary_name, aliases_pipe, good_aka, low_aka)"""
        primary_parts = []
        for part in ["FIRST_NAME", "SECOND_NAME", "THIRD_NAME", "FOURTH_NAME", "UN_LIST_TYPE"]:
            val = text(elem, part)
            if val and val.upper() not in ("NA", "N/A", ""):
                pass  # handled below

        # Build full name from name parts
        name_parts = []
        for tag in ["FIRST_NAME", "SECOND_NAME", "THIRD_NAME", "FOURTH_NAME"]:
            v = text(elem, tag)
            if v and v.upper() not in ("NA", "N/A"):
                name_parts.append(v)

        # Also try ENTITY_NAME for entities
        entity_name = text(elem, "ENTITY_NAME")
        if entity_name and entity_name.upper() not in ("NA", "N/A"):
            name_parts = [entity_name]

        primary = " ".join(name_parts).strip()

        # Aliases: INDIVIDUAL_ALIAS or ENTITY_ALIAS blocks
        good_akas, low_akas, all_aliases = [], [], []
        for alias_block in find_all_stripped(elem, "INDIVIDUAL_ALIAS") + find_all_stripped(
            elem, "ENTITY_ALIAS"
        ):
            quality = text(alias_block, "QUALITY").lower()
            alias_name_parts = []
            for tag in ["ALIAS_NAME", "FIRST_NAME", "SECOND_NAME", "THIRD_NAME"]:
                v = text(alias_block, tag)
                if v and v.upper() not in ("NA", "N/A"):
                    alias_name_parts.append(v)
            alias_str = " ".join(alias_name_parts).strip()
            if not alias_str:
                continue
            all_aliases.append(alias_str)
            if "good" in quality:
                good_akas.append(alias_str)
            else:
                low_akas.append(alias_str)

        return (
            primary,
            " | ".join(all_aliases),
            " | ".join(good_akas),
            " | ".join(low_akas),
        )

    def collect_address(elem) -> str:
        parts = []
        for addr_block in find_all_stripped(elem, "INDIVIDUAL_ADDRESS") + find_all_stripped(
            elem, "ENTITY_ADDRESS"
        ):
            addr_parts = []
            for tag in ["STREET", "CITY", "STATE_PROVINCE", "ZIP_CODE", "COUNTRY"]:
                v = text(addr_block, tag)
                if v and v.upper() not in ("NA", "N/A"):
                    addr_parts.append(v)
            if addr_parts:
                parts.append(", ".join(addr_parts))
        return " | ".join(parts)

    def collect_dob(elem) -> str:
        parts = []
        for dob_block in find_all_stripped(elem, "INDIVIDUAL_DATE_OF_BIRTH"):
            for tag in ["DATE", "YEAR", "FROM_YEAR", "TO_YEAR", "TYPE_OF_DATE"]:
                v = text(dob_block, tag)
                if v and v.upper() not in ("NA", "N/A"):
                    parts.append(v)
        return " | ".join(parts)

    def collect_pob(elem) -> str:
        parts = []
        for pob_block in find_all_stripped(elem, "INDIVIDUAL_PLACE_OF_BIRTH"):
            pob_parts = []
            for tag in ["CITY", "STATE_PROVINCE", "COUNTRY"]:
                v = text(pob_block, tag)
                if v and v.upper() not in ("NA", "N/A"):
                    pob_parts.append(v)
            if pob_parts:
                parts.append(", ".join(pob_parts))
        return " | ".join(parts)

    def collect_documents(elem) -> tuple[str, str]:
        """Returns (passport_nos, national_ids)"""
        passports, nat_ids = [], []
        for doc_block in find_all_stripped(elem, "INDIVIDUAL_DOCUMENT"):
            doc_type = text(doc_block, "TYPE_OF_DOCUMENT").lower()
            doc_num = text(doc_block, "NUMBER")
            if not doc_num or doc_num.upper() in ("NA", "N/A"):
                continue
            if "passport" in doc_type:
                passports.append(doc_num)
            elif "national" in doc_type or "identity" in doc_type:
                nat_ids.append(doc_num)
            else:
                nat_ids.append(f"[{doc_type}] {doc_num}")
        return " | ".join(passports), " | ".join(nat_ids)

    # --- Parse INDIVIDUALS ---
    for ind in find_all_stripped(root, "INDIVIDUAL"):
        r = SanctionRecord(source_list=source_list, entity_type="individual")
        r.ref_number = text(ind, "DATAID") or text(ind, "REFERENCE_NUMBER")
        r.name, r.name_aliases, r.good_quality_aka, r.low_quality_aka = collect_names(ind)
        r.title = text(ind, "TITLE")
        r.designation = text(ind, "DESIGNATION")
        r.dob = collect_dob(ind)
        r.place_of_birth = collect_pob(ind)
        r.nationality = text(ind, "NATIONALITY")
        r.passport_no, r.national_id = collect_documents(ind)
        r.address = collect_address(ind)
        r.listed_on = text(ind, "LISTED_ON")
        r.last_updated = text(ind, "LAST_DAY_UPDATED") or mas_last_updated
        r.other_info = text(ind, "COMMENTS1")
        r.gender = text(ind, "GENDER")
        records.append(r)

    # --- Parse ENTITIES ---
    for ent in find_all_stripped(root, "ENTITY"):
        r = SanctionRecord(source_list=source_list, entity_type="entity")
        r.ref_number = text(ent, "DATAID") or text(ent, "REFERENCE_NUMBER")
        r.name, r.name_aliases, r.good_quality_aka, r.low_quality_aka = collect_names(ent)
        r.designation = text(ent, "DESIGNATION")
        r.address = collect_address(ent)
        r.listed_on = text(ent, "LISTED_ON")
        r.last_updated = text(ent, "LAST_DAY_UPDATED") or mas_last_updated
        r.other_info = text(ent, "COMMENTS1")
        records.append(r)

    log.info(
        "  Parsed %d individuals + %d entities from %s",
        sum(1 for r in records if r.entity_type == "individual"),
        sum(1 for r in records if r.entity_type == "entity"),
        source_list,
    )
    return records


# ---------------------------------------------------------------------------
# Step 3b: Parse legacy HTML sanctions list (Iran flat HTML format)
# ---------------------------------------------------------------------------
def parse_legacy_html(html_content: bytes, source_list: str, mas_last_updated: str = "") -> list[SanctionRecord]:
    """
    Parses the legacy flat HTML format (e.g. iran_all_name_legacy.html).
    Each entry is a <tr> with structured fields inside.
    This is a best-effort parser for the flat text format shown in Image 3.
    """
    records: list[SanctionRecord] = []
    soup = BeautifulSoup(html_content, "lxml")

    # The HTML has rows in a table with class "sanctions" or similar
    # Each individual entry has an id like "IRI.001"
    # The popup/tooltip shows: Name, Designation, DOB, Good AKA, Low AKA,
    #   Nationality, Passport, National ID, Address, Listed on

    # Strategy: find all rows with an ID that looks like a reference number
    ref_pattern = re.compile(r"^[A-Z]{2,5}\.\d{3,}", re.IGNORECASE)

    # Try table rows with structured content
    rows = soup.find_all("tr")
    for row in rows:
        row_id = row.get("id", "")
        if not ref_pattern.match(row_id):
            continue

        r = SanctionRecord(source_list=source_list)
        r.ref_number = row_id

        # Extract text content
        cells = row.find_all("td")
        full_text = " ".join(c.get_text(" ", strip=True) for c in cells)

        # Detect entity type from ref number prefix
        r.entity_type = "entity" if row_id.startswith(("GRPe", "ENT")) else "individual"

        # Parse fields using label patterns found in the HTML
        def extract_field(label: str, text_block: str) -> str:
            pattern = re.compile(
                rf"{re.escape(label)}\s*[:\-]?\s*(.+?)(?=\s+(?:DOB|Title|Designation|"
                r"Good quality|Low quality|Nationality|Passport|National|Address|Listed|$))",
                re.IGNORECASE | re.DOTALL,
            )
            m = pattern.search(text_block)
            if m:
                val = m.group(1).strip().rstrip(".")
                return val if val.upper() not in ("NA", "N/A", "na") else ""
            return ""

        # Name is usually in a <strong> or <b> tag, or first cell
        name_tag = row.find(["strong", "b"])
        if name_tag:
            r.name = name_tag.get_text(strip=True)
        else:
            r.name = cells[0].get_text(strip=True) if cells else ""

        r.designation = extract_field("Designation", full_text)
        r.dob = extract_field("DOB", full_text)
        r.good_quality_aka = extract_field("Good quality a.k.a.", full_text)
        r.low_quality_aka = extract_field("Low quality a.k.a.", full_text)
        r.name_aliases = " | ".join(
            filter(None, [r.good_quality_aka, r.low_quality_aka])
        )
        r.nationality = extract_field("Nationality", full_text)
        r.passport_no = extract_field("Passport no", full_text)
        r.national_id = extract_field("National identification no", full_text)
        r.address = extract_field("Address", full_text)
        r.listed_on = extract_field("Listed on", full_text)
        r.last_updated = extract_field("Last updated", full_text) or mas_last_updated
        r.other_info = extract_field("Other information", full_text)

        records.append(r)

    log.info("  Parsed %d records from HTML (legacy format) for %s", len(records), source_list)
    return records


# ---------------------------------------------------------------------------
# Step 4: Download content and dispatch to correct parser
# ---------------------------------------------------------------------------
def download_and_parse(
    session: requests.Session, source_list: str, url: str, mas_last_updated: str = ""
) -> list[SanctionRecord]:
    log.info("  Downloading: %s", url)
    try:
        resp = session.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error("  Download failed: %s", e)
        return []

    content_type = resp.headers.get("Content-Type", "").lower()
    url_lower = url.lower()

    if "xml" in content_type or url_lower.endswith(".xml"):
        return parse_un_xml(resp.content, source_list, mas_last_updated)
    elif "html" in content_type or url_lower.endswith(".html"):
        return parse_legacy_html(resp.content, source_list, mas_last_updated)
    else:
        # Try XML first (many servers return wrong content-type)
        try:
            records = parse_un_xml(resp.content, source_list, mas_last_updated)
            if records:
                return records
        except Exception:
            pass
        # Then try HTML
        return parse_legacy_html(resp.content, source_list, mas_last_updated)


# ---------------------------------------------------------------------------
# Step 5: Output — CSV + SQLite
# ---------------------------------------------------------------------------
COLUMNS = [f.name for f in fields(SanctionRecord)]


def save_csv(records: list[SanctionRecord], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow(record_to_dict(r))
    log.info("Saved %d records to %s", len(records), path)


def save_sqlite(records: list[SanctionRecord], path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    col_defs = ", ".join(f'"{c}" TEXT' for c in COLUMNS)
    cur.execute(f"CREATE TABLE IF NOT EXISTS sanctions ({col_defs})")

    # Useful indexes for AML name matching
    cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON sanctions (name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON sanctions (source_list)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ref ON sanctions (ref_number)")

    placeholders = ", ".join("?" for _ in COLUMNS)
    for r in records:
        row_dict = record_to_dict(r)
        cur.execute(
            f"INSERT INTO sanctions VALUES ({placeholders})",
            [row_dict[c] for c in COLUMNS],
        )

    conn.commit()
    conn.close()
    log.info("Saved %d records to SQLite: %s", len(records), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MAS Sanctions List Scraper")
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between requests (default: 1.0)"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=OUTPUT_CSV,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--output-db", type=Path, default=OUTPUT_DB,
        help="Output SQLite DB file path"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only list found URLs, don't download/parse"
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    all_records: list[SanctionRecord] = []

    # Step 1: Get all list links from MAS
    list_links = get_mas_list_links(session)
    if not list_links:
        log.error("No list links found. Check if MAS page structure changed.")
        return

    for label, url, mas_last_updated in list_links:
        time.sleep(args.delay)

        if args.dry_run:
            log.info("[DRY RUN] Would process: %s → %s  [last updated: %s]", label, url, mas_last_updated or "—")
            continue

        # Step 2: Get XML URL from each list page
        xml_url = get_xml_url_from_list_page(session, label, url)
        if not xml_url:
            log.warning("Skipping %s (no downloadable list found)", label)
            continue

        time.sleep(args.delay)

        # Steps 3+4: Download and parse
        records = download_and_parse(session, label, xml_url, mas_last_updated)
        all_records.extend(records)

    if args.dry_run:
        log.info("[DRY RUN] Done. Found %d lists total.", len(list_links))
        return

    log.info("\n=== Total records scraped: %d ===", len(all_records))

    # Step 5: Save outputs
    if all_records:
        save_csv(all_records, args.output_csv)
        save_sqlite(all_records, args.output_db)
        log.info("\nOutput files:")
        log.info("  CSV  → %s  (use for Excel/pandas AML pipeline)", args.output_csv)
        log.info("  DB   → %s  (use for fast SQL name lookups)", args.output_db)
    else:
        log.warning("No records scraped. Check logs for errors.")


if __name__ == "__main__":
    main()