# AML Shield

A KYC/AML (Know Your Customer / Anti-Money Laundering) compliance platform combining ML-based transaction monitoring, sanctions screening, and KYC risk checks (PEP + adverse media) — served through a FastAPI backend and a dark-terminal-themed dashboard.

Built as a self-directed project exploring how real-world compliance systems combine rule-based screening with ML risk models.

## What it does

AML Shield surfaces AML/compliance risk from three angles:

1. **Transaction Monitoring** — Random Forest, Decision Tree, and Logistic Regression models (champion selected via `GridSearchCV`) trained on a labeled transaction dataset, scoring each transaction 0–100 using features like amount, hour of day, cross-border flag, currency mismatch, payment type, off-hours flag, and FATF-aligned geo risk. Risk tiers: **CRITICAL / HIGH** (>65), **MEDIUM** (35–65), **LOW** (<35).
2. **Sanctions Screening** — scrapes Monetary Authority of Singapore (MAS) targeted financial sanctions lists (via the underlying UN XML data, not PDF/HTML) into a SQLite database, refreshed automatically every 24 hours. Name matching combines token Jaccard similarity (0.6 weight) and character-trigram cosine similarity (0.4 weight), with a configurable match threshold (default 0.70).
3. **KYC Screening** — PEP (Politically Exposed Person) checks via OpenSanctions, plus adverse media search across DuckDuckGo, Google, GDELT, and optionally Google CSE / NewsAPI (if API keys are supplied). Each article is scored by source credibility (whitelist tier, domain trust) × keyword risk weight, with social-media sources hard-blocked.

All of this is served through one FastAPI app with a dashboard frontend, and role-based login (analyst / compliance / admin).

## Architecture

```
┌──────────────────────┐
│  frontend_v0_7.html   │  ← Dashboard UI (dark-terminal theme), served at "/"
└───────────┬────────────┘
            │ REST API
┌───────────▼────────────┐
│       main_V5.py         │  ← FastAPI app: auth, transactions, ML predict/train,
│                           │     sanctions status/check/refresh, startup orchestration
└───────────┬───────┬───────┘
            │       │ imports as router
            │  ┌────▼─────────────────┐      ┌──────────────────────────┐
            │  │  kyc_screening_v2.py   │─────▶│   media_whitelist.py      │
            │  │  PEP + adverse media   │      │   Trusted domain tiers    │
            │  │  + credibility scoring │      │   used for source scoring │
            │  └─────────────────────────┘      └──────────────────────────┘
            │
┌───────────▼──────────────┐
│  mas_sanctions_scraper.py │  ← Scrapes MAS sanctions lists (XML) → CSV + SQLite
└────────────────────────────┘
```

## Modules

### `main_V5.py` — Core backend & orchestration
- FastAPI app serving the dashboard (`/`) and all API routes
- On startup: trains ML models if `models.pkl` doesn't exist, builds the sanctions DB if missing, starts a 24-hour sanctions refresh loop, and streams the transaction CSV into memory in chunks
- CLI flags: `--retrain`, `--csv <file>`, `--port <n>`, `--db <path>`
- Optional env vars: `GOOGLE_CSE_KEY` / `GOOGLE_CSE_CX` (Google Custom Search, best adverse-media coverage), `NEWSAPI_KEY` (fallback) — the app works without these, just with narrower media coverage
- Role-based login (`/auth/login`) — analyst / compliance / admin passcodes

### `kyc_screening_v2.py` — KYC / PEP / adverse-media module
- Mounted into `main_V5.py` as a router under `/kyc`
- PEP screening via OpenSanctions (no API key needed) — badges mapped to risk categories, hit threshold: name similarity ≥ 0.30 + at least one risk topic badge
- EDD (Enhanced Due Diligence) levels: HIGH (terrorism/sanctions hit), MEDIUM (PEP or close associate), STANDARD
- Adverse media pipeline: fetches from Google CSE → DuckDuckGo → Google scrape → GDELT → NewsAPI, merges and dedupes results
- Each article scored via `keyword_score × credibility_multiplier`, using `media_whitelist.py` for source credibility
- Runtime-configurable keyword weights via `PUT /kyc/media/config` (no restart needed)

### `media_whitelist.py` — Trusted source whitelist
- Four credibility tiers: `WIRE_SERVICES` (0.95 — Reuters, Bloomberg, BBC, FT...), `FINANCIAL_LEGAL` (0.88 — OCCRP, ICIJ, FATF, Law360...), `REGIONAL_PRESS` (0.82 — Straits Times, CNA, SCMP...), `GOVERNMENT_IGO` (1.00 — MAS, Interpol, OFAC, UN...)
- Whitelisted-source articles always surface even at low keyword score; new domains can be added directly to the relevant tier set

### `mas_sanctions_scraper.py` — Sanctions data pipeline
- Scrapes the MAS Targeted Financial Sanctions list page, follows each entry to its underlying UN Sanctions List XML (chosen over PDF/HTML for reliable structured parsing)
- Parses individuals and entities into a consolidated schema (name, aliases, DOB, nationality, passport/national ID, listing/last-updated dates, etc.)
- Outputs both a CSV and a SQLite database for downstream ingestion by `main_V5.py`

### `frontend_v0_7.html` — Dashboard
- Single-file dashboard UI in a dark-terminal visual theme, served automatically at `http://localhost:8000/`
- Talks to the FastAPI backend for live transaction, sanctions, and KYC data

## Running it locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/aml-shield.git
cd aml-shield

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Supply a transaction dataset named SAML-D.csv in the project root
#    (not included in this repo — bring your own labeled transaction data)

# 5. Run the backend
python3 main_V5.py
```

Then open **http://localhost:8000** — the dashboard is served automatically.

### Useful flags & env vars

```bash
python3 main_V5.py --retrain            # force retrain even if models.pkl exists
python3 main_V5.py --csv your_file.csv  # use a different transaction dataset
python3 main_V5.py --port 8080          # use a different port

# Optional, for broader adverse-media coverage:
export GOOGLE_CSE_KEY="AIza..."
export GOOGLE_CSE_CX="12345:abc..."
export NEWSAPI_KEY="your_key_here"
```

To run the sanctions scraper standalone:

```bash
python3 mas_sanctions_scraper.py
```

### Login roles

| Role | Passcode | Access |
|---|---|---|
| analyst | `1234` | Transactions, predictions, KYC |
| compliance | `5678` | Sanctions, transactions, KYC |
| admin | `9999` | Everything |

*(Demo passcodes — replace before any real deployment.)*

## Key API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Server status, model metrics, sanctions count |
| `/predict` | POST | Score a single transaction |
| `/transactions`, `/transactions/summary`, `/transactions/alerts` | GET | Transaction data & risk breakdown |
| `/sanctions/check` | POST | Screen a name against MAS sanctions |
| `/sanctions/entries`, `/sanctions/refresh` | GET / POST | Browse / force-refresh sanctions data |
| `/kyc/screen` | POST | Full KYC: PEP + adverse media combined |
| `/kyc/pep`, `/kyc/media` | POST | PEP-only / media-only checks |
| `/kyc/media/config` | GET / PUT | View / update keyword weights at runtime |
| `/kyc/media/whitelist` | GET | View all whitelisted domains |
| `/kyc/media/debug?name=X` | GET | Debug adverse-media fetch counts per source |
| `/auth/login` | POST | Get a role token |

## Notes / current limitations

- `models.pkl`, `mas_sanctions.db`, and `SAML-D.csv` are generated or user-supplied at runtime and are **not** committed to this repo (see `.gitignore`) — clone the repo, supply your own dataset, and the first run will build everything else automatically.
- This is a personal/academic exploration of AML tooling and is **not** intended for production compliance use without further validation, legal review, and security hardening (the demo login passcodes in particular should never ship as-is).

## Tech stack

- **Backend:** Python, FastAPI, scikit-learn, pandas, NumPy
- **Scraping:** httpx, BeautifulSoup, lxml, tldextract
- **Frontend:** HTML/CSS/JS (single-file dashboard)
- **Data sources:** MAS/UN sanctions lists, OpenSanctions API, DuckDuckGo/Google/GDELT news search
