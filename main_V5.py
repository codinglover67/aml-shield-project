"""
AML Shield — Combined Main (main.py)
=======================================
Single entrypoint. Runs the AML ML backend + MAS Sanctions screening API
together on one FastAPI server. No separate sanctions_api.py needed.

Run:
    pip install fastapi uvicorn sqlite-utils python-multipart scikit-learn pandas numpy
    pip install httpx beautifulsoup4 lxml
    python main.py --csv SAML-D.csv

Then open: http://localhost:8000/
"""

import argparse
import asyncio
import json
import logging
import math
import os
import pickle
import random
import re
import sqlite3
import threading
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("aml")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path("mas_sanctions.db")
REFRESH_INTERVAL_HOURS = 24

# Singapore Time = UTC+8
SGT = timezone(timedelta(hours=8))
_last_scrape_sgt: str = ""   # ISO string of last completed scrape in SGT

def _now_sgt_str() -> str:
    """Return current Singapore time as a human-readable string."""
    return datetime.now(SGT).strftime("%d %b %Y, %I:%M %p SGT")

# ---------------------------------------------------------------------------
# RBAC — role → allowed route prefixes
# ---------------------------------------------------------------------------
ROLE_PASSCODES = {
    "analyst":   "1234",
    "compliance": "5678",
    "admin":     "9999",
}

ROLE_ROUTES: dict[str, list[str]] = {
    "analyst": [
        "/health", "/predict", "/transactions", "/transactions/summary",
        "/transactions/alerts", "/transactions/progress", "/kyc",
    ],
    "compliance": [
        "/health", "/sanctions", "/sanctions/check", "/sanctions/entries", "/sanctions/status",
        "/transactions", "/transactions/summary", "/transactions/alerts",
        "/transactions/progress", "/kyc",
    ],
    "admin": ["*"],   # wildcard = everything
}

# ---------------------------------------------------------------------------
# ── SCRAPER INTEGRATION ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def run_scraper(db_path: Path = None, csv_path: Path = None) -> bool:
    """
    Run mas_sanctions_scraper programmatically.
    Returns True on success, False on failure.
    Deletes the existing DB before writing so stale data is never kept.
    """
    try:
        import mas_sanctions_scraper as scraper
        import requests as _req

        target_db  = db_path  or DB_PATH
        target_csv = csv_path or Path("mas_sanctions.csv")

        log.info("Sanctions scraper starting — writing to %s", target_db)

        # Remove stale DB so a failed scrape doesn't silently leave old data
        if target_db.exists():
            target_db.unlink()
            log.info("Removed stale %s before refresh", target_db)

        session = _req.Session()
        session.headers.update(scraper.HEADERS)

        list_links = scraper.get_mas_list_links(session)
        if not list_links:
            log.error("Scraper: no list links found — MAS page structure may have changed")
            return False

        all_records: list = []
        for label, url, mas_last_updated in list_links:
            time.sleep(1)
            xml_url = scraper.get_xml_url_from_list_page(session, label, url)
            if not xml_url:
                log.warning("Scraper: skipping %s (no downloadable list found)", label)
                continue
            time.sleep(1)
            records = scraper.download_and_parse(session, label, xml_url, mas_last_updated)
            all_records.extend(records)

        if not all_records:
            log.warning("Scraper: finished but found 0 records — DB not written")
            return False

        scraper.save_csv(all_records, target_csv)
        scraper.save_sqlite(all_records, target_db)
        log.info("Sanctions scraper complete — %d records saved to %s", len(all_records), target_db)

        global _last_scrape_sgt
        _last_scrape_sgt = _now_sgt_str()
        log.info("Last scrape time (SGT): %s", _last_scrape_sgt)
        return True

    except Exception:
        log.exception("Sanctions scraper raised an unexpected error")
        return False

def _scraper_needed() -> bool:
    """Return True if the DB is missing or older than REFRESH_INTERVAL_HOURS."""
    if not DB_PATH.exists():
        return True
    age_hours = (time.time() - DB_PATH.stat().st_mtime) / 3600
    return age_hours >= REFRESH_INTERVAL_HOURS

def _scraper_background_loop():
    """
    Background daemon thread:
      • Runs the scraper immediately if the DB is missing or stale.
      • Then sleeps and re-checks every hour, scraping again whenever the DB
        is older than REFRESH_INTERVAL_HOURS (default 24 h).
    """
    # Small startup delay so the server can finish initialising first
    time.sleep(5)
    while True:
        if _scraper_needed():
            log.info(
                "Sanctions DB %s — launching scraper …",
                "missing" if not DB_PATH.exists() else "stale (>%dh)" % REFRESH_INTERVAL_HOURS,
            )
            ok = run_scraper()
            if ok:
                # Reload the in-memory cache with the fresh data
                load_sanctions_cache()
        # Re-check in 1 hour; the next scrape fires once the DB is 24h old
        time.sleep(3600)

DEFAULT_THRESHOLD = 0.70
CSV_PATH = "SAML-D.csv"

# ---------------------------------------------------------------------------
# ── SANCTIONS ENGINE ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
_sanctions_cache: list[dict] = []
_cache_loaded_at: float = 0.0

def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    noise = {"pte", "ltd", "llc", "inc", "co", "sdn", "bhd", "corp",
             "limited", "company", "enterprises", "group", "holdings",
             "international", "trading", "investment", "investments"}
    tokens = [t for t in s.split() if t not in noise]
    return " ".join(tokens)

def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def trigrams(s: str) -> set:
    padded = f"  {s}  "
    return {padded[i:i+3] for i in range(len(padded) - 2)}

def trigram_cosine(a: str, b: str) -> float:
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))

def score_pair(query_norm: str, candidate_norm: str) -> float:
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm == candidate_norm:
        return 1.0
    j = token_jaccard(query_norm, candidate_norm)
    t = trigram_cosine(query_norm, candidate_norm)
    return round(0.6 * j + 0.4 * t, 4)

def load_sanctions_cache():
    global _sanctions_cache, _cache_loaded_at
    if not DB_PATH.exists():
        log.warning(f"{DB_PATH} not found — sanctions screening disabled")
        _sanctions_cache = []
        _cache_loaded_at = time.time()
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT ref_number, entity_type, name, name_aliases,
               good_quality_aka, designation, nationality,
               listed_on, last_updated, source_list,
               dob, address, passport_no, other_info
        FROM sanctions
        WHERE name IS NOT NULL AND name != ''
    """)
    rows = cur.fetchall()
    conn.close()

    _sanctions_cache = []
    for r in rows:
        aliases = [a.strip() for a in (r["name_aliases"] or "").split("|") if a.strip()]
        gq_aka  = [a.strip() for a in (r["good_quality_aka"] or "").split("|") if a.strip()]
        _sanctions_cache.append({
            "ref":          r["ref_number"],
            "type":         r["entity_type"],
            "name":         r["name"],
            "aliases":      aliases,
            "good_aka":     gq_aka,
            "designation":  r["designation"] or "",
            "nationality":  r["nationality"] or "",
            "listed_on":    r["listed_on"] or "",
            "last_updated": r["last_updated"] or "",
            "regime":       r["source_list"] or "MAS",
            "dob":          r["dob"] or "",
            "address":      r["address"] or "",
            "passport_no":  r["passport_no"] or "",
            "other_info":   r["other_info"] or "",
        })
    _cache_loaded_at = time.time()
    log.info(f"Loaded {len(_sanctions_cache)} sanctions entries")

def screen_name(name: str, aliases: list = None, threshold: float = DEFAULT_THRESHOLD,
                entries: list = None) -> dict:
    if entries is None:
        entries = _sanctions_cache
    query_variants = [normalise(name)]
    for a in (aliases or []):
        n = normalise(a)
        if n and n not in query_variants:
            query_variants.append(n)

    best_score = 0.0
    best_entry = None
    best_candidate = None

    for entry in entries:
        candidates = [entry["name"]] + entry["aliases"]
        for candidate in candidates:
            cand_norm = normalise(candidate)
            for qv in query_variants:
                s = score_pair(qv, cand_norm)
                if s > best_score:
                    best_score = s
                    best_entry = entry
                    best_candidate = candidate

    hit = best_score >= threshold
    return {
        "hit":           hit,
        "score":         best_score,
        "matched_name":  best_candidate if hit else None,
        "matched_entry": best_entry if hit else None,
        "threshold":     threshold,
        "screened_at":   datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# ── ML MODELS ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
MODELS:         dict = {}
LABEL_ENCODERS: dict = {}
METRICS:        dict = {}
MODEL_READY          = False

FEATURE_COLS = [
    "Log_amount", "High_value", "Hour", "Off_hours", "DayOfWeek",
    "Is_cross_border", "Currency_mismatch", "Payment_type_enc",
    "Laundering_type_enc",
]
FEATURE_LABELS = {
    "Log_amount":           "Log Amount",
    "High_value":           "High Value (>$10k)",
    "Hour":                 "Hour of Day",
    "Off_hours":            "Off-Hours",
    "DayOfWeek":            "Day of Week",
    "Is_cross_border":      "Cross-border",
    "Currency_mismatch":    "Currency Mismatch",
    "Payment_type_enc":     "Payment Type",
    "Laundering_type_enc":  "Laundering Type",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    amounts = df["Amount"].astype(float)
    out["Log_amount"] = np.log1p(amounts)
    out["High_value"] = (amounts > 10_000).astype(int)

    def _hour(v):
        try:
            s = str(v).replace(":", "").replace(".", "")
            return int(s.zfill(6)[:2])
        except:
            return 12

    out["Hour"]      = df["Time"].apply(_hour)
    out["Off_hours"] = ((out["Hour"] < 8) | (out["Hour"] > 18)).astype(int)
    out["DayOfWeek"] = pd.to_datetime(df["Date"], errors="coerce").dt.dayofweek.fillna(0).astype(int)
    out["Is_cross_border"]   = (df["Sender_bank_location"].astype(str) != df["Receiver_bank_location"].astype(str)).astype(int)
    out["Currency_mismatch"] = (df["Payment_currency"].astype(str)     != df["Received_currency"].astype(str)).astype(int)

    le = LABEL_ENCODERS.get("Payment_type")
    if le is not None:
        known = set(le.classes_)
        out["Payment_type_enc"] = df["Payment_type"].astype(str).apply(
            lambda x: int(le.transform([x])[0]) if x in known else 0
        )
    else:
        out["Payment_type_enc"] = 0

    le_lt = LABEL_ENCODERS.get("Laundering_type")
    if le_lt is not None:
        known_lt = set(le_lt.classes_)
        lt_clean = df["Laundering_type"].astype(str).str.replace(r"^Normal_", "", regex=True)
        out["Laundering_type_enc"] = lt_clean.apply(
            lambda x: int(le_lt.transform([x])[0]) if x in known_lt else 0
        )
    else:
        out["Laundering_type_enc"] = 0
    return out[FEATURE_COLS]

def train_models(csv_path: str):
    global MODEL_READY, _TX_RECORDS, _TX_TOTAL
    log.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    MAX_ROWS = 3_000_000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
    log.info(f"Loaded {len(df):,} rows")

    LABEL_ENCODERS["Payment_type"] = LabelEncoder()
    LABEL_ENCODERS["Payment_type"].fit(df["Payment_type"].astype(str))

    # Strip "Normal_" prefix from Laundering_type before encoding
    df["Laundering_type"] = df["Laundering_type"].astype(str).str.replace(r"^Normal_", "", regex=True)
    LABEL_ENCODERS["Laundering_type"] = LabelEncoder()
    LABEL_ENCODERS["Laundering_type"].fit(df["Laundering_type"])

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Is_laundering"].astype(int)
    )
    X_train = engineer_features(df_train)
    X_test = engineer_features(df_test)
    y_train = df_train["Is_laundering"].astype(int)
    y_test = df_test["Is_laundering"].astype(int)

    logit = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs", class_weight="balanced")
    logit.fit(X_train, y_train)
    logit_auc = roc_auc_score(y_test, logit.predict_proba(X_test)[:, 1])

    cart = DecisionTreeClassifier(max_depth=6, min_impurity_decrease=0.001, random_state=42, class_weight="balanced")
    cart.fit(X_train, y_train)
    cart_prob = cart.predict_proba(X_test)[:, 1]
    cart_auc  = roc_auc_score(y_test, cart_prob)

    rf = RandomForestClassifier(n_estimators=100, max_features=3, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_auc    = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    def _cm_metrics(y_true, y_pred_proba, threshold=0.35):
        y_pred = (y_pred_proba > threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                    precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4))

    cart_cm  = _cm_metrics(y_test, cart_prob)
    rf_cm    = _cm_metrics(y_test, rf.predict_proba(X_test)[:, 1])   # use full test set for consistency
    logit_cm = _cm_metrics(y_test, logit.predict_proba(X_test)[:, 1])

    feat_imp = sorted(zip(FEATURE_COLS, cart.feature_importances_.tolist()), key=lambda x: -x[1])
    pay_rates = (
        df.groupby("Payment_type")["Is_laundering"].mean()
        .reset_index().rename(columns={"Is_laundering": "rate"})
        .sort_values("rate", ascending=False).to_dict("records")
    )

    MODELS.update({"logit": logit, "cart": cart, "rf": rf})
    METRICS.update({
        "logit_auc": round(logit_auc, 4),
        "cart_auc": round(cart_auc, 4),
        "rf_auc": round(rf_auc, 4),
        "rf_sensitivity": rf_cm["recall"],
        "rf_specificity": round(rf_cm["tn"] / (rf_cm["tn"] + rf_cm["fp"]) if (rf_cm["tn"] + rf_cm["fp"]) > 0 else 0, 4),
        "cart_cm": cart_cm,
        "rf_cm": rf_cm,
        "logit_cm": logit_cm,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(df)),
        "laundering_rate": round(float(df["Is_laundering"].astype(int).mean()), 6),
        "n_positive": int(df["Is_laundering"].astype(int).sum()),
        "payment_types": list(LABEL_ENCODERS["Payment_type"].classes_),
        "feature_importance": [{"feature": f, "label": FEATURE_LABELS.get(f, f), "importance": round(i, 6)} for f, i in
                               feat_imp],
        "payment_rate": pay_rates,
        "trained_on": os.path.basename(csv_path),
    })

    with open("models.pkl", "wb") as f:
        pickle.dump({"models": MODELS, "encoders": LABEL_ENCODERS, "metrics": METRICS}, f)
    MODEL_READY = True
    _TX_RECORDS = None
    log.info("✓ Training complete")

def load_models_from_cache() -> bool:
    global MODEL_READY
    if not os.path.exists("models.pkl"):
        return False
    try:
        with open("models.pkl", "rb") as f:
            data = pickle.load(f)
        MODELS.update(data["models"])
        LABEL_ENCODERS.update(data["encoders"])
        METRICS.update(data["metrics"])
        if "feature_importance" not in METRICS and "cart" in MODELS:
            fi = sorted(zip(FEATURE_COLS, MODELS["cart"].feature_importances_.tolist()), key=lambda x: -x[1])
            METRICS["feature_importance"] = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "importance": round(i, 6)} for f, i in fi]
        MODEL_READY = True
        log.info(f"✓ Models loaded from cache")
        return True
    except Exception as e:
        log.error(f"Failed to load models.pkl: {e}")
        return False


# ---------------------------------------------------------------------------
# ── TX CACHE ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
_TX_RECORDS:  list | None = None
_TX_TOTAL:    int         = 0
_TX_LOCK      = threading.Lock()
_TX_PROGRESS  = 0.0
_TX_LOADING   = False
_TX_ERROR: str | None = None
CHUNK_SIZE    = 5_000

NEEDED_COLS = [
    "Time", "Date", "Sender_account", "Receiver_account", "Amount",
    "Payment_currency", "Received_currency", "Sender_bank_location",
    "Receiver_bank_location", "Payment_type", "Is_laundering", "Laundering_type",
]
DAY_MAP = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}


def risk_label(s):
    return "HIGH" if s > 0.65 else "MEDIUM" if s > 0.35 else "LOW"

def build_input_row(amount, hour, payment_type, day, sender, receiver, pay_curr, recv_curr, laundering_type=""):
    known  = list(LABEL_ENCODERS["Payment_type"].classes_)
    pt     = payment_type if payment_type in known else known[0]
    pt_enc = int(LABEL_ENCODERS["Payment_type"].transform([pt])[0])
    amt    = float(amount); hr = int(hour); dow = DAY_MAP.get(str(day), 0)

    lt_enc = 0
    le_lt = LABEL_ENCODERS.get("Laundering_type")
    if le_lt is not None:
        lt_clean = re.sub(r"^Normal_", "", str(laundering_type))
        known_lt = set(le_lt.classes_)
        lt_enc = int(le_lt.transform([lt_clean])[0]) if lt_clean in known_lt else 0

    row = {
        "Log_amount": math.log1p(amt), "High_value": int(amt > 10_000),
        "Hour": hr, "Off_hours": int(hr < 8 or hr > 18), "DayOfWeek": dow,
        "Is_cross_border": int(str(sender) != str(receiver)),
        "Currency_mismatch": int(str(pay_curr) != str(recv_curr)),
        "Payment_type_enc": pt_enc,
        "Laundering_type_enc": lt_enc,
    }
    return pd.DataFrame([row])[FEATURE_COLS]

def _record_from_arrays(i, amts, accts, recvs, pts, snds, rcvs_arr,
                        pcs, rcs, dates, times, lt_a,
                        is_hv, is_cross, is_curr_mm, is_cash, is_launder, scores,
                        base_idx: int) -> dict:
    s = int(scores[i])
    risk = "high" if s >= 65 else "medium" if s >= 35 else "low"
    flags = []
    if is_hv[i]:      flags.append("High Value")
    if is_cross[i]:   flags.append("Cross-border")
    if is_curr_mm[i]: flags.append("Currency Mismatch")
    if is_cash[i]:    flags.append("Cash")
    lt = lt_a[i]
    if is_launder[i] and lt and lt not in ("", "nan", "0"):
        flags.append(lt)
    return {
        "id":              f"TX-{base_idx + i:07d}",
        "row_index":       base_idx + i,
        "account":         accts[i],
        "receiver":        recvs[i],
        "amount":          f"${amts[i]:,.2f}",
        "amount_raw":      float(amts[i]),
        "type":            pts[i],
        "score":           s,
        "risk":            risk,
        "flags":           flags,
        "is_laundering":   int(is_launder[i]),
        "laundering_type": lt if lt not in ("nan", "") else "",
        "sender_country":  snds[i],
        "receiver_country": rcvs_arr[i],
        "pay_curr":        pcs[i],
        "recv_curr":       rcs[i],
        "date":            dates[i],
        "time":            times[i],
    }

def _load_csv_chunked():
    global _TX_RECORDS, _TX_TOTAL, _TX_PROGRESS, _TX_LOADING, _TX_ERROR
    with _TX_LOCK:
        if _TX_RECORDS is not None:
            _TX_LOADING = False
            return
        if _TX_LOADING:
            return
        _TX_LOADING  = True
        _TX_RECORDS  = []
        _TX_PROGRESS = 0.0
        _TX_ERROR    = None

    if not os.path.exists(CSV_PATH):
        _TX_ERROR = f"CSV not found: '{CSV_PATH}'"
        _TX_LOADING = False
        return

    try:
        total = sum(1 for _ in open(CSV_PATH, encoding="utf-8", errors="replace")) - 1
        with _TX_LOCK:
            _TX_TOTAL = total

        processed = 0
        reader = pd.read_csv(CSV_PATH, usecols=NEEDED_COLS, chunksize=CHUNK_SIZE, low_memory=False)

        for chunk_df in reader:
            chunk_df = chunk_df.reset_index(drop=True)
            n = len(chunk_df)
            feat = engineer_features(chunk_df)
            if MODEL_READY and "rf" in MODELS:
                scores = (MODELS["rf"].predict_proba(feat)[:, 1] * 100).astype(int)
            else:
                scores = (chunk_df["Is_laundering"].fillna(0).astype(int).values * 75)

            amounts    = pd.to_numeric(chunk_df["Amount"], errors="coerce").fillna(0)
            is_hv      = (amounts > 10_000).values
            is_cross   = (chunk_df["Sender_bank_location"].astype(str) != chunk_df["Receiver_bank_location"].astype(str)).values
            is_curr_mm = (chunk_df["Payment_currency"].astype(str) != chunk_df["Received_currency"].astype(str)).values
            is_cash    = chunk_df["Payment_type"].astype(str).str.startswith("Cash").values
            is_launder = chunk_df["Is_laundering"].fillna(0).astype(int).values
            lt_a  = (chunk_df["Laundering_type"].fillna("").astype(str)
                     .str.replace(r"^Normal_", "", regex=True).values)
            dates = chunk_df["Date"].fillna("").astype(str).values
            times = chunk_df["Time"].fillna("").astype(str).values
            accts = chunk_df["Sender_account"].fillna("").astype(str).values
            recvs = chunk_df["Receiver_account"].fillna("").astype(str).values
            pts   = chunk_df["Payment_type"].fillna("").astype(str).values
            snds  = chunk_df["Sender_bank_location"].fillna("").astype(str).values
            rcvs_arr = chunk_df["Receiver_bank_location"].fillna("").astype(str).values
            pcs   = chunk_df["Payment_currency"].fillna("").astype(str).values
            rcs   = chunk_df["Received_currency"].fillna("").astype(str).values
            amts  = amounts.values

            chunk_records = [
                _record_from_arrays(i, amts, accts, recvs, pts, snds, rcvs_arr,
                                    pcs, rcs, dates, times, lt_a,
                                    is_hv, is_cross, is_curr_mm, is_cash, is_launder, scores, processed)
                for i in range(n)
            ]

            with _TX_LOCK:
                _TX_RECORDS.extend(chunk_records)
                processed    += n
                _TX_PROGRESS  = min(processed / total, 1.0) if total > 0 else 1.0

        with _TX_LOCK:
            _TX_PROGRESS = 1.0
            _TX_LOADING  = False
        log.info(f"✓ TX cache: {len(_TX_RECORDS):,} records")

    except Exception as e:
        _TX_ERROR   = str(e)
        _TX_LOADING = False
        log.error(f"TX load failed: {e}", exc_info=True)

def _ensure_loading():
    global _TX_RECORDS
    if _TX_RECORDS is None and not _TX_LOADING:
        t = threading.Thread(target=_load_csv_chunked, daemon=True, name="tx-loader")
        t.start()


# ---------------------------------------------------------------------------
# ── LIFESPAN ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Sanctions: scrape now if DB is missing, then keep it fresh daily ---
    if not DB_PATH.exists():
        log.info("mas_sanctions.db not found — running scraper before startup ...")
        ok = run_scraper()
        if not ok:
            log.warning(
                "Initial scrape failed. Sanctions screening disabled until "
                "the scraper succeeds. The background thread will keep retrying."
            )

    load_sanctions_cache()

    # Background thread: re-scrapes whenever DB is older than REFRESH_INTERVAL_HOURS
    t = threading.Thread(target=_scraper_background_loop, daemon=True, name="sanctions-refresh")
    t.start()

    # --- ML models ---
    if not load_models_from_cache():
        if os.path.exists(CSV_PATH):
            train_models(CSV_PATH)
        else:
            log.warning(f"No models.pkl and no CSV at '{CSV_PATH}'.")
    if os.path.exists(CSV_PATH):
        _ensure_loading()
    yield

app = FastAPI(title="AML Shield", version="3.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def rbac_middleware(request: Request, call_next):
    path = request.url.path
    open_paths = {"/", "/health", "/auth/login", "/docs", "/openapi.json", "/redoc"}
    if path in open_paths or request.method == "OPTIONS":
        return await call_next(request)

    role = "admin"
    allowed = ROLE_ROUTES.get(role, [])

    if "*" in allowed or any(path == r or path.startswith(r + "/") for r in allowed):
        return await call_next(request)

    return JSONResponse(
        status_code=403,
        content={"detail": f"Role '{role}' cannot access {path}"},
        headers={"Access-Control-Allow-Origin": "*"},
    )

from kyc_screening_v2 import router as kyc_router
app.include_router(kyc_router, prefix="/kyc")

# ---------------------------------------------------------------------------
# ── SCHEMAS ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    amount: float; hour: int = 12; payment_type: str = "ACH"; day: str = "Mon"
    sender: str = ""; receiver: str = ""; pay_curr: str = ""; recv_curr: str = ""
    laundering_type: str = ""

class ScreenRequest(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)

class TrainRequest(BaseModel):
    csv_path: str = "SAML-D.csv"

# ---------------------------------------------------------------------------
# ── STATIC FRONTEND ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
# The frontend HTML is in aml_shield.html — served at /
FRONTEND_PATH = Path(__file__).parent / ("frontend_v0.7.html")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if FRONTEND_PATH.exists():
        return HTMLResponse(content=FRONTEND_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>AML Shield</h1><p>Place frontend_v0.7.html in the same folder as main.py</p>")

# ---------------------------------------------------------------------------
# ── SANCTIONS ROUTES ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
@app.get("/sanctions/status")
def get_sanctions_status():
    age_h = (time.time() - _cache_loaded_at) / 3600
    return {
        "entry_count":     len(_sanctions_cache),
        "loaded_at":       datetime.fromtimestamp(_cache_loaded_at, timezone.utc).isoformat(),
        "age_hours":       round(age_h, 2),
        "is_stale":        age_h > REFRESH_INTERVAL_HOURS,
        "db_path":         str(DB_PATH),
        "db_exists":       DB_PATH.exists(),
        "last_scrape_sgt": _last_scrape_sgt or "—",
    }

@app.get("/sanctions/entries")
def get_sanctions_entries(
    limit: int = Query(default=500, le=2000),
    offset: int = Query(default=0),
    entity_type: Optional[str] = None,
    regime: Optional[str] = None,
):
    entries = _sanctions_cache
    if entity_type:
        entries = [e for e in entries if e["type"] == entity_type]
    if regime:
        entries = [e for e in entries if regime.lower() in e["regime"].lower()]
    total = len(entries)
    return {"total": total, "offset": offset, "limit": limit,
            "entries": entries[offset: offset + limit]}

@app.post("/sanctions/check")
def check_sanctions(req: ScreenRequest):
    return screen_name(req.name, req.aliases, req.threshold)

@app.post("/sanctions/refresh", status_code=202)
def refresh_sanctions(background_tasks: BackgroundTasks):
    def _full_refresh():
        log.info("Force refresh requested — running full scraper ...")
        ok = run_scraper()
        if ok:
            load_sanctions_cache()
            log.info("Force refresh complete. Last scrape SGT: %s", _last_scrape_sgt)
        else:
            log.warning("Force refresh: scraper failed — cache not updated")
    background_tasks.add_task(_full_refresh)
    return {"status": "scrape_queued"}

class LoginRequest(BaseModel):
    role: str
    passcode: str

@app.post("/auth/login")
def login(req: LoginRequest):
    expected = ROLE_PASSCODES.get(req.role)
    if not expected or req.passcode != expected:
        raise HTTPException(status_code=401, detail="Invalid role or passcode")
    return {"ok": True, "role": req.role}

# ---------------------------------------------------------------------------
# ── ML ROUTES ────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    feat_imp = []
    if MODEL_READY and "rf" in MODELS:
        fi = sorted(zip(FEATURE_COLS, MODELS["rf"].feature_importances_.tolist()), key=lambda x: -x[1])
        feat_imp = [{"feature": f, "label": FEATURE_LABELS.get(f,f), "importance": round(i,6)} for f,i in fi]
    return {
        "status": "ok", "version": "3.1.0",
        "model_ready": MODEL_READY, "models": list(MODELS.keys()),
        "metrics": METRICS, "csv_path": CSV_PATH,
        "csv_rows": _TX_TOTAL, "cache_ready": _TX_RECORDS is not None,
        "feature_importance": feat_imp,
        "payment_rate": METRICS.get("payment_rate", []),
        "sanctions": {
            "entry_count": len(_sanctions_cache),
            "db_exists": DB_PATH.exists(),
            "age_hours": round((time.time() - _cache_loaded_at) / 3600, 2),
        }
    }

@app.post("/predict")
def predict(tx: TransactionInput):
    if not MODEL_READY:
        raise HTTPException(503, "Models not ready. POST /train first.")
    X = build_input_row(tx.amount, tx.hour, tx.payment_type, tx.day,
                        tx.sender, tx.receiver, tx.pay_curr, tx.recv_curr,
                        tx.laundering_type)
    lp = float(MODELS["logit"].predict_proba(X)[0,1])
    cp = float(MODELS["cart"].predict_proba(X)[0,1])
    rp = float(MODELS["rf"].predict_proba(X)[0,1])
    return {
        "risk_score": round(rp,4), "risk_label": risk_label(rp),
        "cart_prob": round(cp,4), "rf_prob": round(rp,4), "logit_prob": round(lp,4),
        "cart": "Laundering" if cp>0.35 else "Clean",
        "rf":   "Laundering" if rp>0.35 else "Clean",
        "logit":"Laundering" if lp>0.35 else "Clean",
        "cart_label": risk_label(cp), "rf_label": risk_label(rp), "logit_label": risk_label(lp),
        "demo_mode": False,
    }

@app.post("/train")
def train(req: TrainRequest):
    if not os.path.exists(req.csv_path):
        raise HTTPException(404, f"CSV not found: {req.csv_path}")
    global CSV_PATH
    CSV_PATH = req.csv_path
    train_models(req.csv_path)
    return {"status": "trained", "metrics": METRICS}

@app.get("/transactions/progress")
async def transactions_progress():
    async def _stream():
        while True:
            with _TX_LOCK:
                loaded  = len(_TX_RECORDS) if _TX_RECORDS is not None else 0
                total   = _TX_TOTAL
                prog    = _TX_PROGRESS
                loading = _TX_LOADING
                err     = _TX_ERROR
            payload = json.dumps({"progress": round(prog,4), "loaded": loaded,
                                   "total": total, "loading": loading, "error": err})
            yield f"data: {payload}\n\n"
            if not loading and prog >= 1.0:
                break
            await asyncio.sleep(1)
    _ensure_loading()
    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/transactions")
def get_transactions(
    limit: int = 200, offset: int = 0,
    laundering_only: bool = False, sample: bool = False,
    risk: str = "", payment_type: str = "",
):
    _ensure_loading()
    if _TX_ERROR and (_TX_RECORDS is None or len(_TX_RECORDS) == 0):
        raise HTTPException(503, detail=_TX_ERROR)

    with _TX_LOCK:
        all_records   = list(_TX_RECORDS) if _TX_RECORDS else []
        current_total = _TX_TOTAL
        progress      = _TX_PROGRESS

    limit = min(max(1, limit), 2000)
    filtered = all_records
    if laundering_only: filtered = [r for r in filtered if r["is_laundering"] == 1]
    if risk:            filtered = [r for r in filtered if r["risk"] == risk]
    if payment_type:
        pt = payment_type.lower()
        filtered = [r for r in filtered if pt in r["type"].lower()]

    total_filtered = len(filtered)
    if sample:
        n      = min(limit, total_filtered)
        result = random.sample(filtered, n) if n < total_filtered else filtered[:n]
    else:
        result = filtered[offset: offset + limit]

    return {
        "total_in_csv": current_total, "total_loaded": len(all_records),
        "total_filtered": total_filtered, "returned": len(result),
        "offset": offset, "progress": round(progress, 4),
        "loading": _TX_LOADING, "error": _TX_ERROR, "records": result,
    }

@app.get("/transactions/summary")
def transactions_summary():
    _ensure_loading()
    with _TX_LOCK:
        snap     = list(_TX_RECORDS) if _TX_RECORDS else []
        total    = _TX_TOTAL
        progress = _TX_PROGRESS
    from collections import Counter
    n       = len(snap)
    launder = sum(1 for r in snap if r["is_laundering"] == 1)
    high    = sum(1 for r in snap if r["risk"] == "high")
    return {
        "total": total, "total_loaded": n, "progress": round(progress, 4),
        "loading": _TX_LOADING, "laundering": launder,
        "launder_rate": round(launder/n*100, 4) if n else 0,
        "high_risk": high,
        "by_payment_type": dict(Counter(r["type"] for r in snap).most_common(10)),
    }

@app.get("/transactions/alerts")
def get_alert_candidates():
    _ensure_loading()
    with _TX_LOCK:
        snap = list(_TX_RECORDS) if _TX_RECORDS else []
    candidates = [r for r in snap if r["risk"] in ("high", "medium")]
    return {"records": candidates, "total": len(candidates)}

@app.post("/transactions/invalidate")
def invalidate_tx_cache():
    global _TX_RECORDS, _TX_TOTAL, _TX_PROGRESS, _TX_LOADING, _TX_ERROR
    with _TX_LOCK:
        _TX_RECORDS = None; _TX_TOTAL = 0; _TX_PROGRESS = 0.0
        _TX_LOADING = False; _TX_ERROR = None
    _ensure_loading()
    return {"status": "cache_cleared_and_reloading"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Shield — Combined Backend")
    parser.add_argument("--csv",     default="SAML-D.csv")
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    default=8000, type=int)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--db",      default="mas_sanctions.db", help="Path to sanctions SQLite DB")
    args = parser.parse_args()

    CSV_PATH = args.csv
    DB_PATH  = Path(args.db)

    if args.retrain and os.path.exists("models.pkl"):
        os.remove("models.pkl")
        log.info("Deleted models.pkl — will retrain.")

    uvicorn.run(app, host=args.host, port=args.port)