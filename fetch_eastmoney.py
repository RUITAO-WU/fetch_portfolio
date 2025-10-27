#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch Eastmoney fund historical NAVs and produce:
- data/{code}.csv   (Date,Price)
- site/{code}.html  (HTML table usable by Portfolio Performance: Table on website)

Usage: run directly. Fund codes are read from data/funds.json:
{
  "codes": ["001316"]
}
"""

import io
import re
import json
import html
import sys
import datetime as dt
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SITE_DIR = ROOT / "site"
DATA_DIR.mkdir(exist_ok=True)
SITE_DIR.mkdir(exist_ok=True)

API_TMPL = (
    "https://fund.eastmoney.com/f10/F10DataApi.aspx"
    "?type=lsjz&code={code}&page=1&per=5000&sdate=2000-01-01&edate=2099-12-31"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def log(msg: str):
    print(msg, flush=True)


def _extract_table_html(js_text: str) -> str:
    """
    Extract the first <table>...</table> from Eastmoney's JS payload.
    The HTML table is usually embedded in apidata.content with escapes.
    """
    # Preferred: apidata.content = "...."
    m = re.search(r'content\s*[:=]\s*"(.+?)"\s*,\s*(records|pages|curpage)', js_text, re.S)
    if m:
        raw = m.group(1)
        # Undo common JS escapes
        s = raw.replace(r'\"', '"').replace(r"\/", "/")
        s = s.replace(r"\n", "").replace(r"\r", "").replace(r"\t", "")
        # Decode \uXXXX etc.
        try:
            s = s.encode("utf-8").decode("unicode_escape", errors="ignore")
        except Exception:
            pass
        # Decode HTML entities
        s = html.unescape(s)
        t = re.search(r"<table[\s\S]*?</table>", s, re.I)
        if t:
            return t.group(0)

    # Fallback: sometimes the response is already plain HTML
    t2 = re.search(r"<table[\s\S]*?</table>", js_text, re.I)
    if t2:
        return t2.group(0)

    raise RuntimeError("Failed to locate <table> HTML in response")


def _pick_column(columns, keywords, default_idx=None):
    for kw in keywords:
        for c in columns:
            if kw in str(c):
                return c
    if default_idx is not None and 0 <= default_idx < len(columns):
        return columns[default_idx]
    return None


def fetch_one(code: str) -> pd.DataFrame:
    url = API_TMPL.format(code=code)
    log(f"[REQ] {url}")
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()

    table_html = _extract_table_html(r.text)

    # read_html prefers file-like to avoid deprecation
    tables = pd.read_html(io.StringIO(table_html), flavor="lxml")
    if not tables:
        raise RuntimeError("pandas.read_html found no table")

    df = tables[0].copy()

    # In some cases the first row is actually the header; fix it
    if not any(("日" in str(c) or "期" in str(c)) for c in df.columns):
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

    cols = list(df.columns)

    date_col = _pick_column(cols, ["净值日期", "日期"], default_idx=0)
    if date_col is None:
        raise RuntimeError(f"Cannot find date column. Headers: {cols}")

    nav_col = _pick_column(cols, ["单位净值", "单位净值(元)", "单位", "净值"], default_idx=1)
    if nav_col is None:
        raise RuntimeError(f"Cannot find NAV column. Headers: {cols}")

    out = df[[date_col, nav_col]].rename(columns={date_col: "Date", nav_col: "Price"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out = out.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Parsed data is empty")
    return out


def write_csv(df: pd.DataFrame, code: str) -> Path:
    p = DATA_DIR / f"{code}.csv"
    df.to_csv(p, index=False)
    log(f"[OUT] CSV -> {p}")
    return p

def write_html_table(df: pd.DataFrame, code: str) -> Path:
    # 输出两列且列名固定为 Date / Close（PP 识别更稳）
    out = df.rename(columns={"Price": "Close"})[["Date", "Close"]].copy()

    # 生成极简 HTML，首元素就是 table
    table_html = out.to_html(index=False, border=1)  # <table>...两列
    html_page = f"""<!doctype html>
<meta charset="utf-8">
{table_html}
"""
    p = SITE_DIR / f"{code}.html"
    p.write_text(html_page, encoding="utf-8")
    return p


def main():
    cfg_path = DATA_DIR / "funds.json"
    if not cfg_path.exists():
        log(f"[ERR] Missing config: {cfg_path}")
        sys.exit(1)

    conf = json.loads(cfg_path.read_text(encoding="utf-8"))
    codes = conf.get("codes", [])
    if not codes:
        log("[ERR] No fund codes found in data/funds.json")
        sys.exit(1)

    total_rows = 0
    for code in codes:
        try:
            log(f"[START] Fetching {code} ...")
            df = fetch_one(code)
            log(f"[INFO] Got {len(df)} rows for {code}")
            write_csv(df, code)
            write_html_table(df, code)
            total_rows += len(df)
            log(f"[DONE] {code}")
        except Exception as e:
            log(f"[FAIL] {code}: {e}")
            # continue with other codes, but remember failure
            continue

    if total_rows == 0:
        # Make the failure visible to CI logs but do not crash deployment steps that
        # conditionally upload 'site/*.html'
        log("[WARN] No data generated for any code. Check logs above.")
    else:
        log(f"[OK] Finished. Total rows: {total_rows}")


if __name__ == "__main__":
    main()
