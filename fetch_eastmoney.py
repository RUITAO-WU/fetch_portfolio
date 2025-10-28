#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch Eastmoney fund historical NAVs and produce:
- data/{code}.csv    (Date,Price)
- site/{code}.html   (minimal <table> with Date/Close)
- site/{code}.json   ([{"date":"YYYY-MM-DD","close":X.YZ}, ...])

Fund codes in data/funds.json, e.g.:
{ "codes": ["001316", "022907"] }
"""

import io
import re
import json
import html
import sys
import time
import datetime as dt
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests

# ------------------------ Paths & Constants ------------------------

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

RETRY_TIMES = 3
RETRY_SLEEP_SEC = 3
BETWEEN_FUNDS_SLEEP = 2

# ------------------------ Utils ------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def _extract_table_html(js_text: str) -> str:
    m = re.search(r'content\s*[:=]\s*"(.+?)"\s*,\s*(records|pages|curpage)', js_text, re.S)
    if m:
        raw = m.group(1)
        s = raw.replace(r'\"', '"').replace(r"\/", "/")
        s = s.replace(r"\n", "").replace(r"\r", "").replace(r"\t", "")
        try:
            s = s.encode("utf-8").decode("unicode_escape", errors="ignore")
        except Exception:
            pass
        s = html.unescape(s)
        t = re.search(r"<table[\s\S]*?</table>", s, re.I)
        if t:
            return t.group(0)
    t2 = re.search(r"<table[\s\S]*?</table>", js_text, re.I)
    if t2:
        return t2.group(0)
    raise RuntimeError("Failed to locate <table> HTML in response")

def _pick_col_index(cols: List[str], keywords: List[str], default_idx: Optional[int]=None) -> Optional[int]:
    for kw in keywords:
        for i, c in enumerate(cols):
            if kw in str(c):
                return i
    if default_idx is not None and 0 <= default_idx < len(cols):
        return default_idx
    return None

def _first_numeric_col_index(df: pd.DataFrame, exclude_idx: Optional[int]) -> Optional[int]:
    for i in range(len(df.columns)):
        if exclude_idx is not None and i == exclude_idx:
            continue
        s = df.iloc[:, i].astype(str).str.replace(",", "", regex=False)
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() >= max(3, int(len(s) * 0.2)):
            return i
    return None

# ------------------------ Core ------------------------

def fetch_one(code: str) -> pd.DataFrame:
    import io, re, html, pandas as pd, requests
    from pandas import DataFrame

    def _extract_table_html(js_text: str) -> str:
        m = re.search(r'content\s*[:=]\s*"(.+?)"\s*,\s*(records|pages|curpage)', js_text, re.S)
        if m:
            raw = m.group(1)
            s = (raw.replace(r'\"','"')
                   .replace(r"\/","/")
                   .replace(r"\n","").replace(r"\r","").replace(r"\t",""))
            s = s.encode("utf-8").decode("unicode_escape", errors="ignore")
            s = html.unescape(s)
            t = re.search(r"<table[\s\S]*?</table>", s, re.I)
            if t: return t.group(0)
        t2 = re.search(r"<table[\s\S]*?</table>", js_text, re.I)
        if t2: return t2.group(0)
        raise RuntimeError("no <table> in response")

    # 取第1页（足够覆盖最新行），不改你的分页策略
    url = API_TMPL.format(code=code)
    r = requests.get(url, headers=HEADERS, timeout=25); r.raise_for_status()
    table_html = _extract_table_html(r.text)

    tables = pd.read_html(io.StringIO(table_html), flavor="lxml")
    if not tables:
        raise RuntimeError("pandas.read_html found no table")
    df = tables[0].copy()

    # 首行可能是表头
    if not any(("日" in str(c) or "期" in str(c)) for c in df.columns):
        df.columns = df.iloc[0]; df = df.iloc[1:].reset_index(drop=True)

    # 列名索引（用下标避免重名列导致 Series/DataFrame 混淆）
    cols = list(df.columns)

    def pick_idx(columns, keys, default=None):
        for k in keys:
            for i, c in enumerate(columns):
                if k in str(c): return i
        return default

    date_idx = pick_idx(cols, ["净值日期", "日期"], default=0)
    unit_idx = pick_idx(cols, ["单位净值", "单位净值(元)", "单位"])
    accu_idx = pick_idx(cols, ["累计净值", "累计净值(元)", "累计"])

    if date_idx is None:
        raise RuntimeError(f"no date column, headers={cols}")
    if unit_idx is None and accu_idx is None:
        raise RuntimeError(f"no NAV columns, headers={cols}")

    # 清洗函数：只保留数字与小数点（防止出现 '1.2345*'、'—' 等）
    import numpy as np
    def to_num_series(s):
        return pd.to_numeric(
            s.astype(str).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce"
        )

    date_ser = pd.to_datetime(df.iloc[:, date_idx], errors="coerce").dt.strftime("%Y-%m-%d")

    unit_ser = to_num_series(df.iloc[:, unit_idx]) if unit_idx is not None else None
    accu_ser = to_num_series(df.iloc[:, accu_idx]) if accu_idx is not None else None

    # 行级回退：优先单位净值；若为空则用累计净值
    if unit_ser is not None and accu_ser is not None:
        price = unit_ser.combine_first(accu_ser)
    else:
        price = unit_ser if unit_ser is not None else accu_ser

    out = pd.DataFrame({"Date": date_ser, "Price": price}) \
            .dropna(subset=["Date", "Price"]) \
            .drop_duplicates() \
            .sort_values("Date") \
            .reset_index(drop=True)

    if out.empty:
        raise RuntimeError("parsed empty after unit->accu fallback")
    return out
# ------------------------ Writers ------------------------

def write_csv(df: pd.DataFrame, code: str) -> Path:
    p = DATA_DIR / f"{code}.csv"
    df.to_csv(p, index=False)
    log(f"[OUT] CSV  -> {p}")
    return p

def write_html_table(df: pd.DataFrame, code: str) -> Path:
    out = df.rename(columns={"Price": "Close"})[["Date", "Close"]].copy()
    table_html = out.to_html(index=False, border=1)
    html_page = f"""<!doctype html>
<meta charset="utf-8">
{table_html}
"""
    p = SITE_DIR / f"{code}.html"
    p.write_text(html_page, encoding="utf-8")
    log(f"[OUT] HTML -> {p}")
    return p

def write_json(df: pd.DataFrame, code: str) -> Path:
    payload = [{"date": d, "close": float(p)} for d, p in df[["Date", "Price"]].itertuples(index=False, name=None)]
    p = SITE_DIR / f"{code}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log(f"[OUT] JSON -> {p}")
    return p

# ------------------------ Main ------------------------

def main() -> None:
    cfg_path = DATA_DIR / "funds.json"
    if not cfg_path.exists():
        log(f"[ERR] Missing config: {cfg_path}")
        sys.exit(1)

    conf = json.loads(cfg_path.read_text(encoding="utf-8"))
    codes: List[str] = conf.get("codes", [])
    if not codes:
        log("[ERR] No fund codes found in data/funds.json")
        sys.exit(1)

    total_rows = 0
    ok_cnt = 0
    fail_cnt = 0

    for idx, code in enumerate(codes, 1):
        log(f"[START] ({idx}/{len(codes)}) {code}")
        try:
            df = fetch_one(code)
            log(f"[INFO] {code}: {len(df)} rows")
            write_csv(df, code)
            write_html_table(df, code)
            write_json(df, code)
            total_rows += len(df)
            ok_cnt += 1
            log(f"[DONE] {code}")
        except Exception as e:
            fail_cnt += 1
            log(f"[FAIL] {code}: {e}")
        time.sleep(BETWEEN_FUNDS_SLEEP)

    if ok_cnt == 0:
        log("[WARN] No data generated for any code.")
    log(f"[OK] Finished. success={ok_cnt}, fail={fail_cnt}, total_rows={total_rows}")

if __name__ == "__main__":
    main()
