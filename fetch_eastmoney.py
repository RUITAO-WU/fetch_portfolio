#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch Eastmoney fund historical NAVs and produce:
- data/{code}.csv    (Date,Price)
- site/{code}.html   (minimal <table> with Date/Close)
- site/{code}.json   ([{"date":"YYYY-MM-DD","close":X.YZ}, ...])

Fund codes are read from data/funds.json, e.g.:
{ "codes": ["001316", "161725"] }
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

RETRY_TIMES = 3          # 每只基金最多重试次数
RETRY_SLEEP_SEC = 3      # 单次失败后的等待
BETWEEN_FUNDS_SLEEP = 2  # 相邻两只基金之间的等待（防限流）

# ------------------------ Utilities ------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def _extract_table_html(js_text: str) -> str:
    """
    从 Eastmoney 的响应中提取第一张 <table>…</table>
    兼容 apidata.content 的转义字符串和直接 HTML 两种返回。
    """
    # 优先：从 apidata.content 里取
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

    # 兜底：响应本身已是 HTML
    t2 = re.search(r"<table[\s\S]*?</table>", js_text, re.I)
    if t2:
        return t2.group(0)

    raise RuntimeError("Failed to locate <table> HTML in response")

def _pick_column(columns: List[str], keywords: List[str], default_idx: Optional[int]=None) -> Optional[str]:
    for kw in keywords:
        for c in columns:
            if kw in str(c):
                return c
    if default_idx is not None and 0 <= default_idx < len(columns):
        return columns[default_idx]
    return None

def _first_numeric_column(df: pd.DataFrame, exclude_cols: List[str]) -> Optional[str]:
    """
    在 DataFrame 中找第一个可转换为数值的列（排除 exclude_cols）。
    """
    for c in df.columns:
        if c in exclude_cols:
            continue
        s = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
        if s.notna().sum() >= max(3, int(len(s) * 0.2)):  # 有一定比例可转为数值
            return c
    return None

# ------------------------ Core Fetch ------------------------

def fetch_one(code: str) -> pd.DataFrame:
    """
    抓取单只基金，带重试与解析兜底。
    返回标准列：Date, Price
    """
    url = API_TMPL.format(code=code)
    last_err = None

    for attempt in range(1, RETRY_TIMES + 1):
        try:
            log(f"[REQ] {code} -> {url} (try {attempt}/{RETRY_TIMES})")
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()

            table_html = _extract_table_html(r.text)
            tables = pd.read_html(io.StringIO(table_html), flavor="lxml")
            if not tables:
                raise RuntimeError("pandas.read_html found no table")

            df = tables[0].copy()

            # 某些返回第一行是表头，显式设置
            if not any(("日" in str(c) or "期" in str(c)) for c in df.columns):
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)

            # 清洗：去除全空列/空行
            df = df.loc[:, ~df.columns.astype(str).str.fullmatch(r"\s*")]
            df = df.dropna(how="all").reset_index(drop=True)

            cols = list(df.columns)

            # 日期列
            date_col = _pick_column(cols, ["净值日期", "日期"], default_idx=0)
            if date_col is None:
                raise RuntimeError(f"Cannot find date column. Headers: {cols}")

            # 净值列：优先关键词；不行则找首个数值列
            nav_col = _pick_column(cols, ["单位净值", "单位净值(元)", "单位", "净值"], default_idx=None)
            if nav_col is None:
                nav_col = _first_numeric_column(df, exclude_cols=[date_col])
            if nav_col is None:
                raise RuntimeError(f"Cannot find numeric NAV column. Headers: {cols}")

            out = df[[date_col, nav_col]].rename(columns={date_col: "Date", nav_col: "Price"}).copy()

            # 规范化
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
            # 去掉千分位逗号
            out["Price"] = pd.to_numeric(out["Price"].astype(str).str.replace(",", ""), errors="coerce")
            out = out.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)

            if out.empty:
                raise RuntimeError("Parsed data is empty")

            return out

        except Exception as e:
            last_err = e
            log(f"[WARN] {code} attempt {attempt} failed: {e}")
            if attempt < RETRY_TIMES:
                time.sleep(RETRY_SLEEP_SEC)

    # 重试失败
    raise RuntimeError(f"Failed after {RETRY_TIMES} attempts for {code}: {last_err}")

# ------------------------ Writers ------------------------

def write_csv(df: pd.DataFrame, code: str) -> Path:
    p = DATA_DIR / f"{code}.csv"
    df.to_csv(p, index=False)
    log(f"[OUT] CSV  -> {p}")
    return p

def write_html_table(df: pd.DataFrame, code: str) -> Path:
    # PP 对两列英文字段名最稳：Date / Close；放在文档最前面且尽量简洁
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
    payload = [
        {"date": d, "close": float(p)}
        for d, p in df[["Date", "Price"]].itertuples(index=False, name=None)
    ]
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
        # 相邻两只基金之间等待，降低被限流概率
        time.sleep(BETWEEN_FUNDS_SLEEP)

    if ok_cnt == 0:
        log("[WARN] No data generated for any code.")
    log(f"[OK] Finished. success={ok_cnt}, fail={fail_cnt}, total_rows={total_rows}")

if __name__ == "__main__":
    main()
