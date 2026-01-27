#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch Eastmoney data and produce:
- data/{stem}.csv    (Date,Price)
- site/{stem}.html   (minimal <table> with Date/Close)
- site/{stem}.json   ([{"date":"YYYY-MM-DD","close":X.YZ}, ...])

Supports two symbol types in data/funds.json:

1) OTC funds (NAV): 6 digits
   e.g. "001316", "022907"
   -> fetch historical NAV (单位净值/累计净值) via type=lsjz

2) LOF (exchange close): 6 digits + .SS / .SZ
   e.g. "166009.SZ", "501018.SS"
   -> fetch daily Kline close price via push2his kline API

Example data/funds.json:
{ "codes": ["001316", "022907", "166009.SZ", "501018.SS"] }
"""

import io
import re
import json
import html
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import requests

# ------------------------ Paths & Constants ------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SITE_DIR = ROOT / "site"
DATA_DIR.mkdir(exist_ok=True)
SITE_DIR.mkdir(exist_ok=True)

# -------- LOF recent window --------
LOF_DAYS = 35   # 只抓最近约 1 个月（含缓冲）

# --- OTC Fund NAV (历史净值) ---
API_TMPL_PAGE_NAV = (
    "https://fund.eastmoney.com/f10/F10DataApi.aspx"
    "?type=lsjz&code={code}&page={page}&per=50&sdate=2000-01-01&edate=2099-12-31"
)

# --- LOF Close (历史K线) ---
KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
KLINE_BASE_PARAMS = {
    "klt": "101",  # 101=日线
    "fqt": "0",    # 0=不复权；1=前复权；2=后复权
    "beg": "20000101",
    "end": "20991231",
    "lmt": "100000",
    "fields1": "f1,f2,f3,f4,f5,f6",
    # f51 日期, f52 开, f53 收, f54 高, f55 低, f56 量, f57 额
    "fields2": "f51,f52,f53,f54,f55,f56,f57",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

RETRY_TIMES = 8          # 每只标的最多重试次数
RETRY_SLEEP_SEC = 2.6      # 失败后的等待
BETWEEN_FUNDS_SLEEP = 4.8  # 相邻标的之间等待，降低限流

# ------------------------ Utilities ------------------------

CODE_NAV_RE = re.compile(r"^\d{6}$")
CODE_LOF_RE = re.compile(r"^(?P<code>\d{6})\.(?P<mkt>SS|SZ)$", re.I)

def log(msg: str) -> None:
    print(msg, flush=True)

def parse_symbol(sym: str) -> Dict[str, Any]:
    """
    Returns dict:
      kind: "nav" | "lof"
      code: 6-digit
      secid: "1.xxxxxx" / "0.xxxxxx" (lof only)
      stem: filename stem (safe, unique)
    """
    s = sym.strip()
    if CODE_NAV_RE.match(s):
        return {"kind": "nav", "code": s, "secid": None, "stem": s, "raw": s}

    m = CODE_LOF_RE.match(s)
    if m:
        code = m.group("code")
        mkt = m.group("mkt").upper()
        secid = ("1." if mkt == "SS" else "0.") + code
        stem = f"{code}_{mkt}"  # avoid dot in filenames; avoid collision with OTC same 6 digits
        return {"kind": "lof", "code": code, "secid": secid, "stem": stem, "raw": s}

    raise ValueError(f"invalid symbol: {sym}")

def request_with_retry(method, url: str, *, params=None, headers=None, timeout=25, expect_json=False) -> Any:
    import random

    last_err = None
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            # 每次重试强制新连接：新 Session + Connection: close
            h = dict(headers or {})
            h.setdefault("Connection", "close")

            with requests.Session() as s:
                r = s.request("GET", url, params=params, headers=h, timeout=timeout)
                r.raise_for_status()
                if expect_json:
                    return r.json()
                return r.text

        except Exception as e:
            last_err = e
            log(f"[WARN] request failed (attempt {attempt}/{RETRY_TIMES}): {e}")

            if attempt < RETRY_TIMES:
                # 指数退避 + 抖动（更容易下一次走到不同后端）
                sleep = min(25, (2 ** (attempt - 1)) * 0.7 + random.uniform(0.2, 1.8))
                time.sleep(sleep)

    raise last_err  # type: ignore
   
def _extract_table_html(js_text: str) -> str:
    """
    从 Eastmoney 的响应中提取第一张 <table>…</table>
    兼容 apidata.content 的转义字符串和直接 HTML 两种返回。
    """
    m = re.search(r'content\s*[:=]\s*"(.+?)"\s*,\s*(records|pages|curpage)', js_text, re.S)
    if m:
        raw = m.group(1)
        s = (raw.replace(r'\"', '"')
                .replace(r"\/", "/")
                .replace(r"\n", "")
                .replace(r"\r", "")
                .replace(r"\t", ""))
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

def _find_header_row(df: pd.DataFrame) -> int:
    """
    在无表头读取的 DataFrame 中，寻找真正的表头行（返回行索引；未找到返回 -1）
    """
    keys = ("净值日期", "单位净值", "累计净值", "日增长率")
    up_to = min(6, len(df))
    for i in range(up_to):
        row_text = "".join(map(str, df.iloc[i].tolist()))
        if any(k in row_text for k in keys):
            return i
    return -1

def _pick_idx(columns: List[str], keys: List[str], default: Optional[int] = None) -> Optional[int]:
    for k in keys:
        for i, c in enumerate(columns):
            if k in str(c):
                return i
    return default

def _to_num_allow_minus(s: pd.Series) -> pd.Series:
    """
    数值清洗：去千分位逗号，保留负号与小数点（处理 '-', '—', '1.234*' 等）
    """
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    )

# ------------------------ Fetchers ------------------------

def fetch_nav_one_page(code: str) -> pd.DataFrame:
    """
    OTC 场外基金：抓取历史净值（单位净值优先，缺失回退累计净值）
    当前默认抓第一页（最近的 50 条），并按日期升序返回 Date, Price。
    """
    url = API_TMPL_PAGE_NAV.format(code=code, page=1)
    log(f"[REQ] NAV {code} -> {url}")
    text = request_with_retry(requests.get, url, headers=HEADERS, timeout=25, expect_json=False)

    table_html = _extract_table_html(text)
    tables = pd.read_html(io.StringIO(table_html), flavor="lxml", header=None)
    if not tables:
        raise RuntimeError("pandas.read_html found no table")
    raw = tables[0].copy()

    raw = raw.dropna(how="all", axis=1)
    raw = raw.dropna(how="all", axis=0).reset_index(drop=True)
    if raw.empty:
        raise RuntimeError("empty table")

    hdr = _find_header_row(raw)
    if hdr >= 0:
        raw.columns = raw.iloc[hdr].astype(str).tolist()
        df = raw.iloc[hdr + 1:].reset_index(drop=True)
    else:
        cols_template = ["净值日期", "单位净值", "累计净值", "日增长率", "申购状态", "赎回状态", "分红送配"]
        first_val = str(raw.iloc[0, 0])
        if re.match(r"^\d{4}-\d{2}-\d{2}$", first_val):
            df = raw.copy()
            df.columns = cols_template[: df.shape[1]]
        else:
            df = raw.copy()
            df.columns = df.iloc[0].astype(str).tolist()
            df = df.iloc[1:].reset_index(drop=True)

    cols = list(map(str, df.columns))
    date_idx = _pick_idx(cols, ["净值日期", "日期"], default=0)
    unit_idx = _pick_idx(cols, ["单位净值", "单位净值(元)", "单位"])
    accu_idx = _pick_idx(cols, ["累计净值", "累计净值(元)", "累计"])

    if date_idx is None:
        raise RuntimeError(f"no date column, headers={cols}")

    date_ser = pd.to_datetime(df.iloc[:, date_idx], errors="coerce").dt.strftime("%Y-%m-%d")
    unit_ser = _to_num_allow_minus(df.iloc[:, unit_idx]) if unit_idx is not None else None
    accu_ser = _to_num_allow_minus(df.iloc[:, accu_idx]) if accu_idx is not None else None

    if unit_ser is not None and accu_ser is not None:
        price = unit_ser.combine_first(accu_ser)
    elif unit_ser is not None:
        price = unit_ser
    elif accu_ser is not None:
        price = accu_ser
    else:
        raise RuntimeError(f"no NAV columns, headers={cols}")

    out = (
        pd.DataFrame({"Date": date_ser, "Price": price})
        .dropna(subset=["Date", "Price"])
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if out.empty:
        raise RuntimeError("parsed empty NAV after cleaning")
    return out

def fetch_lof_close(secid: str, stem: str) -> pd.DataFrame:
    """
    LOF 场内：抓取历史日K收盘价（Close），返回 Date, Price（升序）
    secid: "1.xxxxxx"(沪) / "0.xxxxxx"(深)
    """
    params = dict(KLINE_BASE_PARAMS)
    params["secid"] = secid
    
    # ---- limit to recent ~1 month ----
    end = pd.Timestamp.today().strftime("%Y%m%d")
    beg = (pd.Timestamp.today() - pd.Timedelta(days=LOF_DAYS)).strftime("%Y%m%d")
    params["beg"] = beg
    params["end"] = end
    params["lmt"] = "200"

    headers = {**HEADERS, "Referer": "https://quote.eastmoney.com/"}
    log(f"[REQ] LOF {stem} (secid={secid}) -> {KLINE_URL}")
    j = request_with_retry(requests.get, KLINE_URL, params=params, headers=headers, timeout=25, expect_json=True)

    data = (j or {}).get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise RuntimeError("no kline data (empty klines)")

    rows = [k.split(",") for k in klines]
    df = pd.DataFrame(rows, columns=["Date", "Open", "Close", "High", "Low", "Vol", "Amt"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Price"] = pd.to_numeric(df["Close"], errors="coerce")

    out = (
        df.dropna(subset=["Date", "Price"])[["Date", "Price"]]
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if out.empty:
        raise RuntimeError("parsed empty LOF kline after cleaning")
    return out

# ------------------------ Writers ------------------------

def write_csv(df: pd.DataFrame, stem: str) -> Path:
    p = DATA_DIR / f"{stem}.csv"
    df.to_csv(p, index=False)
    log(f"[OUT] CSV  -> {p}")
    return p

def write_html_table(df: pd.DataFrame, stem: str) -> Path:
    out = df.rename(columns={"Price": "Close"})[["Date", "Close"]].copy()
    table_html = out.to_html(index=False, border=1)
    html_page = f"""<!doctype html>
<meta charset="utf-8">
{table_html}
"""
    p = SITE_DIR / f"{stem}.html"
    p.write_text(html_page, encoding="utf-8")
    log(f"[OUT] HTML -> {p}")
    return p

def write_json(df: pd.DataFrame, stem: str) -> Path:
    clean = df[["Date", "Price"]].copy()
    clean["Price"] = pd.to_numeric(clean["Price"], errors="coerce")
    clean = clean.dropna(subset=["Date", "Price"])
    clean = clean[np.isfinite(clean["Price"].to_numpy())]

    payload = [{"date": d, "close": float(p)} for d, p in clean.itertuples(index=False, name=None)]
    p = SITE_DIR / f"{stem}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
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
        log("[ERR] No codes found in data/funds.json")
        sys.exit(1)

    total_rows = 0
    ok_cnt = 0
    fail_cnt = 0

    for idx, sym in enumerate(codes, 1):
        log(f"[START] ({idx}/{len(codes)}) {sym}")
        try:
            info = parse_symbol(sym)
            stem = info["stem"]

            if info["kind"] == "nav":
                df = fetch_nav_one_page(info["code"])
            else:
                df = fetch_lof_close(info["secid"], stem)

            log(f"[INFO] {sym} -> {stem}: {len(df)} rows")
            write_csv(df, stem)
            write_html_table(df, stem)
            write_json(df, stem)

            total_rows += len(df)
            ok_cnt += 1
            log(f"[DONE] {sym}")
        except Exception as e:
            fail_cnt += 1
            log(f"[FAIL] {sym}: {e}")

        time.sleep(BETWEEN_FUNDS_SLEEP)

    if ok_cnt == 0:
        log("[WARN] No data generated for any code.")
    log(f"[OK] Finished. success={ok_cnt}, fail={fail_cnt}, total_rows={total_rows}")

if __name__ == "__main__":
    main()
