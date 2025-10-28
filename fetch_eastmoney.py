#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch Eastmoney fund historical NAVs and produce:
- data/{code}.csv    (Date,Price)
- site/{code}.html   (minimal <table> with Date/Close)
- site/{code}.json   ([{"date":"YYYY-MM-DD","close":X.YZ}, ...])

Fund codes are read from data/funds.json, e.g.:
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

API_TMPL_PAGE = (
    "https://fund.eastmoney.com/f10/F10DataApi.aspx"
    "?type=lsjz&code={code}&page={page}&per=50&sdate=2000-01-01&edate=2099-12-31"
)
API_TMPL_PROBE = (
    "https://fund.eastmoney.com/f10/F10DataApi.aspx"
    "?type=lsjz&code={code}&page=1&per=50&sdate=2000-01-01&edate=2099-12-31"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

RETRY_TIMES = 3          # 每只基金最多重试
RETRY_SLEEP_SEC = 3      # 失败后的等待
BETWEEN_FUNDS_SLEEP = 2  # 相邻基金之间等待，降低限流
MAX_PAGES = None         # None 表示抓取全部页；如需仅抓第一页，可设为 1

# ------------------------ Utilities ------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

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

def _pick_idx(columns: List[str], keys: List[str], default: Optional[int]=None) -> Optional[int]:
    for k in keys:
        for i, c in enumerate(columns):
            if k in str(c):
                return i
    return default

def _to_num(s: pd.Series) -> pd.Series:
    """
    数值清洗：去千分位逗号，剔除非数字/小数点字符（处理 '-', '—', '1.234*' 等）
    """
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce"
    )

def _read_one_page(code: str, page: int) -> pd.DataFrame:
    """
    读取单页，强制无表头（header=None），在表内定位真正表头行然后标准化为两列：Date, Price
    仅做结构化，不做去重/排序。
    """
    url = API_TMPL_PAGE.format(code=code, page=page)
    log(f"[REQ] {code} page {page} -> {url}")
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    table_html = _extract_table_html(r.text)

    # 关键：强制无表头读取，避免“首行数据被当表头”
    tables = pd.read_html(io.StringIO(table_html), flavor="lxml", header=None)
    if not tables:
        raise RuntimeError("pandas.read_html found no table")
    raw = tables[0].copy()

    # 去全空列/行
    raw = raw.dropna(how="all", axis=1)
    raw = raw.dropna(how="all", axis=0).reset_index(drop=True)

    if raw.empty:
        return pd.DataFrame(columns=["Date", "Price"])

    # 寻找真正的表头行
    hdr = _find_header_row(raw)
    if hdr >= 0:
        raw.columns = raw.iloc[hdr].astype(str).tolist()
        df = raw.iloc[hdr+1:].reset_index(drop=True)
    else:
        # 找不到表头行：如果首列像日期，则用模板列名；否则退而求其次用第一行做表头
        cols_template = ["净值日期","单位净值","累计净值","日增长率","申购状态","赎回状态","分红送配"]
        first_val = str(raw.iloc[0, 0])
        if re.match(r"^\d{4}-\d{2}-\d{2}$", first_val):
            df = raw.copy()
            df.columns = cols_template[: df.shape[1]]
        else:
            df = raw.copy()
            df.columns = df.iloc[0].astype(str).tolist()
            df = df.iloc[1:].reset_index(drop=True)

    # 选择列（名称优先，失败再兜底数值列）
    cols = list(map(str, df.columns))
    date_idx = _pick_idx(cols, ["净值日期", "日期"], default=0)
    unit_idx = _pick_idx(cols, ["单位净值", "单位净值(元)", "单位"])
    accu_idx = _pick_idx(cols, ["累计净值", "累计净值(元)", "累计"])

    if date_idx is None:
        # 无日期列则直接返回空
        return pd.DataFrame(columns=["Date", "Price"])

    # 行级回退：优先单位净值；若为空则用累计净值；再不行尝试首个数值列
    date_ser = pd.to_datetime(df.iloc[:, date_idx], errors="coerce").dt.strftime("%Y-%m-%d")
    unit_ser = _to_num(df.iloc[:, unit_idx]) if unit_idx is not None else None
    accu_ser = _to_num(df.iloc[:, accu_idx]) if accu_idx is not None else None

    if unit_ser is not None and accu_ser is not None:
        price = unit_ser.combine_first(accu_ser)
    elif unit_ser is not None:
        price = unit_ser
    elif accu_ser is not None:
        price = accu_ser
    else:
        # 兜底：从第2列起找首个更像数值的列
        best_i, best_cnt = None, -1
        for i in range(1, df.shape[1]):
            s = _to_num(df.iloc[:, i])
            c = s.notna().sum()
            if c > best_cnt:
                best_i, best_cnt = i, c
        if best_i is None:
            return pd.DataFrame(columns=["Date", "Price"])
        price = _to_num(df.iloc[:, best_i])

    out = pd.DataFrame({"Date": date_ser, "Price": price})
    out = out.dropna(subset=["Date", "Price"])
    return out

def _probe_total_pages(code: str) -> int:
    """
    探测总页数；若未能解析则返回 1。
    """
    url = API_TMPL_PROBE.format(code=code)
    log(f"[PROBE] {code} -> {url}")
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    text = r.text
    m_pages = re.search(r'pages\s*:\s*(\d+)', text)
    pages = int(m_pages.group(1)) if m_pages else 1
    if MAX_PAGES is not None:
        pages = min(pages, MAX_PAGES)
    log(f"[INFO] {code}: total pages = {pages}")
    return max(1, pages)

# ------------------------ Core fetch (with retry) ------------------------

def fetch_one(code: str) -> pd.DataFrame:
    """
    只抓第 1 页；修正表头误判/乱码；单位净值缺失时回退累计净值。
    返回列：Date, Price（升序）。
    """
    import io, re, html, pandas as pd, requests

    def _extract_table_html(js_text: str) -> str:
        m = re.search(r'content\s*[:=]\s*"(.+?)"\s*,\s*(records|pages|curpage)', js_text, re.S)
        if m:
            raw = m.group(1)
            s = (raw.replace(r'\"','"')
                   .replace(r"\/","/")
                   .replace(r"\n","").replace(r"\r","").replace(r"\t",""))
            try:
                s = s.encode("utf-8").decode("unicode_escape", errors="ignore")
            except Exception:
                pass
            s = html.unescape(s)
            t = re.search(r"<table[\s\S]*?</table>", s, re.I)
            if t: return t.group(0)
        t2 = re.search(r"<table[\s\S]*?</table>", js_text, re.I)
        if t2: return t2.group(0)
        raise RuntimeError("no <table> in response")

    def _find_header_row(df: pd.DataFrame) -> int:
        keys = ("净值日期", "单位净值", "累计净值", "日增长率")
        for i in range(min(6, len(df))):
            row_text = "".join(map(str, df.iloc[i].tolist()))
            if any(k in row_text for k in keys):
                return i
        return -1

    def _pick_idx(columns, keys, default=None):
        for k in keys:
            for i, c in enumerate(columns):
                if k in str(c):
                    return i
        return default

    def _to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(
            s.astype(str)
             .str.replace(",", "", regex=False)
             .str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce"
        )

    # —— 仅第 1 页（per 放大以覆盖更多近期记录） ——
    url = API_TMPL_PAGE.format(code=code, page=1)
    log(f"[REQ] {code} -> {url}")
    r = requests.get(url, headers=HEADERS, timeout=25); r.raise_for_status()
    table_html = _extract_table_html(r.text)

    # 强制无表头读取，防“首行数据当表头”
    tables = pd.read_html(io.StringIO(table_html), flavor="lxml", header=None)
    if not tables:
        raise RuntimeError("pandas.read_html found no table")
    raw = tables[0].copy()

    # 去空列/空行
    raw = raw.dropna(how="all", axis=1)
    raw = raw.dropna(how="all", axis=0).reset_index(drop=True)
    if raw.empty:
        raise RuntimeError("empty table")

    # 在表内定位真实表头
    hdr = _find_header_row(raw)
    if hdr >= 0:
        raw.columns = raw.iloc[hdr].astype(str).tolist()
        df = raw.iloc[hdr+1:].reset_index(drop=True)
    else:
        cols_template = ["净值日期","单位净值","累计净值","日增长率","申购状态","赎回状态","分红送配"]
        first_val = str(raw.iloc[0,0])
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
    unit_ser = _to_num(df.iloc[:, unit_idx]) if unit_idx is not None else None
    accu_ser = _to_num(df.iloc[:, accu_idx]) if accu_idx is not None else None

    # 行级回退：单位净值缺 -> 用累计净值
    if unit_ser is not None and accu_ser is not None:
        price = unit_ser.combine_first(accu_ser)
    elif unit_ser is not None:
        price = unit_ser
    elif accu_ser is not None:
        price = accu_ser
    else:
        raise RuntimeError(f"no NAV columns, headers={cols}")

    out = (pd.DataFrame({"Date": date_ser, "Price": price})
           .dropna(subset=["Date","Price"])
           .drop_duplicates()
           .sort_values("Date")
           .reset_index(drop=True))

    if out.empty:
        raise RuntimeError("parsed empty after fixes")
    return out

# ------------------------ Writers ------------------------

def write_csv(df: pd.DataFrame, code: str) -> Path:
    p = DATA_DIR / f"{code}.csv"
    df.to_csv(p, index=False)
    log(f"[OUT] CSV  -> {p}")
    return p

def write_html_table(df: pd.DataFrame, code: str) -> Path:
    # PP 对两列英文字段名最稳：Date / Close；尽量简洁，首元素即表格
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
