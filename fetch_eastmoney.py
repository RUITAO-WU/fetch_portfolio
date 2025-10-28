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
    """
    解决：部分基金表格首行被误判为表头，或表头乱码；并支持单位净值缺失时回退累计净值。
    """
    import io, re, html, pandas as pd, requests
    from typing import Optional, List

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
        """返回表头所在行索引；未找到则 -1。"""
        keys = ("净值日期", "单位净值", "累计净值", "日增长率")
        for i in range(min(5, len(df))):  # 只在前几行里找
            row = "".join(map(str, df.iloc[i].tolist()))
            if any(k in row for k in keys):
                return i
        return -1

    def _pick_idx(columns: List[str], keys: List[str], default: Optional[int]=None) -> Optional[int]:
        for k in keys:
            for i, c in enumerate(columns):
                if k in str(c):
                    return i
        return default

    def _to_num(s: pd.Series) -> pd.Series:
        # 只保留数字与小数点，去千分位
        return pd.to_numeric(
            s.astype(str)
             .str.replace(",", "", regex=False)
             .str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce"
        )

    # —— 抓取第 1 页（最新行所在页）；如需全量可叠加分页逻辑 ——
    url = API_TMPL.format(code=code)
    r = requests.get(url, headers=HEADERS, timeout=25); r.raise_for_status()
    table_html = _extract_table_html(r.text)

    # 关键：强制无表头读取
    tables = pd.read_html(io.StringIO(table_html), flavor="lxml", header=None)
    if not tables:
        raise RuntimeError("pandas.read_html found no table")
    raw = tables[0].copy()

    # 去除全空列/行
    raw = raw.dropna(how="all", axis=1)
    raw = raw.dropna(how="all", axis=0).reset_index(drop=True)

    # 在表内定位真正表头行
    hdr = _find_header_row(raw)
    if hdr >= 0:
        raw.columns = raw.iloc[hdr].astype(str).tolist()
        df = raw.iloc[hdr+1:].reset_index(drop=True)
    else:
        # 找不到表头行：若首列长得像日期，则用已知模板回填列名
        cols_template = ["净值日期","单位净值","累计净值","日增长率","申购状态","赎回状态","分红送配"]
        if re.match(r"^\d{4}-\d{2}-\d{2}$", str(raw.iloc[0,0])):
            df = raw.copy()
            df.columns = cols_template[: df.shape[1]]
        else:
            # 退一步：把第一行当表头（比把数据行当表头强）
            df = raw.copy()
            df.columns = df.iloc[0].astype(str).tolist()
            df = df.iloc[1:].reset_index(drop=True)

    # 选择列（按名字；失败再兜底）
    cols = list(map(str, df.columns))
    date_idx = _pick_idx(cols, ["净值日期", "日期"], default=0)
    unit_idx = _pick_idx(cols, ["单位净值", "单位净值(元)", "单位"])
    accu_idx = _pick_idx(cols, ["累计净值", "累计净值(元)", "累计"])

    if date_idx is None:
        raise RuntimeError(f"no date column, headers={cols}")
    if unit_idx is None and accu_idx is None:
        # 再兜底：从第2列起找首个“多数可转数值”的列当净值
        best_i, best_cnt = None, -1
        for i in range(1, df.shape[1]):
            s = _to_num(df.iloc[:, i])
            c = s.notna().sum()
            if c > best_cnt:
                best_i, best_cnt = i, c
        if best_i is None:
            raise RuntimeError(f"no NAV columns, headers={cols}")
        unit_idx = best_i  # 用兜底列

    # 行级回退：单位净值缺时用累计净值
    date_ser = pd.to_datetime(df.iloc[:, date_idx], errors="coerce").dt.strftime("%Y-%m-%d")
    unit_ser = _to_num(df.iloc[:, unit_idx]) if unit_idx is not None else None
    accu_ser = _to_num(df.iloc[:, accu_idx]) if accu_idx is not None else None
    price = unit_ser.combine_first(accu_ser) if (unit_ser is not None and accu_ser is not None) \
            else (unit_ser if unit_ser is not None else accu_ser)

    out = pd.DataFrame({"Date": date_ser, "Price": price}) \
           .dropna(subset=["Date","Price"]) \
           .drop_duplicates() \
           .sort_values("Date") \
           .reset_index(drop=True)

    if out.empty:
        raise RuntimeError("parsed empty after header-fix and unit->accu fallback")
    return outdef fetch_one(code: str) -> pd.DataFrame:
    """
    解决：部分基金表格首行被误判为表头，或表头乱码；并支持单位净值缺失时回退累计净值。
    """
    import io, re, html, pandas as pd, requests
    from typing import Optional, List

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
        """返回表头所在行索引；未找到则 -1。"""
        keys = ("净值日期", "单位净值", "累计净值", "日增长率")
        for i in range(min(5, len(df))):  # 只在前几行里找
            row = "".join(map(str, df.iloc[i].tolist()))
            if any(k in row for k in keys):
                return i
        return -1

    def _pick_idx(columns: List[str], keys: List[str], default: Optional[int]=None) -> Optional[int]:
        for k in keys:
            for i, c in enumerate(columns):
                if k in str(c):
                    return i
        return default

    def _to_num(s: pd.Series) -> pd.Series:
        # 只保留数字与小数点，去千分位
        return pd.to_numeric(
            s.astype(str)
             .str.replace(",", "", regex=False)
             .str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce"
        )

    # —— 抓取第 1 页（最新行所在页）；如需全量可叠加分页逻辑 ——
    url = API_TMPL.format(code=code)
    r = requests.get(url, headers=HEADERS, timeout=25); r.raise_for_status()
    table_html = _extract_table_html(r.text)

    # 关键：强制无表头读取
    tables = pd.read_html(io.StringIO(table_html), flavor="lxml", header=None)
    if not tables:
        raise RuntimeError("pandas.read_html found no table")
    raw = tables[0].copy()

    # 去除全空列/行
    raw = raw.dropna(how="all", axis=1)
    raw = raw.dropna(how="all", axis=0).reset_index(drop=True)

    # 在表内定位真正表头行
    hdr = _find_header_row(raw)
    if hdr >= 0:
        raw.columns = raw.iloc[hdr].astype(str).tolist()
        df = raw.iloc[hdr+1:].reset_index(drop=True)
    else:
        # 找不到表头行：若首列长得像日期，则用已知模板回填列名
        cols_template = ["净值日期","单位净值","累计净值","日增长率","申购状态","赎回状态","分红送配"]
        if re.match(r"^\d{4}-\d{2}-\d{2}$", str(raw.iloc[0,0])):
            df = raw.copy()
            df.columns = cols_template[: df.shape[1]]
        else:
            # 退一步：把第一行当表头（比把数据行当表头强）
            df = raw.copy()
            df.columns = df.iloc[0].astype(str).tolist()
            df = df.iloc[1:].reset_index(drop=True)

    # 选择列（按名字；失败再兜底）
    cols = list(map(str, df.columns))
    date_idx = _pick_idx(cols, ["净值日期", "日期"], default=0)
    unit_idx = _pick_idx(cols, ["单位净值", "单位净值(元)", "单位"])
    accu_idx = _pick_idx(cols, ["累计净值", "累计净值(元)", "累计"])

    if date_idx is None:
        raise RuntimeError(f"no date column, headers={cols}")
    if unit_idx is None and accu_idx is None:
        # 再兜底：从第2列起找首个“多数可转数值”的列当净值
        best_i, best_cnt = None, -1
        for i in range(1, df.shape[1]):
            s = _to_num(df.iloc[:, i])
            c = s.notna().sum()
            if c > best_cnt:
                best_i, best_cnt = i, c
        if best_i is None:
            raise RuntimeError(f"no NAV columns, headers={cols}")
        unit_idx = best_i  # 用兜底列

    # 行级回退：单位净值缺时用累计净值
    date_ser = pd.to_datetime(df.iloc[:, date_idx], errors="coerce").dt.strftime("%Y-%m-%d")
    unit_ser = _to_num(df.iloc[:, unit_idx]) if unit_idx is not None else None
    accu_ser = _to_num(df.iloc[:, accu_idx]) if accu_idx is not None else None
    price = unit_ser.combine_first(accu_ser) if (unit_ser is not None and accu_ser is not None) \
            else (unit_ser if unit_ser is not None else accu_ser)

    out = pd.DataFrame({"Date": date_ser, "Price": price}) \
           .dropna(subset=["Date","Price"]) \
           .drop_duplicates() \
           .sort_values("Date") \
           .reset_index(drop=True)

    if out.empty:
        raise RuntimeError("parsed empty after header-fix and unit->accu fallback")
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
