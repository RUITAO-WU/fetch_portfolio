import io
import requests
import pandas as pd
import re
import html

API_TMPL = ("https://fund.eastmoney.com/f10/F10DataApi.aspx"
            "?type=lsjz&code={code}&page=1&per=5000&sdate=2000-01-01&edate=2099-12-31")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

def _extract_table_html(js_text: str) -> str:
    m = re.search(r'content\s*[:=]\s*"(.+?)"\s*,\s*(records|pages|curpage)', js_text, re.S)
    if not m:
        # 少数场景返回已是纯 HTML
        t2 = re.search(r"<table[\s\S]*?</table>", js_text, re.I)
        if not t2:
            raise RuntimeError("未在响应中找到 content 或 <table> 片段")
        return t2.group(0)
    raw = m.group(1)
    # 还原转义
    s = raw.replace(r'\"', '"').replace(r"\/", "/")
    s = s.replace(r"\n", "").replace(r"\r", "").replace(r"\t", "")
    s = s.encode("utf-8").decode("unicode_escape", errors="ignore")
    s = html.unescape(s)
    t = re.search(r"<table[\s\S]*?</table>", s, re.I)
    if not t:
        raise RuntimeError("content 中未找到 <table>")
    return t.group(0)

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
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    table_html = _extract_table_html(r.text)

    # 用 StringIO 传入，避免未来版本警告
    tables = pd.read_html(io.StringIO(table_html), flavor="lxml")
    if not tables:
        raise RuntimeError("pandas 未能读取到表格")

    df = tables[0].copy()

    # 有些返回首行不是表头，尝试把第一行设为表头
    if not any(("日" in str(c) or "期" in str(c)) for c in df.columns):
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

    cols = list(df.columns)

    # 选择日期列：优先含“净值日期/日期”的列；兜底取第1列
    date_col = _pick_column(cols, ["净值日期", "日期"], default_idx=0)
    if date_col is None:
        raise RuntimeError(f"未找到日期列，实际表头：{cols}")

    # 选择净值列：常见“单位净值/单位净值(元)/净值”；兜底取第2列
    nav_col = _pick_column(cols, ["单位净值", "单位净值(元)", "单位", "净值"], default_idx=1)
    if nav_col is None:
        raise RuntimeError(f"未找到净值列，实际表头：{cols}")

    out = df[[date_col, nav_col]].rename(columns={date_col: "Date", nav_col: "Price"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out = out.dropna(subset=["Date", "Price"]).sort_values("Date")
    if out.empty:
        raise RuntimeError("解析后数据为空，请检查接口返回")
    return out
