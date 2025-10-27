import re
import json
import os
import datetime as dt
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SITE_DIR = ROOT / "site"
DATA_DIR.mkdir(exist_ok=True)
SITE_DIR.mkdir(exist_ok=True)

API_TMPL = ("https://fund.eastmoney.com/f10/F10DataApi.aspx"
            "?type=lsjz&code={code}&page=1&per=5000&sdate=2000-01-01&edate=2099-12-31")

def fetch_one(code: str) -> pd.DataFrame:
    url = API_TMPL.format(code=code)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    text = r.text

    # 提取 apidata.content 中的 HTML 表格片段
    m = re.search(r"var\s+apidata\s*=\s*\{.*?content:\"(.*)\"\s*,\s*records", text, re.S)
    if not m:
        raise RuntimeError(f"无法解析接口返回（{code}）")
    # 还原转义
    html = m.group(1).encode('utf-8').decode('unicode_escape')

    # 读表（历史净值表头通常含：净值日期、单位净值、累计净值、日增长率、申购状态、赎回状态）
    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError(f"未找到表格（{code}）")
    df = tables[0].copy()

    # 兼容不同表头：取“净值日期”“单位净值”
    date_col = [c for c in df.columns if "日期" in str(c)][0]
    nav_col = None
    for key in ["单位净值", "单位", "净值"]:
        cand = [c for c in df.columns if key in str(c)]
        if cand:
            nav_col = cand[0]
            break
    if nav_col is None:
        raise RuntimeError(f"未找到单位净值列（{code}）")

    out = df[[date_col, nav_col]].rename(columns={date_col: "Date", nav_col: "Price"})

    # 规范化
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out = out.dropna().sort_values("Date")

    return out

def write_csv(df: pd.DataFrame, code: str):
    p = DATA_DIR / f"{code}.csv"
    df.to_csv(p, index=False)  # 头部：Date,Price
    return p

def write_html_table(df: pd.DataFrame, code: str):
    # 生成一个简单的 HTML 页面，首张表格为 PP 可识别表
    title = f"Fund {code} (Eastmoney) - Historical NAV"
    table_html = df.to_html(index=False)  # 表头为 Date / Price
    meta = (f"<!-- Generated: {dt.datetime.utcnow().isoformat()}Z -->")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
{meta}
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; padding: 16px; }}
table {{ border-collapse: collapse; width: 100%; max-width: 860px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
caption {{ text-align:left; font-weight:600; margin-bottom:8px; }}
.note {{ margin-top:12px; color:#555; font-size:0.9em; }}
</style>
</head>
<body>
<h1>{title}</h1>
{table_html}
<div class="note">
Columns: <strong>Date</strong> (yyyy-MM-dd), <strong>Price</strong> (CNY).<br/>
Source: Eastmoney F10DataApi.
</div>
</body>
</html>"""
    p = SITE_DIR / f"{code}.html"
    p.write_text(html, encoding="utf-8")
    return p

def main():
    conf = json.loads((DATA_DIR / "funds.json").read_text(encoding="utf-8"))
    for code in conf.get("codes", []):
        df = fetch_one(code)
        write_csv(df, code)
        write_html_table(df, code)

if __name__ == "__main__":
    main()