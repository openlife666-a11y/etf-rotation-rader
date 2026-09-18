import json
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="ETF轮动雷达", page_icon="📡", layout="wide")

# 固定轮动池
ETF_POOL = {
    "588170": ("科创半导体ETF", "SH", False),
    "512220": ("TMTETF景顺", "SH", False),
    "563360": ("A500ETF华泰柏瑞", "SH", False),
    "512890": ("红利低波ETF华泰柏瑞", "SH", False),
    "515050": ("5GETF", "SH", False),
    "588200": ("科创芯片ETF嘉实", "SH", False),
    "513350": ("标普油气ETF富国", "SH", True),
    "513290": ("纳指生物科技ETF汇添富", "SH", True),
    "513390": ("纳指100ETF博时", "SH", True),
    "513500": ("标普500ETF博时", "SH", True),
    "513880": ("日经225ETF华安", "SH", True),
    "560770": ("机器人ETF招商", "SH", False),
    "560860": ("工业有色ETF万家", "SH", False),
    "512480": ("半导体ETF国联安", "SH", False),
    "515790": ("光伏ETF华泰柏瑞", "SH", False),
    "159869": ("游戏ETF华夏", "SZ", False),
}

SWITCH_GAP = 0.005
IER_WEIGHT = 0.05
LOOKBACK_21 = 21
LOOKBACK_1M = 21
LOOKBACK_3M = 63
MIN_ROWS = 210


def make_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Safari/605.1",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://finance.qq.com/",
        "Connection": "keep-alive",
    })
    return s


@st.cache_resource
def get_session():
    return make_session()


SESSION = get_session()


def em_secid(code):
    # 东方财富：沪市=1，深市=0
    market = "0" if ETF_POOL[code][1] == "SZ" else "1"
    return f"{market}.{code}"


def tq_code(code):
    market = "sz" if ETF_POOL[code][1] == "SZ" else "sh"
    return f"{market}{code}"


def normalize_rows(rows):
    if not rows:
        return None
    out = []
    for x in rows:
        if isinstance(x, str):
            parts = x.split(",")
        else:
            parts = list(x)
        if len(parts) < 5:
            continue
        try:
            # 腾讯：date, open, close, high, low, volume
            out.append([
                pd.to_datetime(parts[0]),
                float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),
                float(parts[5]) if len(parts) > 5 else np.nan,
            ])
        except Exception:
            continue
    if not out:
        return None
    df = pd.DataFrame(out, columns=["date", "open", "close", "high", "low", "volume"])
    return df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_eastmoney(code):
    hosts = [
        "push2his.eastmoney.com",
        "7.push2his.eastmoney.com",
        "33.push2his.eastmoney.com",
        "63.push2his.eastmoney.com",
        "91.push2his.eastmoney.com",
    ]
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "1",
        "secid": em_secid(code),
        "beg": "20240101",
        "end": "20500101",
        "lmt": "500",
    }
    last = ""
    for host in hosts:
        try:
            r = SESSION.get(f"https://{host}/api/qt/stock/kline/get", params=params, timeout=12)
            payload = r.json()
            data = payload.get("data") or {}
            rows = data.get("klines") or []
            if not rows:
                last = f"{host}: 空数据"
                continue
            parsed = []
            for row in rows:
                p = row.split(",")
                if len(p) >= 6:
                    parsed.append([p[0], p[1], p[2], p[3], p[4], p[5]])
            df = normalize_rows(parsed)
            if df is not None and len(df) >= MIN_ROWS:
                return df, f"东方财富/{host}"
            last = f"{host}: 数据不足"
        except Exception as e:
            last = f"{host}: {type(e).__name__}: {e}"
        time.sleep(0.25)
    return None, last


def fetch_tencent(code):
    tq = tq_code(code)
    # 腾讯公开接口，返回 JSONP。先尝试前复权，再尝试不复权。
    attempts = ["qfq", "none"]
    for fq in attempts:
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {
            "_var": "kline_dayqfq",
            "param": f"{tq},day,2024-01-01,2050-01-01,500,{fq}",
        }
        try:
            r = SESSION.get(url, params=params, timeout=12)
            text = r.text.strip()
            if "=" in text:
                text = text.split("=", 1)[1].rstrip(";")
            obj = json.loads(text)
            block = (obj.get("data") or {}).get(tq) or {}
            rows = block.get("qfqday") or block.get("day") or block.get("qfqday")
            df = normalize_rows(rows)
            if df is not None and len(df) >= MIN_ROWS:
                return df, f"腾讯/qfq" if fq == "qfq" else "腾讯/raw"
        except Exception:
            pass
    return None, "腾讯接口无有效K线"


def fetch_sina(code):
    symbol = tq_code(code)
    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {"symbol": symbol, "scale": 240, "ma": "no", "datalen": 500}
    try:
        r = SESSION.get(url, timeout=12, headers={"Referer": "https://finance.sina.com.cn/"})
        # 这个接口有时忽略 query 参数，第二次明确拼接参数。
        if r.status_code != 200 or not r.text.strip():
            r = SESSION.get(url, params=params, timeout=12, headers={"Referer": "https://finance.sina.com.cn/"})
        else:
            try:
                json.loads(r.text)
            except Exception:
                r = SESSION.get(url, params=params, timeout=12, headers={"Referer": "https://finance.sina.com.cn/"})
        data = r.json()
        rows = []
        for x in data or []:
            rows.append([x.get("day"), x.get("open"), x.get("close"), x.get("high"), x.get("low"), x.get("volume")])
        df = normalize_rows(rows)
        if df is not None and len(df) >= MIN_ROWS:
            return df, "新浪"
    except Exception:
        pass
    return None, "新浪接口无有效K线"


def fetch_kline(code):
    # 关键：三个完全不同的公开数据源，而不是只更换同一家公司的域名。
    errors = []
    for fn in (fetch_eastmoney, fetch_tencent, fetch_sina):
        df, source = fn(code)
        if df is not None:
            return df, source
        errors.append(source)
    return None, "；".join(errors)


def calc_adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    up, down = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    mdi = 100 * minus.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    return float(dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean().iloc[-1])


def efficiency_ratio(close, n=21):
    if len(close) <= n:
        return np.nan
    net = abs(float(close.iloc[-1] - close.iloc[-1-n]))
    path = float(close.diff().abs().iloc[-n:].sum())
    return net / path if path else 0.0


def calc_metrics(df):
    c = df["close"]
    now = float(c.iloc[-1])
    m21 = now / float(c.iloc[-1-LOOKBACK_21]) - 1
    m1 = now / float(c.iloc[-1-LOOKBACK_1M]) - 1
    m3 = now / float(c.iloc[-1-LOOKBACK_3M]) - 1
    ier = efficiency_ratio(c, LOOKBACK_21)
    score = m21 + IER_WEIGHT * ier
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma60 = float(c.rolling(60).mean().iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    return {
        "最新价": now,
        "3个月动量": m3,
        "1个月动量": m1,
        "21日动量": m21,
        "IER": ier,
        "Score": score,
        "ADX14": calc_adx(df),
        "IBIAS20": now / ma20 - 1,
        "MA60": ma60,
        "MA200": ma200,
        "收盘日": df["date"].iloc[-1].date(),
        "MA60趋势": "上方" if now >= ma60 else "下方",
        "MA200趋势": "上方" if now >= ma200 else "下方",
    }


def load_all(progress):
    results, errors = {}, {}
    items = list(ETF_POOL.items())
    for i, (code, info) in enumerate(items, 1):
        progress.progress(i / len(items), text=f"正在获取 {code} {info[0]}  ({i}/{len(items)})")
        df, source = fetch_kline(code)
        if df is None:
            errors[code] = source
        else:
            try:
                results[code] = calc_metrics(df)
                results[code]["数据源"] = source
            except Exception as e:
                errors[code] = f"计算失败：{e}"
        time.sleep(0.10)
    return results, errors


st.title("📡 ETF 轮动雷达")
st.caption("日K收盘模型｜多数据源自动切换｜Score = 21日动量 + 5% × IER｜切换阈值 0.50 个百分点")

with st.sidebar:
    st.subheader("当前持仓")
    options = ["未持有"] + [f"{c} {n}" for c, (n, _, _) in ETF_POOL.items()]
    current_label = st.selectbox("选择当前持仓", options)
    current_code = current_label.split()[0] if current_label != "未持有" else None
    st.divider()
    st.write("新标的 Score 领先当前持仓 ≥ 0.50 个百分点才换入。")
    st.write("全部 Score < 0 时退出轮动池。")
    st.write("MA200 只作风险预警，不作硬门槛。")

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.errors = {}
    st.session_state.updated_at = None

c1, c2 = st.columns([1, 4])
with c1:
    run = st.button("🔄 获取最新数据", type="primary", use_container_width=True)
with c2:
    if st.session_state.updated_at:
        st.write(f"上次刷新：{st.session_state.updated_at}")

if run:
    bar = st.empty()
    with st.spinner("正在依次尝试东方财富 → 腾讯 → 新浪，请稍候…"):
        results, errors = load_all(bar)
    st.session_state.results = results
    st.session_state.errors = errors
    st.session_state.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar.empty()
    st.rerun()

results = st.session_state.results
errors = st.session_state.errors

if results is None:
    st.info("点击“🔄 获取最新数据”开始抓取。")
    st.stop()

rows = []
for code, m in results.items():
    name = ETF_POOL[code][0]
    rows.append({"代码": code, "名称": name, **m, "QDII": "是" if ETF_POOL[code][2] else "否"})

dfout = pd.DataFrame(rows)
if not dfout.empty:
    dfout = dfout.sort_values("Score", ascending=False).reset_index(drop=True)

st.subheader("🎯 当前轮动结论")
if dfout.empty:
    st.error("本次没有任何品种成功取得数据。请稍后再次点击获取。")
else:
    top = dfout.iloc[0]
    all_negative = bool((dfout["Score"] < 0).all())
    current_score = None
    if current_code and current_code in set(dfout["代码"]):
        current_score = float(dfout.loc[dfout["代码"] == current_code, "Score"].iloc[0])
    lead = None if current_score is None else float(top["Score"] - current_score)

    if all_negative:
        action = "退出轮动池"
        detail = "所有成功取数品种的 Score 均 < 0。"
    elif current_code is None:
        action = f"候选：{top['代码']} {top['名称']}"
        detail = "尚未设置当前持仓。"
    elif current_code == top["代码"]:
        action = "持有"
        detail = f"当前持仓为第一名，Score={top['Score']:.2%}。"
    elif current_score is None:
        action = "观察"
        detail = "当前持仓本次取数失败，无法计算可靠的切换差值。"
    elif lead >= SWITCH_GAP:
        action = f"换入 {top['代码']} {top['名称']}"
        detail = f"领先当前持仓 {lead:.2%}，达到 0.50 个百分点阈值。"
    else:
        action = "继续持有 / 观察"
        detail = f"领先当前持仓 {lead:.2%}，未达到 0.50 个百分点阈值。"

    st.metric("模型动作", action)
    st.write(detail)
    if top["QDII"] == "是":
        st.warning("⚠️ 第一名是QDII。模型信号与实际执行分开，下单前请另外核对场内价格、IOPV及溢价/折价。")

st.subheader("📊 全部评分")
show_cols = ["代码", "名称", "3个月动量", "1个月动量", "21日动量", "IER", "Score", "ADX14", "IBIAS20", "MA60趋势", "MA200趋势", "收盘日", "数据源"]
if not dfout.empty:
    fmt = dfout[show_cols].copy()
    for c in ["3个月动量", "1个月动量", "21日动量", "Score", "IBIAS20"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    fmt["IER"] = fmt["IER"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    fmt["ADX14"] = fmt["ADX14"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    st.dataframe(fmt, use_container_width=True, hide_index=True)

if errors:
    st.subheader("⚠️ 数据状态")
    err_rows = [{"代码": c, "名称": ETF_POOL[c][0], "状态": "获取失败", "原因": e} for c, e in errors.items()]
    st.dataframe(pd.DataFrame(err_rows), use_container_width=True, hide_index=True)
    st.caption("程序会依次尝试三套公开行情源。单个ETF失败不会影响其他ETF。再次点击“获取最新数据”即可重试。")

st.subheader("🧭 模型说明")
st.markdown("""
- **Score = 21日动量 + 5% × IER**。
- **IER** = 21日净位移 ÷ 21日实际波动路径。
- 3个月70% + 1个月30%用于中期强弱背景，本版本不改变固定核心Score。
- **ADX14**看趋势强度；**IBIAS20 = 收盘价 / MA20 - 1**，用于观察短线偏离。
- **MA60**看中期趋势；**MA200**只作风险预警。
- 新标的必须比当前持仓高 **0.50 个百分点**才触发换入。
""")

st.caption("数据源采用东方财富、腾讯财经、新浪财经公开行情接口的自动兜底。数据可能存在延迟或接口临时不可用，实际交易请以交易所/券商行情为准。")
