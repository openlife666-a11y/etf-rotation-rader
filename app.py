import time
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title='ETF轮动雷达', page_icon='📡', layout='wide')

ETF_POOL = {
    '588170': ('科创半导体ETF', '上海', False),
    '512220': ('TMTETF景顺', '上海', False),
    '563360': ('A500ETF华泰柏瑞', '上海', False),
    '512890': ('红利低波ETF华泰柏瑞', '上海', False),
    '515050': ('5GETF', '上海', False),
    '588200': ('科创芯片ETF嘉实', '上海', False),
    '513350': ('标普油气ETF富国', '上海', True),
    '513290': ('纳指生物科技ETF汇添富', '上海', True),
    '513390': ('纳指100ETF博时', '上海', True),
    '513500': ('标普500ETF博时', '上海', True),
    '513880': ('日经225ETF华安', '上海', True),
    '560770': ('机器人ETF招商', '上海', False),
    '560860': ('工业有色ETF万家', '上海', False),
    '512480': ('半导体ETF国联安', '上海', False),
    '515790': ('光伏ETF华泰柏瑞', '上海', False),
    '159869': ('游戏ETF华夏', '深圳', False),
}

# Model constants
SWITCH_GAP = 0.005       # 0.50 percentage point
IER_WEIGHT = 0.05
LOOKBACK_21 = 21
LOOKBACK_1M = 21
LOOKBACK_3M = 63
MIN_ROWS = 210

@st.cache_resource
def make_session():
    s = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(['GET']),
        raise_on_status=False,
    )
    s.mount('https://', HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Safari/605.1',
        'Referer': 'https://quote.eastmoney.com/',
        'Accept': 'application/json,text/plain,*/*',
        'Connection': 'keep-alive',
    })
    return s

SESSION = make_session()


def secid(code):
    market = '0' if code.startswith('159') else '1'
    return f'{market}.{code}'


def fetch_kline(code, retries=3):
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': secid(code),
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',
        'fqt': '1',
        'beg': '20240101',
        'end': '20500101',
        'lmt': '420',
    }
    last = ''
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            r.raise_for_status()
            payload = r.json()
            lines = (payload.get('data') or {}).get('klines') or []
            if not lines:
                raise ValueError('接口返回空数据')
            rows = [x.split(',') for x in lines]
            cols = ['date','open','close','high','low','volume','amount','amplitude','pct','change','turnover']
            df = pd.DataFrame(rows, columns=cols[:len(rows[0])])
            for c in cols[1:]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date','close']).sort_values('date').reset_index(drop=True)
            if len(df) < MIN_ROWS:
                raise ValueError(f'历史数据仅 {len(df)} 条，少于 {MIN_ROWS} 条')
            return df, 'OK'
        except Exception as e:
            last = str(e)
            time.sleep(0.8 * (attempt + 1))
    return None, last[:120]


def calc_adx(df, n=14):
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    return float(dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean().iloc[-1])


def efficiency_ratio(close, n=21):
    s = close.dropna()
    if len(s) <= n:
        return np.nan
    net = abs(float(s.iloc[-1] - s.iloc[-1-n]))
    path = float(s.diff().abs().iloc[-n:].sum())
    return net / path if path else 0.0


def metrics(df):
    c = df['close']
    now = float(c.iloc[-1])
    m21 = now / float(c.iloc[-1-LOOKBACK_21]) - 1
    m1 = now / float(c.iloc[-1-LOOKBACK_1M]) - 1
    m3 = now / float(c.iloc[-1-LOOKBACK_3M]) - 1
    ier = efficiency_ratio(c, LOOKBACK_21)
    score = m21 + IER_WEIGHT * ier
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma60 = float(c.rolling(60).mean().iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    ibias = now / ma20 - 1
    adx = calc_adx(df)
    return {
        '最新价': now, '3个月动量': m3, '1个月动量': m1, '21日动量': m21,
        'IER': ier, 'Score': score, 'ADX14': adx, 'IBIAS20': ibias,
        'MA60': ma60, 'MA200': ma200, '收盘日': df['date'].iloc[-1].date(),
        'MA60趋势': '上方' if now >= ma60 else '下方',
        'MA200趋势': '上方' if now >= ma200 else '下方',
    }


def load_all(progress=None):
    results, errors = {}, {}
    items = list(ETF_POOL.items())
    for i, (code, info) in enumerate(items, 1):
        if progress:
            progress.progress(i/len(items), text=f'正在获取 {code} {info[0]}  ({i}/{len(items)})')
        df, status = fetch_kline(code)
        if df is None:
            errors[code] = status
        else:
            try:
                results[code] = metrics(df)
            except Exception as e:
                errors[code] = f'计算失败: {e}'
        time.sleep(0.15)
    return results, errors

st.title('📡 ETF 轮动雷达 · 最终版')
st.caption('日K收盘模型 | 3个月70% + 1个月30%用于中期过滤；21日动量 + 5%×IER为核心 Score；ADX / IBIAS / MA60 / MA200用于确认与风险提示')

with st.sidebar:
    st.subheader('当前持仓')
    options = ['未持有'] + [f'{c} {n}' for c,(n,_,_) in ETF_POOL.items()]
    current_label = st.selectbox('选择当前持仓', options, index=0)
    current_code = current_label.split()[0] if current_label != '未持有' else None
    st.divider()
    st.markdown('**切换规则**')
    st.write('新标的 Score 必须领先当前持仓 ≥ 0.50 个百分点才触发换入。')
    st.write('全部 Score < 0：退出轮动池，不强行选一个。')
    st.write('MA200 只做风险预警，不作为硬门槛。')
    st.divider()
    st.info('数据源：东方财富公开 K 线接口。程序内置重试、退避、单品种隔离。')

if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.errors = {}
    st.session_state.updated_at = None

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button('🔄 获取最新数据', type='primary', use_container_width=True)
with col2:
    if st.session_state.updated_at:
        st.write(f"上次刷新：{st.session_state.updated_at}")

if run:
    bar = st.empty()
    with st.spinner('正在连接行情源并计算指标，请稍候…'):
        results, errors = load_all(bar)
    st.session_state.results = results
    st.session_state.errors = errors
    st.session_state.updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    bar.empty()
    st.rerun()

results = st.session_state.results
errors = st.session_state.errors

if results is None:
    st.warning('第一次使用请点击“🔄 获取最新数据”。')
    st.stop()

rows = []
for code, m in results.items():
    name = ETF_POOL[code][0]
    rows.append({'代码': code, '名称': name, **m, 'QDII': '是' if ETF_POOL[code][2] else '否'})

dfout = pd.DataFrame(rows).sort_values('Score', ascending=False).reset_index(drop=True)

st.subheader('🎯 当前轮动结论')
if len(dfout) == 0:
    st.error('本次没有任何品种成功取得数据。不要依据本次结果交易，请稍后重试。')
else:
    top = dfout.iloc[0]
    all_negative = bool((dfout['Score'] < 0).all())
    current_score = float(dfout.loc[dfout['代码'] == current_code, 'Score'].iloc[0]) if current_code in set(dfout['代码']) else None
    lead = None if current_score is None else float(top['Score'] - current_score)

    if all_negative:
        action = '退出轮动池'
        detail = '所有已成功取数的品种 Score 均 < 0。'
    elif current_code is None:
        action = f"观察/候选：{top['代码']} {top['名称']}"
        detail = '尚未设置当前持仓。先看候选，不自动假设你已持有。'
    elif current_code == top['代码']:
        action = '持有'
        detail = f"当前持仓就是 Score 第一名，Score={top['Score']:.2%}。"
    elif current_score is None:
        action = '观察'
        detail = '当前持仓本次取数失败，无法计算可靠的切换差值。'
    elif lead >= SWITCH_GAP:
        action = f"换入 {top['代码']} {top['名称']}"
        detail = f"新标的领先当前持仓 {lead:.2%}，达到 ≥0.50 个百分点的切换阈值。"
    else:
        action = '继续持有/观察'
        detail = f"第一名只领先当前持仓 {lead:.2%}，未达到 0.50 个百分点切换阈值。"

    st.metric('模型动作', action)
    st.write(detail)
    if bool(top['QDII'] == '是'):
        st.warning('⚠️ 第一名属于QDII。模型信号与实际买卖执行要分开：下单前请另外核对场内实时价格、IOPV及溢价/折价，尤其是513350等曾出现明显溢价风险提示的产品。')

st.subheader('📊 全部评分')
show_cols = ['代码','名称','3个月动量','1个月动量','21日动量','IER','Score','ADX14','IBIAS20','MA60趋势','MA200趋势','收盘日']
fmt = dfout[show_cols].copy()
for c in ['3个月动量','1个月动量','21日动量','Score','IBIAS20']:
    fmt[c] = fmt[c].map(lambda x: f'{x:.2%}' if pd.notna(x) else '-')
for c in ['IER']:
    fmt[c] = fmt[c].map(lambda x: f'{x:.3f}' if pd.notna(x) else '-')
for c in ['ADX14']:
    fmt[c] = fmt[c].map(lambda x: f'{x:.1f}' if pd.notna(x) else '-')
for c in ['MA60趋势','MA200趋势']:
    fmt[c] = fmt[c].fillna('-')
st.dataframe(fmt, use_container_width=True, hide_index=True)

st.subheader('🧭 模型解释')
st.markdown('''
- **Score** = 21日动量 + 5% × IER。IER为21日趋势效率比，衡量“净位移/实际波动路径”。
- **3个月70% + 1个月30%**：用于判断中期强弱背景，不直接替代核心Score。
- **ADX14**：看趋势强度，不单独决定买卖。
- **IBIAS20**：本版本定义为 `收盘价 / MA20 - 1`，用于识别短线偏离程度。
- **MA60**：中期趋势确认；**MA200**：风险预警，不做硬门槛。
- **切换阈值**：新标的必须比当前持仓高至少 **0.50个百分点**。
''')

if errors:
    st.subheader('⚠️ 数据状态')
    err_rows = [{'代码': c, '名称': ETF_POOL[c][0], '状态': '获取失败/需重试', '原因': e} for c,e in errors.items()]
    st.dataframe(pd.DataFrame(err_rows), use_container_width=True, hide_index=True)
    st.caption('单个品种失败不会影响其他品种计算。再次点击“获取最新数据”即可重试。')

st.caption('本工具只计算机械模型信号，不自动下单。QDII场内溢价/IOPV属于执行层风险，应在实际下单前单独核对。')
