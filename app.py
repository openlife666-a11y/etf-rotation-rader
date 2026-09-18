import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import akshare as ak

st.set_page_config(page_title='ETF轮动雷达', page_icon='📡', layout='wide')

ETF_POOL = {
    '588170':'科创半导体', '512220':'TMTETF景顺', '563360':'A500ETF',
    '512890':'红利低波', '515050':'5GETF', '588200':'科创芯片',
    '513350':'标普油气', '513290':'纳指生物科技', '513390':'纳指100',
    '159941':'广发纳指100', '513500':'标普500', '513880':'日经225',
    '560770':'机器人ETF', '512400':'有色ETF', '512560':'军工ETF'
}

@st.cache_data(ttl=300, show_spinner=False)
def get_hist(code):
    df = ak.fund_etf_hist_em(symbol=code, period='daily', start_date='20240101', end_date=datetime.now().strftime('%Y%m%d'), adjust='')
    if df is None or df.empty:
        raise ValueError('无历史数据')
    df = df.rename(columns={'日期':'date','收盘':'close','成交量':'volume','开盘':'open','最高':'high','最低':'low'})
    df['date'] = pd.to_datetime(df['date'])
    for c in ['close','open','high','low','volume']:
        if c in df: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.sort_values('date').dropna(subset=['close']).reset_index(drop=True)

def adx(df, n=14):
    h,l,c=df['high'].astype(float),df['low'].astype(float),df['close'].astype(float)
    up=h.diff(); dn=-l.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,adjust=False).mean()
    pdi=100*pd.Series(plus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/atr
    mdi=100*pd.Series(minus,index=df.index).ewm(alpha=1/n,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean()

def calc(df):
    x=df.copy(); c=x.close
    x['m21']=c/c.shift(21)-1
    x['m63']=c/c.shift(63)-1
    x['ier']=abs(c-c.shift(21))/(c.diff().abs().rolling(21).sum())
    x['score21']=x['m21']+0.05*x['ier']
    x['mid_score']=0.7*x['m63']+0.3*x['m21']
    x['ma20']=c.rolling(20).mean(); x['ma60']=c.rolling(60).mean(); x['ma200']=c.rolling(200).mean()
    x['ibias']=c/x['ma20']-1
    x['adx']=adx(x)
    r=x.iloc[-1]
    return {'m21':r.m21,'m63':r.m63,'ier':r.ier,'score21':r.score21,'mid_score':r.mid_score,'ma60':r.ma60,'ma200':r.ma200,'close':r.close,'ibias':r.ibias,'adx':r.adx,'date':r.date}

st.title('📡 ETF 轮动雷达')
st.caption('V1.1：3个月70% + 1个月30%中期过滤；21日动量 + 5% IER；ADX / IBIAS / MA60 / MA200；换仓门槛 0.50 个百分点。')

codes=list(ETF_POOL.keys())
with st.sidebar:
    st.header('参数')
    current=st.selectbox('当前持仓', [f'{c}  {ETF_POOL[c]}' for c in codes], index=codes.index('513390'))
    current_code=current.split()[0]
    switch_gap=st.number_input('换仓领先门槛（百分点）', min_value=0.0, max_value=5.0, value=0.50, step=0.05)
    adx_good=st.number_input('ADX强趋势阈值', min_value=10.0, max_value=50.0, value=25.0, step=1.0)
    ibias_hot=st.number_input('IBIAS过热阈值', min_value=0.01, max_value=0.30, value=0.08, step=0.01, format='%.2f')
    run=st.button('🚀 运行最新数据', use_container_width=True, type='primary')

if run or 'results' not in st.session_state:
    results=[]; errors=[]
    with st.spinner('正在抓取 ETF 历史行情并计算指标…'):
        for code,name in ETF_POOL.items():
            try:
                if code in ['159941']:
                    # 同样走 EM ETF 历史接口
                    pass
                d=calc(get_hist(code))
                results.append({'代码':code,'名称':name,**d})
            except Exception as e:
                errors.append((code,name,str(e)))
    st.session_state.results=results
    st.session_state.errors=errors

res=pd.DataFrame(st.session_state.results)
if res.empty:
    st.error('暂时没有取得行情数据。请稍后再点“运行最新数据”。')
    st.stop()

res['中期过滤']= (res.m63>0) & (res.m21>0)
res['MA60趋势']=np.where(res.close>=res.ma60,'向上','向下')
res['MA200警报']=np.where(res.close>=res.ma200,'正常','风险')
res['过热']=res.ibias>=ibias_hot
res['候选']=res.中期过滤 & (res.score21>0)
res=res.sort_values(['候选','score21','mid_score'], ascending=[False,False,False]).reset_index(drop=True)

cur=res[res['代码']==current_code]
best=res.iloc[0]
lead=(best.score21-cur.score21)*100 if not cur.empty else np.nan
all_negative=(res.score21<0).all()

if all_negative:
    action='退出轮动池'; color='red'; reason='三个候选的21日 Score 全部低于0。'
elif best['代码']==current_code:
    action='继续持有'; color='green'; reason='当前持仓仍位于模型第一位。'
elif (not best['候选']) or best['过热']:
    action='继续观察'; color='orange'; reason='第一名未通过中期趋势/过热过滤。'
elif lead < switch_gap:
    action='继续持有'; color='orange'; reason=f'第一名领先当前持仓仅 {lead:.2f} 个百分点，未达到 {switch_gap:.2f} 的换仓门槛。'
else:
    action=f'换入 {best.代码} {best.名称}'; color='blue'; reason=f'第一名领先当前持仓 {lead:.2f} 个百分点，达到换仓门槛。'

c1,c2,c3,c4=st.columns(4)
c1.metric('当前持仓',f'{current_code} {ETF_POOL[current_code]}')
c2.metric('模型第一名',f"{best.代码} {best.名称}")
c3.metric('Score21',f"{best.score21*100:.2f}%")
c4.metric('领先当前持仓', '—' if np.isnan(lead) else f'{lead:.2f} 个百分点')

if color=='green': st.success(f'### 🟢 {action}\n\n{reason}')
elif color=='red': st.error(f'### 🔴 {action}\n\n{reason}')
elif color=='orange': st.warning(f'### 🟡 {action}\n\n{reason}')
else: st.info(f'### 🔵 {action}\n\n{reason}')

st.subheader('模型排名')
display=res[['代码','名称','mid_score','m21','ier','score21','adx','ibias','MA60趋势','MA200警报','候选']].copy()
for col in ['mid_score','m21','score21','ibias']:
    display[col]=(display[col]*100).round(2).astype(str)+'%'
display['ier']=display['ier'].round(2); display['adx']=display['adx'].round(1)
st.dataframe(display, use_container_width=True, hide_index=True)

if st.session_state.errors:
    with st.expander(f'数据异常 {len(st.session_state.errors)} 个'):
        st.write(pd.DataFrame(st.session_state.errors,columns=['代码','名称','错误']))

st.divider()
st.caption('说明：本版本的 IBIAS 定义为 收盘价 / MA20 - 1；MA200 仅作为风险警报。QDII 场内溢价/IOPV 尚未作为自动交易闸门，因此涉及海外 ETF 时应额外核对场内溢价。模型仅用于规则化分析，不构成投资建议。')
