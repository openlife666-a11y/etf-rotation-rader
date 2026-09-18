import streamlit as st
from rotation_model import run, DEFAULT_ETFS, ModelConfig

st.set_page_config(page_title='ETF行业轮动 V1.0', layout='wide')
st.title('ETF 行业轮动模型 V1.0')
st.caption('3个月70% + 1个月30%中期过滤 → 21日动量 + 5% IER → ADX → IBIAS → MA60 → MA200风险警示 → 0.50个百分点换仓门槛')

codes={c:n for c,n,g in DEFAULT_ETFS}
current=st.selectbox('当前持有（可选）',['不指定']+list(codes.keys()), format_func=lambda x:'不指定' if x=='不指定' else f'{x} {codes[x]}')
gap=st.slider('换仓优势门槛',0.0,0.02,0.005,0.001,format='%.3f')
ibias=st.slider('IBIAS追高阈值',0.03,0.15,0.08,0.01,format='%.2f')
if st.button('运行模型', type='primary'):
    r=run(current=None if current=='不指定' else current,cfg=ModelConfig(switch_gap=gap,ibias_limit=ibias))
    st.success(f"模型日期：{r.get('asof')} | 最终信号：{r.get('action')}")
    if r.get('execution_warning'): st.warning(r['execution_warning'])
    if r.get('risk_warning'): st.warning(r['risk_warning'])
    rows=r['rows']
    cols=['rank','code','name','group','m3','m1','m21','ier','score21','adx','ibias','ma60_ok','risk200','candidate']
    df=__import__('pandas').DataFrame(rows)[cols]
    for c in ['m3','m1','m21','ier','score21','ibias']:
        df[c]=(df[c]*100).round(2).astype(str)+'%'
    df['adx']=df['adx'].round(1)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if r.get('current'):
        st.info(f"当前持有评分：{r['current']['score21']*100:.2f}% | 第一名领先：{r['lead']*100:.2f}个百分点")
else:
    st.info('点击“运行模型”后，工具会抓取最近交易数据并输出排名与动作信号。')

st.markdown('### 当前池（已核验代码）')
st.write('、'.join(f'{c} {n}' for c,n,g in DEFAULT_ETFS))
st.caption('注：QDII ETF 的场内溢价/折价是独立执行层，本版先不把溢价硬编码成淘汰条件，后续接入实时IOPV/溢价监控。')
