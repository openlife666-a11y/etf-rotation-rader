import argparse, json, math
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
import requests

DEFAULT_ETFS = [
    ('588170','科创半导体','A股科技'),
    ('512220','景顺TMT','A股科技'),
    ('563360','A500','A股宽基'),
    ('512890','红利低波','A股红利'),
    ('515050','5G ETF','A股科技'),
    ('588200','科创芯片','A股科技'),
    ('513350','标普油气','海外能源'),
    ('513290','纳指生物','海外医药'),
    ('513390','博时纳指100','海外科技'),
    ('159941','广发纳指100','海外科技'),
    ('513500','博时标普500','海外宽基'),
    ('513880','日经225','日本宽基'),
    ('560770','机器人ETF','A股科技'),
]

@dataclass
class ModelConfig:
    switch_gap: float = 0.005
    adx_strong: float = 25.0
    adx_watch: float = 20.0
    ibias_limit: float = 0.08


def eastmoney_quote_url(code):
    market = '1' if code.startswith(('5','6')) else '0'
    return f'https://push2.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code}&klt=101&fqt=1&beg=0&end=20500101&lmt=260&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'


def fetch_history(code):
    r = requests.get(eastmoney_quote_url(code), timeout=12, headers={'User-Agent':'Mozilla/5.0'})
    r.raise_for_status()
    j = r.json()
    if not j.get('data') or not j['data'].get('klines'):
        raise RuntimeError(f'No history for {code}')
    rows=[]
    for x in j['data']['klines']:
        p=x.split(',')
        rows.append([p[0],float(p[1]),float(p[2]),float(p[3]),float(p[4]),float(p[5]),float(p[6])])
    df=pd.DataFrame(rows,columns=['date','open','close','high','low','volume','amount'])
    df['date']=pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def adx(df, n=14):
    h,l,c=df.high,df.low,df.close
    up=h.diff(); down=-l.diff()
    plus_dm=np.where((up>down)&(up>0),up,0.0)
    minus_dm=np.where((down>up)&(down>0),down,0.0)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,adjust=False).mean()
    pdi=100*pd.Series(plus_dm,index=df.index).ewm(alpha=1/n,adjust=False).mean()/atr
    mdi=100*pd.Series(minus_dm,index=df.index).ewm(alpha=1/n,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean()


def metrics(df):
    c=df.close
    m21=c.iloc[-1]/c.iloc[-22]-1
    m63=c.iloc[-1]/c.iloc[-64]-1
    m1=m21
    ier=abs(c.iloc[-1]-c.iloc[-22])/c.diff().abs().iloc[-21:].sum() if c.diff().abs().iloc[-21:].sum() else 0
    score21=m21+0.05*ier
    a=adx(df).iloc[-1]
    ma20=c.rolling(20).mean().iloc[-1]
    ma60=c.rolling(60).mean().iloc[-1]
    ma200=c.rolling(200).mean().iloc[-1]
    ibias=c.iloc[-1]/ma20-1
    return dict(m1=m1,m3=m63,ier=ier,score21=score21,adx=a,ibias=ibias,ma60=ma60,ma200=ma200,close=c.iloc[-1],date=str(df.date.iloc[-1].date()))


def run(universe=None, current=None, cfg=ModelConfig()):
    universe = universe or DEFAULT_ETFS
    out=[]
    for code,name,group in universe:
        try:
            df=fetch_history(code)
            m=metrics(df)
            m.update(code=code,name=name,group=group)
            m['mid_ok']=m['m3']>0 and m['m1']>0
            m['candidate']=m['mid_ok'] and m['score21']>0 and m['adx']>=cfg.adx_watch
            m['ma60_ok']=m['close']>=m['ma60']
            m['risk200']=m['close']<m['ma200']
            m['overextended']=m['ibias']>cfg.ibias_limit
            out.append(m)
        except Exception as e:
            out.append({'code':code,'name':name,'group':group,'error':str(e)})
    good=[x for x in out if 'error' not in x]
    good.sort(key=lambda x:x['score21'], reverse=True)
    for i,x in enumerate(good,1): x['rank']=i
    result={'asof':good[0]['date'] if good else None,'rows':good,'errors':[x for x in out if 'error' in x]}
    if not good: return result
    top=good[0]
    if all(x['score21']<0 for x in good):
        action='退出轮动池'
    elif current:
        cur=next((x for x in good if x['code']==current),None)
        lead=top['score21']-(cur['score21'] if cur else -999)
        if top['code']==current:
            action='继续持有'
        elif top['candidate'] and lead>=cfg.switch_gap:
            action='换入 '+top['name']
        else:
            action='继续持有/观察'
        result['current']=cur
        result['lead']=lead
    else:
        action='候选 '+top['name'] if top['candidate'] else '继续观察'
    if top['overextended']:
        result['execution_warning']='第一名短期偏离MA20超过阈值，谨慎追高'
    if top['risk200']:
        result['risk_warning']='第一名跌破MA200，仅作大周期风险警示，不作为硬性淘汰'
    result['action']=action
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--current'); ap.add_argument('--gap',type=float,default=.005); ap.add_argument('--ibias',type=float,default=.08)
    a=ap.parse_args(); r=run(current=a.current,cfg=ModelConfig(switch_gap=a.gap,ibias_limit=a.ibias)); print(json.dumps(r,ensure_ascii=False,indent=2))
