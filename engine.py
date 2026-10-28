import numpy as np
import pandas as pd

def number(x):
    return float(x) if x is not None and np.isfinite(x) else None

def path(prices, weights, rebalance):
    """Invest at first close. Rebalance at first common close of new month/year,
    after earning that session's return, affecting the next session."""
    weights=np.asarray(weights,dtype=float)
    relative=prices.pct_change(fill_method=None).iloc[1:]+1
    holdings=weights.copy(); values=[1.0]; previous=prices.index[0]
    for date, gross in relative.iterrows():
        holdings=holdings*gross.to_numpy()
        total=float(holdings.sum());values.append(total)
        reset=(rebalance=='daily' or
               rebalance=='monthly' and (date.year,date.month)!=(previous.year,previous.month) or
               rebalance=='yearly' and date.year!=previous.year)
        if reset: holdings=total*weights
        previous=date
    return pd.Series(values,index=prices.index)

def metrics(level,benchmark,inflation,amount):
    r=level.pct_change(fill_method=None).dropna(); b=benchmark.pct_change(fill_method=None).dropna()
    years=(level.index[-1]-level.index[0]).days/365.25
    performance=level.iloc[-1]-1
    std=r.std(ddof=1);variance=b.var(ddof=1)
    return {'performance':number(performance),'annualized':number(level.iloc[-1]**(1/years)-1),
      'volatility':number(std*np.sqrt(252)), 'beta':number(r.cov(b)/variance) if variance>0 else None,
      'sharpe':number(r.mean()/std*np.sqrt(252)) if std>0 else None,
      'real':number(level.iloc[-1]/(1+inflation)**years-1),
      'drawdown':number((level/level.cummax()-1).min()),
      'final_value':number(amount*level.iloc[-1]),'gain':number(amount*performance)}

def compute(common,allocations,benchmark,mode,months,start,end,rebalance,inflation,amount):
    available_start=common.index[0];available_end=common.index[-1]
    if mode=='custom':
        begin=pd.Timestamp(start);finish=pd.Timestamp(end)
        if begin>=finish: raise ValueError('La date de début doit précéder la fin.')
    elif mode=='ytd':
        begin=pd.Timestamp(year=available_end.year,month=1,day=1);finish=available_end
    elif mode=='max': begin=available_start;finish=available_end
    else: finish=available_end;begin=finish-pd.DateOffset(months=months)
    frame=common.loc[(common.index>=begin)&(common.index<=finish)]
    if len(frame)<3: raise ValueError('Moins de trois clôtures communes dans la fenêtre choisie.')
    base=frame[benchmark]/frame[benchmark].iloc[0]
    output=[];levels=[]
    for a in allocations:
        symbols=[p['ticker'] for p in a['positions']]
        level=path(frame[symbols],[p['weight']/100 for p in a['positions']],rebalance)
        levels.append(level)
        calendar=[]
        returns=level.pct_change(fill_method=None).dropna()
        br=base.pct_change(fill_method=None).dropna()
        for year,part in returns.groupby(returns.index.year):
            calendar.append({'year':int(year),'performance':number((1+part).prod()-1),'benchmark':number((1+br.loc[part.index]).prod()-1)})
        output.append({'id':a['id'],'name':a['name'],'metrics':metrics(level,base,inflation,amount),'calendar':calendar})
    curve=[{'date':d.strftime('%Y-%m-%d'),'benchmark':number(base.loc[d]-1),'portfolios':[number(level.loc[d]-1) for level in levels]} for d in frame.index]
    warnings=[]
    if begin<available_start: warnings.append('Historique demandé incomplet : le titre le plus récent limite la fenêtre commune.')
    if finish>available_end: warnings.append('Fin limitée à la dernière séance commune disponible.')
    if len(frame)<61: warnings.append('Fenêtre courte : les KPI annualisés peuvent être instables.')
    return {'portfolios':output,'curve':curve,'start':frame.index[0].strftime('%Y-%m-%d'),'end':frame.index[-1].strftime('%Y-%m-%d'),
      'requested_start':begin.strftime('%Y-%m-%d'),'requested_end':finish.strftime('%Y-%m-%d'),'observations':len(frame)-1,
      'benchmark_performance':number(base.iloc[-1]-1),'warnings':warnings,'amount':amount,'rebalance':rebalance,'inflation':inflation}
