from flask import Flask, request, jsonify, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
import re, time, threading
from datetime import datetime, timezone
from pathlib import Path
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=20000
CACHE={}
LOCK=threading.Lock()
BASE=Path(__file__).resolve().parent

def history(symbol):
    with LOCK:
        cached=CACHE.get(symbol)
        if cached and time.time()-cached[0]<900:
            return cached[1]
    ticker=yf.Ticker(symbol)
    frame=ticker.history(period='2y',interval='1d',auto_adjust=False,repair=False,timeout=15)
    if frame is None or frame.empty or 'Close' not in frame or 'Adj Close' not in frame:
        raise ValueError(f'{symbol} : historique indisponible. Réessayez plus tard.')
    metadata=ticker.get_history_metadata() or {}
    currency=metadata.get('currency')
    if currency!='USD':
        raise ValueError(f'{symbol} : devise {currency or "inconnue"}. Cette version accepte uniquement les titres en USD, sans conversion de change.')
    # Exclude today's potentially incomplete daily candle. Use exchange-local date.
    today=pd.Timestamp.now(tz=frame.index.tz).date()
    frame=frame.loc[[d.date()<today for d in frame.index]].copy()
    if len(frame)<3:
        raise ValueError(f'{symbol} : historique clôturé insuffisant.')
    frame.index=pd.to_datetime([d.date() for d in frame.index])
    frame=frame[['Close','Adj Close']].replace([np.inf,-np.inf],np.nan).dropna()
    if (frame<=0).any().any():
        raise ValueError(f'{symbol} : prix non valides.')
    result={'frame':frame,'currency':currency}
    with LOCK:
        CACHE[symbol]=(time.time(),result)
    return result

def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None

@app.get('/')
def index():
    return send_from_directory(BASE,'index.html')

@app.post('/api/analyze')
def analyze():
    try:
        data=request.get_json(silent=True) or {}
        positions=data.get('positions',[])
        if not isinstance(positions,list) or not 1<=len(positions)<=20:
            raise ValueError('Choisissez entre 1 et 20 positions.')
        symbols=[]; weights=[]
        for p in positions:
            s=str(p.get('ticker','')).strip().upper()
            w=float(p.get('weight',0))
            if not re.fullmatch(r'[A-Z0-9.^=-]{1,15}',s) or not np.isfinite(w) or not 0<w<=100:
                raise ValueError('Ticker ou poids invalide.')
            if s in symbols: raise ValueError('Les tickers doivent être uniques.')
            symbols.append(s);weights.append(w/100)
        if abs(sum(weights)-1)>0.00001:
            raise ValueError('La somme des poids doit être égale à 100 %.')
        months=int(data.get('months',12))
        if months not in [1,3,6,12,24]: raise ValueError('Période invalide.')
        inflation=float(data.get('inflation',0))/100
        if not np.isfinite(inflation) or not -0.5<=inflation<=1:
            raise ValueError('Inflation annuelle attendue entre -50 et 100 %.')
        histories={s:history(s) for s in symbols}
        benchmark=history('SPY')
        series={s:histories[s]['frame']['Adj Close'] for s in symbols}
        series['__benchmark__']=benchmark['frame']['Adj Close']
        common=pd.concat(series,axis=1,join='inner').dropna().sort_index()
        if len(common)<3: raise ValueError('Historique commun insuffisant.')
        cutoff=common.index[-1]-pd.DateOffset(months=months)
        # Retain one observation before cutoff to compute the first return.
        before=common.loc[common.index<=cutoff].tail(1)
        common=pd.concat([before,common.loc[common.index>cutoff]])
        if len(common)<3: raise ValueError('Période trop courte.')
        returns=common.pct_change(fill_method=None).dropna()
        port=returns[symbols].mul(weights,axis=1).sum(axis=1)
        bench=returns['__benchmark__']
        growth=(1+port).cumprod()
        perf=growth.iloc[-1]-1
        years=(common.index[-1]-common.index[0]).days/365.25
        annual=(1+perf)**(1/years)-1 if years>0 else None
        volatility=port.std(ddof=1)*np.sqrt(252)
        variance=bench.var(ddof=1)
        beta=port.cov(bench)/variance if variance>0 else None
        sharpe=port.mean()/port.std(ddof=1)*np.sqrt(252) if port.std(ddof=1)>0 else None
        levels=pd.concat([pd.Series([1.0],index=[common.index[0]]),growth])
        drawdown=(levels/levels.cummax()-1).min()
        real=(1+perf)/(1+inflation)**years-1
        bg=(1+bench).cumprod()
        curve=[{'date':common.index[0].strftime('%Y-%m-%d'),'portfolio':0.0,'benchmark':0.0}]
        curve += [{'date':d.strftime('%Y-%m-%d'),'portfolio':finite(growth.loc[d]-1),'benchmark':finite(bg.loc[d]-1)} for d in port.index]
        calendar=[]
        for year, part in port.groupby(port.index.year):
            calendar.append({'year':int(year),'portfolio':finite((1+part).prod()-1),'benchmark':finite((1+bench.loc[part.index]).prod()-1)})
        quotes=[]
        for s in symbols:
            f=histories[s]['frame']
            quotes.append({'ticker':s,'price':finite(f['Close'].iloc[-1]),'change':finite(f['Close'].iloc[-1]/f['Close'].iloc[-2]-1),'date':f.index[-1].strftime('%Y-%m-%d'),'currency':'USD'})
        return jsonify({'curve':curve,'calendar':calendar,'quotes':quotes,'metrics':{'performance':finite(perf),'annualized':finite(annual),'volatility':finite(volatility),'beta':finite(beta),'sharpe':finite(sharpe),'real':finite(real),'drawdown':finite(drawdown),'benchmark':finite((1+bench).prod()-1)},'start':common.index[0].strftime('%Y-%m-%d'),'end':common.index[-1].strftime('%Y-%m-%d'),'observations':len(port),'fetched':datetime.now(timezone.utc).isoformat(),'inflation':inflation})
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        return jsonify({'error':str(exc)}),400
    except Exception:
        app.logger.exception('Market data retrieval failed')
        return jsonify({'error':'Yahoo Finance ne répond pas ou limite les requêtes. Aucun chiffre de remplacement n’a été utilisé. Réessayez plus tard.'}),502

if __name__=='__main__':
    print('MarketPort : ouvrir http://127.0.0.1:5000 dans votre navigateur')
    app.run(host='127.0.0.1',port=5000,debug=False)
