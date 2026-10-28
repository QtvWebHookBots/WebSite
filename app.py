from flask import Flask,request,jsonify,send_from_directory
from pathlib import Path
from datetime import datetime,timezone
import threading,time,re
import pandas as pd
import numpy as np
import yfinance as yf
from engine import compute,number
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=50000
BASE=Path(__file__).resolve().parent
CACHE={};LOCK=threading.Lock()
BENCHMARKS={'SPY':'S&P 500 (ETF SPY)','QQQ':'Nasdaq-100 (ETF QQQ)','IWM':'Petites capitalisations (ETF IWM)','VT':'Actions mondiales (ETF VT)'}

def history(symbol):
    # Serialized requests limit duplicate fetches across concurrent browser requests.
    with LOCK:
        cached=CACHE.get(symbol)
        if cached and time.time()-cached[0]<900: return cached[1]
        ticker=yf.Ticker(symbol)
        frame=ticker.history(period='10y',interval='1d',auto_adjust=False,timeout=20)
        if frame is None or frame.empty or not {'Close','Adj Close'}.issubset(frame.columns):
            raise ValueError(symbol+' : historique indisponible.')
        metadata=ticker.get_history_metadata() or {}
        if metadata.get('currency')!='USD':
            raise ValueError(symbol+' : devise non USD ou inconnue. Conversion de change non disponible.')
        today=pd.Timestamp.now(tz=frame.index.tz).date()
        frame=frame.loc[[d.date()<today for d in frame.index],['Close','Adj Close']].copy()
        frame.index=pd.to_datetime([d.date() for d in frame.index])
        frame=frame[~frame.index.duplicated(keep='last')].sort_index().replace([np.inf,-np.inf],np.nan).dropna()
        if len(frame)<3 or (frame<=0).any().any(): raise ValueError(symbol+' : données insuffisantes ou invalides.')
        CACHE[symbol]=(time.time(),frame)
        return frame

@app.get('/')
def index(): return send_from_directory(BASE,'index.html')
@app.get('/api/health')
def health(): return jsonify({'status':'ok','version':'8','source':'yfinance','live_data_checked':False})
@app.post('/api/analyze')
def analyze():
    try:
        data=request.get_json(silent=True)
        if not isinstance(data,dict): raise ValueError('Requête JSON attendue.')
        allocations=data.get('allocations')
        if not isinstance(allocations,list) or not 1<=len(allocations)<=2: raise ValueError('Choisissez un ou deux portefeuilles.')
        validated=[]
        for i,a in enumerate(allocations):
            if not isinstance(a,dict): raise ValueError('Allocation invalide.')
            positions=a.get('positions',[])
            if not isinstance(positions,list) or not 1<=len(positions)<=20: raise ValueError('Entre 1 et 20 positions par portefeuille.')
            clean=[];seen=set()
            for p in positions:
                if not isinstance(p,dict): raise ValueError('Position invalide.')
                symbol=str(p.get('ticker','')).strip().upper();w=float(p.get('weight',0))
                if not re.fullmatch(r'[A-Z0-9.^=-]{1,15}',symbol) or symbol in seen or not np.isfinite(w) or not 0<w<=100:
                    raise ValueError('Titres uniques et poids strictement positifs requis.')
                seen.add(symbol);clean.append({'ticker':symbol,'weight':w})
            if abs(sum(p['weight'] for p in clean)-100)>.001: raise ValueError('Le total de chaque allocation doit être de 100 %.')
            validated.append({'id':str(a.get('id',i))[:80],'name':str(a.get('name','Portefeuille'))[:100],'positions':clean})
        benchmark=data.get('benchmark','SPY')
        if benchmark not in BENCHMARKS: raise ValueError('Benchmark non autorisé.')
        rebalance=data.get('rebalance','monthly')
        if rebalance not in ['none','monthly','yearly','daily']: raise ValueError('Rééquilibrage invalide.')
        mode=data.get('mode','months');months=int(data.get('months',12))
        if mode not in ['months','custom','ytd','max'] or months not in [1,3,6,12,24,36,60,120]: raise ValueError('Durée invalide.')
        start=data.get('start');end=data.get('end')
        if mode=='custom':
            start=datetime.strptime(str(start),'%Y-%m-%d').strftime('%Y-%m-%d');end=datetime.strptime(str(end),'%Y-%m-%d').strftime('%Y-%m-%d')
        inflation=float(data.get('inflation',3))/100;amount=float(data.get('amount',10000))
        if not np.isfinite(inflation) or not -.5<=inflation<=1: raise ValueError('Inflation entre -50 et 100 %.')
        if not np.isfinite(amount) or not 0<amount<=1e9: raise ValueError('Montant investi entre 0 et 1 milliard USD, strictement positif.')
        symbols=sorted(set([benchmark]+[p['ticker'] for a in validated for p in a['positions']]))
        frames={symbol:history(symbol) for symbol in symbols}
        common=pd.concat({symbol:frame['Adj Close'] for symbol,frame in frames.items()},axis=1,join='inner').dropna()
        if len(common)<3: raise ValueError('Historique commun insuffisant.')
        result=compute(common,validated,benchmark,mode,months,start,end,rebalance,inflation,amount)
        quotes=[]
        for symbol,frame in frames.items():
            f=frame.loc[:result['end']]
            quotes.append({'ticker':symbol,'price':number(f['Close'].iloc[-1]),'change':number(f['Close'].iloc[-1]/f['Close'].iloc[-2]-1) if len(f)>1 else None,'date':f.index[-1].strftime('%Y-%m-%d')})
        result.update({'quotes':quotes,'benchmark':benchmark,'benchmark_name':BENCHMARKS[benchmark],'fetched':datetime.now(timezone.utc).isoformat()})
        return jsonify(result)
    except (ValueError,TypeError,KeyError,OverflowError) as exc: return jsonify({'error':str(exc)}),400
    except Exception:
        app.logger.exception('Yahoo data retrieval failed')
        return jsonify({'error':'Yahoo Finance indisponible ou requêtes limitées. Aucun chiffre fictif utilisé.'}),502
if __name__=='__main__': app.run(host='127.0.0.1',port=5001,debug=False)
