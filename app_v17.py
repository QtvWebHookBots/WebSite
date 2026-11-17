from flask import Flask,request,jsonify,send_from_directory,Response
from pathlib import Path
from datetime import datetime,timezone
import os,threading,time,re,json,csv,io,secrets
import pandas as pd
import numpy as np
import yfinance as yf
from engine_v17 import compute,number
BASE=Path(__file__).resolve().parent; DATA=BASE/'data_v17.json'; MEDIA=BASE/'media_v17'; MEDIA.mkdir(exist_ok=True)
ALLOWED_IMAGES={'image/jpeg':'.jpg','image/png':'.png','image/webp':'.webp','image/gif':'.gif'}
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=2_000_000
LOCK=threading.RLock(); CACHE={}
BENCHMARKS={'SPY':'S&P 500 (ETF SPY)','QQQ':'Nasdaq-100 (ETF QQQ)','IWM':'Russell 2000 (ETF IWM)','VT':'Actions mondiales (ETF VT)'}
def load():
 with LOCK:return json.loads(DATA.read_text(encoding='utf-8'))
def save(d):
 with LOCK:
  t=DATA.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(DATA)
def is_admin():return bool(os.getenv('ADMIN_TOKEN')) and request.headers.get('X-Admin-Token')==os.getenv('ADMIN_TOKEN')
def guard():
 if not is_admin():raise PermissionError('Jeton administrateur invalide.')
def history(s):
 with LOCK:
  c=CACHE.get(s)
  if c and time.time()-c[0]<900:return c[1]
  t=yf.Ticker(s);f=t.history(period='10y',interval='1d',auto_adjust=False,timeout=25)
  if f is None or f.empty:raise ValueError(s+' : historique indisponible.')
  if 'Adj Close' not in f.columns:f['Adj Close']=f['Close']
  today=pd.Timestamp.now(tz=f.index.tz).date();f=f.loc[[d.date()<today for d in f.index],[c for c in ['Close','Adj Close','Dividends'] if c in f.columns]].copy();f['Dividends']=f['Dividends'] if 'Dividends' in f.columns else 0.0;f.index=pd.to_datetime([d.date() for d in f.index]);f=f[~f.index.duplicated()].replace([np.inf,-np.inf],np.nan).dropna()
  if len(f)<3:raise ValueError(s+' : donnees insuffisantes.')
  CACHE[s]=(time.time(),f);return f
def ledger(p,prices={}):
 lots={};realized=0.;fees=0.
 for t in sorted(p.get('transactions',[]),key=lambda x:x['date']):
  s=t['ticker'];q=float(t['quantity']);px=float(t['price']);fee=float(t.get('fees',0));fees+=fee;a=t['action'].upper();h=lots.setdefault(s,{'quantity':0.,'cost':0.,'name':t.get('name',s)})
  if a=='BUY':h['quantity']+=q;h['cost']+=q*px+fee
  elif a=='SELL':
   if q>h['quantity']+1e-9:raise ValueError('Vente superieure a la quantite detenue pour '+s)
   avg=h['cost']/h['quantity'] if h['quantity'] else 0;realized+=q*(px-avg)-fee;h['quantity']-=q;h['cost']-=q*avg
 rows=[];total=0
 for s,h in lots.items():
  if h['quantity']<=1e-9:continue
  last=prices.get(s);value=h['quantity']*last if last else None;total+=value or 0
  rows.append({'ticker':s,'name':h['name'],'quantity':number(h['quantity']),'average_price':number(h['cost']/h['quantity']),'cost_basis':number(h['cost']),'last_price':number(last) if last else None,'market_value':number(value) if value else None})
 for r in rows:r['current_weight']=number(r['market_value']/total) if total and r['market_value'] is not None else None;r['unrealized']=number(r['market_value']-r['cost_basis']) if r['market_value'] is not None else None
 return {'holdings':rows,'market_value':number(total),'realized_pnl':number(realized),'fees':number(fees),'transaction_count':len(p.get('transactions',[]))}

def transaction_performance(portfolio,frames,start,end):
    transactions=sorted(portfolio.get('transactions',[]),key=lambda x:(x['date'],x.get('action','')))
    if not transactions:return {'available':False,'reason':'Aucune transaction enregistree.','first_transactions':[]}
    first_date=pd.Timestamp(transactions[0]['date']).normalize();window_start=pd.Timestamp(start);window_end=pd.Timestamp(end)
    market_symbols=sorted(set(t.get('ticker','') for t in transactions if t.get('action','').upper() in ('BUY','SELL') and t.get('ticker') in frames))
    if not market_symbols:return {'available':False,'reason':'Aucun achat ou vente valorisable.','first_transactions':[]}
    dates=None
    for symbol in market_symbols:
        idx=frames[symbol].loc[first_date:window_end].index
        dates=idx if dates is None else dates.union(idx)
    dates=dates.sort_values().unique()
    if len(dates)<2:return {'available':False,'reason':'Historique transactionnel insuffisant.','first_transactions':[]}
    first_transactions=[{'date':t['date'],'action':t.get('action','').upper(),'ticker':t.get('ticker',''),'name':t.get('name',t.get('ticker','')),'quantity':number(float(t.get('quantity',0))) if t.get('quantity') not in (None,'') else None} for t in transactions if pd.Timestamp(t['date']).normalize()==first_date]
    txmap={}
    for t in transactions:txmap.setdefault(pd.Timestamp(t['date']).normalize(),[]).append(t)
    qty={};cash=0.;prev_prices={};index_level=100.;full_curve=[];component_totals={};started=False
    for date in dates:
        day=pd.Timestamp(date).normalize();begin_value=cash
        for symbol,q in qty.items():
            if q<=0:continue
            f=frames[symbol];available=f.loc[:date]
            if len(available):begin_value+=q*float(available.Close.iloc[-2] if len(available)>1 else available.Close.iloc[-1])
        asset_contrib={};dividend_cash=0.
        for symbol,q in list(qty.items()):
            if q<=0:continue
            f=frames[symbol];available=f.loc[:date]
            if not len(available):continue
            price=float(available.Close.iloc[-1]);prior=prev_prices.get(symbol,float(available.Close.iloc[-2] if len(available)>1 else price));div=float(available.Dividends.iloc[-1]) if 'Dividends' in available.columns and available.index[-1]==date else 0.
            pnl=q*(price-prior)+q*div;asset_contrib[symbol]=pnl;dividend_cash+=q*div;prev_prices[symbol]=price
        cash+=dividend_cash;external_flow=0.;fees_day=0.
        day_transactions=txmap.get(day,[])
        if not started and not any(t.get('action','').upper() in ('DEPOSIT','APPORT') for t in day_transactions):
            implicit_funding=sum(float(t.get('quantity',0))*float(t.get('price',0))+float(t.get('fees',0) or 0) for t in day_transactions if t.get('action','').upper()=='BUY')
            cash+=implicit_funding;external_flow+=implicit_funding
        for t in day_transactions:
            action=t.get('action','').upper();fee=float(t.get('fees',0) or 0);fees_day+=fee
            if action in ('DEPOSIT','APPORT'):
                amount=float(t.get('amount',t.get('price',0)) or 0);cash+=amount;external_flow+=amount
            elif action in ('WITHDRAWAL','RETRAIT'):
                amount=float(t.get('amount',t.get('price',0)) or 0);cash-=amount;external_flow-=amount
            elif action in ('BUY','SELL'):
                symbol=t['ticker'];q=float(t['quantity']);px=float(t['price']);qty.setdefault(symbol,0.)
                if action=='BUY':cash-=q*px+fee;qty[symbol]+=q
                else:
                    if q>qty[symbol]+1e-9:raise ValueError('Vente superieure a la position pour '+symbol)
                    cash+=q*px-fee;qty[symbol]-=q
                prev_prices[symbol]=px
        end_value=cash
        for symbol,q in qty.items():
            if q<=0:continue
            available=frames[symbol].loc[:date]
            if len(available):end_value+=q*float(available.Close.iloc[-1])
        if not started:
            # The first external funding establishes capital; it is not performance.
            if end_value>0:started=True;begin_value=end_value-external_flow if end_value-external_flow>0 else end_value
            daily_return=0.
        elif begin_value>0:
            daily_return=(end_value-external_flow)/begin_value-1
        else:daily_return=0.
        old_index=index_level;index_level*=1+daily_return
        raw={k:v/begin_value for k,v in asset_contrib.items()} if begin_value>0 else {}
        if fees_day and begin_value>0:raw['FRAIS']=raw.get('FRAIS',0)-fees_day/begin_value
        explained=sum(raw.values())
        if abs(explained)>1e-15:
            scale=daily_return/explained
            raw={k:v*scale for k,v in raw.items()}
        # Geometric linking: previous contribution grows with the day's return, then today's contribution is added.
        keys=set(component_totals)|set(raw)
        component_totals={k:component_totals.get(k,0)*(1+daily_return)+raw.get(k,0) for k in keys}
        full_curve.append({'date':day.strftime('%Y-%m-%d'),'index':number(index_level),'daily_return':number(daily_return)})
    selected=[x for x in full_curve if pd.Timestamp(x['date'])>=window_start and pd.Timestamp(x['date'])<=window_end]
    if len(selected)<2:return {'available':False,'reason':'Moins de deux valorisations sur la fenetre.','first_transactions':first_transactions,'inception_date':first_date.strftime('%Y-%m-%d')}
    base=selected[0]['index'];curve=[{'date':x['date'],'index':number(x['index']/base*100),'performance':number(x['index']/base-1)} for x in selected]
    perf=curve[-1]['performance'];names={t.get('ticker',''):t.get('name',t.get('ticker','')) for t in transactions}
    total_component=sum(component_totals.values());factor=perf/total_component if abs(total_component)>1e-15 else 0
    contributions=[{'ticker':k,'name':'Frais' if k=='FRAIS' else names.get(k,k),'contribution':number(v*factor)} for k,v in component_totals.items()]
    contributions.sort(key=lambda x:abs(x['contribution']),reverse=True)
    return {'available':True,'inception_date':first_date.strftime('%Y-%m-%d'),'start':curve[0]['date'],'end':curve[-1]['date'],'performance':number(perf),'index_start':100.0,'index_end':number(curve[-1]['index']),'curve':curve,'contributions':contributions,'first_transactions':first_transactions,'external_flows_supported':['DEPOSIT','WITHDRAWAL'],'method':'TWR quotidien, indice base 100. Les apports et retraits sont neutralises; achats, ventes, dividendes et frais sont integres.'}

@app.get('/')
def root():return send_from_directory(BASE,'index_v17.html')
@app.get('/api/content')
def content():
 d=load();return jsonify({'portfolios':[{k:v for k,v in p.items() if k!='transactions'} for p in d['portfolios']],'articles':[a for a in d['articles'] if a.get('published',True)]})
@app.get('/media/<name>')
def media(name):
 if not re.fullmatch(r'[A-Za-z0-9_-]+\.(?:jpg|png|webp|gif)',name):return 'Introuvable',404
 return send_from_directory(MEDIA,name,max_age=31536000)
@app.get('/api/health')
def health():return jsonify({'status':'ok','version':'17','admin':is_admin()})

@app.get('/api/markets')
def markets():
 try:
  symbols={'SPY':'S&P 500','QQQ':'Nasdaq-100','IWM':'Russell 2000','VT':'Actions mondiales','EURUSD=X':'EUR/USD','^VIX':'VIX','^TNX':'Taux US 10 ans'};out=[]
  for symbol,name in symbols.items():
   f=history(symbol); last=float(f.Close.iloc[-1]); prev=float(f.Close.iloc[-2]);out.append({'ticker':symbol,'name':name,'value':number(last),'change':number(last/prev-1),'date':f.index[-1].strftime('%Y-%m-%d')})
  return jsonify(out)
 except Exception:return jsonify({'error':'Indicateurs de marche indisponibles.'}),502

@app.get('/api/admin/content')
def admin_content():
 try:guard();return jsonify(load())
 except PermissionError as e:return jsonify({'error':str(e)}),401
@app.put('/api/admin/content')
def put_content():
 try:
  guard();d=request.get_json();
  if not isinstance(d,dict) or not isinstance(d.get('portfolios'),list) or not isinstance(d.get('articles'),list):raise ValueError('Structure invalide.')
  for p in d['portfolios']:
   if not p.get('id') or not p.get('name') or not p.get('positions'):raise ValueError('Portefeuille incomplet.')
   if abs(sum(float(x['weight']) for x in p['positions'])-100)>.01:raise ValueError('Les poids doivent totaliser 100 %.')
   ledger(p)
  save(d);CACHE.clear();return jsonify({'ok':True})
 except PermissionError as e:return jsonify({'error':str(e)}),401
 except (ValueError,TypeError,KeyError) as e:return jsonify({'error':str(e)}),400
def clean_rich_html(value):
 value=str(value or '')[:200000]
 value=re.sub(r'<\s*(script|style|iframe|object|embed|form)[^>]*>.*?<\s*/\s*\1\s*>','',value,flags=re.I|re.S)
 value=re.sub(r'\son\w+\s*=\s*(["\']).*?\1','',value,flags=re.I|re.S)
 value=re.sub(r'javascript\s*:','',value,flags=re.I)
 # Images must reference this application's own media route.
 value=re.sub(r'<img\s+([^>]*?)src=["\'](?!/media/)[^"\']*["\']([^>]*)>','',value,flags=re.I)
 return value

@app.post('/api/admin/image')
def upload_article_image():
 try:
  guard();f=request.files.get('image')
  if not f or not f.filename:raise ValueError('Image requise.')
  mime=(f.mimetype or '').lower()
  if mime not in ALLOWED_IMAGES:raise ValueError('Formats autorises : JPG, PNG, WEBP, GIF.')
  blob=f.read(5_000_001)
  if len(blob)>5_000_000:raise ValueError('Image limitee a 5 Mo.')
  if len(blob)<16:raise ValueError('Image vide ou invalide.')
  sig_ok=(mime=='image/jpeg' and blob[:3]==b'\xff\xd8\xff') or (mime=='image/png' and blob[:8]==b'\x89PNG\r\n\x1a\n') or (mime=='image/gif' and blob[:6] in (b'GIF87a',b'GIF89a')) or (mime=='image/webp' and blob[:4]==b'RIFF' and blob[8:12]==b'WEBP')
  if not sig_ok:raise ValueError('Signature du fichier incoherente avec son type.')
  name=secrets.token_hex(16)+ALLOWED_IMAGES[mime];(MEDIA/name).write_bytes(blob)
  return jsonify({'url':'/media/'+name,'name':name})
 except PermissionError as e:return jsonify({'error':str(e)}),401
 except (ValueError,TypeError) as e:return jsonify({'error':str(e)}),400

@app.post('/api/admin/article')
def article():
 try:
  guard();d=load();a=request.get_json() or {}
  if not a.get('title') or not a.get('body_html'):raise ValueError('Titre et contenu obligatoires.')
  tags=[str(x)[:40] for x in a.get('tags',[]) if str(x).strip()][:12];portfolios=[str(x)[:80] for x in a.get('portfolio_ids',[])][:12];tickers=[re.sub('[^A-Z0-9.^=-]','',str(x).upper())[:15] for x in a.get('tickers',[])][:30]
  article={'id':str(a.get('id') or 'article-'+str(int(time.time())))[:100],'title':str(a['title'])[:160],'subtitle':str(a.get('subtitle',''))[:240],'summary':str(a.get('summary',''))[:500],'body_html':clean_rich_html(a['body_html']),'cover_image':str(a.get('cover_image',''))[:200] if str(a.get('cover_image','')).startswith('/media/') else '', 'date':a.get('date') or datetime.now().date().isoformat(),'published':bool(a.get('published',False)),'tags':tags,'portfolio_ids':portfolios,'tickers':tickers}
  old=next((i for i,x in enumerate(d['articles']) if x.get('id')==article['id']),None)
  if old is None:d['articles'].append(article)
  else:d['articles'][old]=article
  save(d);return jsonify(article)
 except PermissionError as e:return jsonify({'error':str(e)}),401
 except (ValueError,TypeError) as e:return jsonify({'error':str(e)}),400

@app.delete('/api/admin/article/<aid>')
def delete_article(aid):
 try:
  guard();d=load();before=len(d['articles']);d['articles']=[a for a in d['articles'] if a.get('id')!=aid]
  if len(d['articles'])==before:return jsonify({'error':'Article introuvable.'}),404
  save(d);return jsonify({'ok':True})
 except PermissionError as e:return jsonify({'error':str(e)}),401
@app.post('/api/admin/transaction/<pid>')
def transaction(pid):
 try:
  guard();d=load();p=next((x for x in d['portfolios'] if x['id']==pid),None)
  if not p:raise ValueError('Portefeuille introuvable.')
  raw=request.get_json() or {};action=str(raw.get('action','')).upper();date=datetime.strptime(raw['date'],'%Y-%m-%d').strftime('%Y-%m-%d')
  if action in ('DEPOSIT','WITHDRAWAL'):
   amount=float(raw.get('amount',raw.get('price',0)))
   if amount<=0:raise ValueError('Montant strictement positif requis.')
   t={'date':date,'action':action,'amount':amount,'ticker':'','name':'Apport' if action=='DEPOSIT' else 'Retrait','quantity':0,'price':amount,'fees':0,'comment':str(raw.get('comment',''))[:300]}
  elif action in ('BUY','SELL'):
   ticker=re.sub('[^A-Z0-9.^=-]','',str(raw.get('ticker','')).upper())[:15];quantity=float(raw.get('quantity',0));price=float(raw.get('price',0));fees=float(raw.get('fees',0))
   if not ticker or quantity<=0 or price<=0 or fees<0:raise ValueError('Achat ou vente invalide.')
   t={'date':date,'action':action,'ticker':ticker,'name':str(raw.get('name') or ticker)[:100],'quantity':quantity,'price':price,'fees':fees,'comment':str(raw.get('comment',''))[:300]}
  else:raise ValueError('Type autorise : BUY, SELL, DEPOSIT ou WITHDRAWAL.')
  candidate=dict(p);candidate['transactions']=list(p.get('transactions',[]))+[t]
  # Validate oversells through the ledger for security.
  if action in ('BUY','SELL'):ledger(candidate)
  p.setdefault('transactions',[]).append(t);save(d);return jsonify({'ok':True})
 except PermissionError as e:return jsonify({'error':str(e)}),401
 except (ValueError,TypeError,KeyError) as e:return jsonify({'error':str(e)}),400
@app.get('/api/transactions/<pid>.csv')
def export_tx(pid):
 p=next((x for x in load()['portfolios'] if x['id']==pid),None)
 if not p:return 'Introuvable',404
 out=io.StringIO();w=csv.writer(out,delimiter=';');w.writerow(['Date','Sens','Ticker','Action','Quantite','Prix unitaire USD','Frais USD','Montant brut USD','Commentaire'])
 for t in sorted(p.get('transactions',[]),key=lambda x:x['date']):w.writerow([t['date'],t['action'],t['ticker'],t.get('name',''),t['quantity'],t['price'],t.get('fees',0),float(t['quantity'])*float(t['price']),t.get('comment','')])
 l=ledger(p);w.writerow([]);w.writerow(['SYNTHESE']);w.writerow(['Ticker','Action','Quantite detenue','Prix moyen USD','Cout de revient USD'])
 for h in l['holdings']:w.writerow([h['ticker'],h['name'],h['quantity'],h['average_price'],h['cost_basis']])
 return Response('\ufeff'+out.getvalue(),mimetype='text/csv',headers={'Content-Disposition':f'attachment; filename=transactions-{pid}.csv'})
@app.post('/api/analyze')
def analyze():
 try:
  d=request.get_json();db={p['id']:p for p in load()['portfolios']};ids=d.get('portfolio_ids',[])
  if not 1<=len(ids)<=8 or any(x not in db for x in ids):raise ValueError('Choisissez un ou deux portefeuilles.')
  a=[db[x] for x in ids];b=d.get('benchmark','SPY');mode=d.get('mode','months');months=int(d.get('months',12));reb=d.get('rebalance','monthly');start=d.get('start');end=d.get('end')
  if b not in BENCHMARKS or mode not in ('months','custom','ytd','max') or reb not in ('none','monthly','yearly','daily'):raise ValueError('Parametres invalides.')
  sy=sorted(set([b,'EURUSD=X']+[x['ticker'] for p in a for x in p['positions']]));frames={s:history(s) for s in sy};common=pd.concat({s:f['Adj Close'] for s,f in frames.items()},axis=1,join='inner').dropna()
  preliminary=compute(common,a,b,mode,months,start,end,reb,float(d.get('inflation',3))/100,float(d.get('amount',10000)))
  inception_dates=[]
  for portfolio in a:
   tx_dates=[pd.Timestamp(t['date']) for t in portfolio.get('transactions',[]) if t.get('date')]
   inception_dates.append(min(tx_dates) if tx_dates else pd.Timestamp(portfolio.get('inception_date',common.index[0])))
  comparison_start=max(inception_dates)
  requested_start=pd.Timestamp(preliminary['requested_start']);effective_start=max(requested_start,comparison_start)
  was_locked=effective_start>requested_start
  if pd.Timestamp(preliminary['end'])<=effective_start:raise ValueError('Fenetre vide apres application de la date commune de mise en place des portefeuilles.')
  r=compute(common,a,b,'custom',months,effective_start.strftime('%Y-%m-%d'),preliminary['end'],reb,float(d.get('inflation',3))/100,float(d.get('amount',10000)))
  r['requested_start']=preliminary['requested_start'];r['comparison_start']=comparison_start.strftime('%Y-%m-%d');r['date_lock_applied']=was_locked;r['inception_dates']=[x.strftime('%Y-%m-%d') for x in inception_dates]
  fx=common.loc[r['start']:r['end'],'EURUSD=X'];fx_factor=fx.iloc[0]/fx;
  for i,row in enumerate(r['curve']):
   row['eurusd']=number(fx.loc[pd.Timestamp(row['date'])]);row['fx_effect']=number(fx_factor.loc[pd.Timestamp(row['date'])]-1);row['portfolios_eur']=[number((1+v)*fx_factor.loc[pd.Timestamp(row['date'])]-1) for v in row['portfolios']]
  r['fx_summary']={'start':number(fx.iloc[0]),'end':number(fx.iloc[-1]),'effect':number(fx_factor.iloc[-1]-1)};quotes=[];prices={}
  for s,f in frames.items():
   q=f.loc[:r['end']];prices[s]=float(q.Close.iloc[-1]);quotes.append({'ticker':s,'price':number(prices[s]),'change':number(q.Close.iloc[-1]/q.Close.iloc[-2]-1),'date':q.index[-1].strftime('%Y-%m-%d')})
  contributions=[]
  frame=common.loc[r['start']:r['end']]
  for portfolio in a:
   rows=[]
   for pos in portfolio['positions']:
    asset_return=float(frame[pos['ticker']].iloc[-1]/frame[pos['ticker']].iloc[0]-1);rows.append({'ticker':pos['ticker'],'name':pos.get('name',pos['ticker']),'weight':pos['weight'],'asset_return':number(asset_return),'contribution':number(asset_return*pos['weight']/100)})
   contributions.append(rows)
  management=[transaction_performance(p,frames,r['start'],r['end']) for p in a]
  # Management benchmark starts only when real management history exists.
  benchmark_management=[]
  for mg in management:
   if not mg.get('available'):
    benchmark_management.append({'available':False});continue
   bd=common.loc[mg['start']:r['end'],b];base_b=float(bd.iloc[0]);benchmark_management.append({'available':True,'start':mg['start'],'curve':[{'date':dt.strftime('%Y-%m-%d'),'performance':number(float(v/base_b-1))} for dt,v in bd.items()]})
  r.update({'quotes':quotes,'benchmark':b,'benchmark_name':BENCHMARKS[b],'ledgers':[ledger(p,prices) for p in a],'contributions':contributions,'management':management,'benchmark_management':benchmark_management,'fetched':datetime.now(timezone.utc).isoformat()});return jsonify(r)
 except (ValueError,TypeError,KeyError,OverflowError) as e:return jsonify({'error':str(e)}),400
 except Exception:app.logger.exception('analyse');return jsonify({'error':'Donnees de marche indisponibles.'}),502
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.getenv('PORT','5001')),debug=False)
