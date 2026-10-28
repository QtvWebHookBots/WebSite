import unittest
import numpy as np
import pandas as pd
from engine import path,compute

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2018-01-02','2026-10-27')
        t=np.arange(len(self.dates))
        self.prices=pd.DataFrame({s:100*np.cumprod(1+.0001*k+.007*np.sin(t/(7+k))) for s,k in [('AAA',1),('BBB',2),('CCC',3),('SPY',4),('QQQ',5)]},index=self.dates)
        self.allocations=[{'id':'a','name':'A','positions':[{'ticker':'AAA','weight':60},{'ticker':'BBB','weight':40}]},{'id':'b','name':'B','positions':[{'ticker':'BBB','weight':30},{'ticker':'CCC','weight':70}]}]
    def reference(self,prices,weights,mode):
        # Independent calculation from price ratios since the last reset.
        anchor=prices.iloc[0].to_numpy();capital=1.;out=[1.];previous=prices.index[0]
        for date,row in prices.iloc[1:].iterrows():
            value=capital*np.dot(weights,row.to_numpy()/anchor);out.append(value)
            reset=mode=='daily' or mode=='monthly' and (date.year,date.month)!=(previous.year,previous.month) or mode=='yearly' and date.year!=previous.year
            if reset:anchor=row.to_numpy();capital=value
            previous=date
        return pd.Series(out,index=prices.index)
    def test_rebalance_paths(self):
        for mode in ['none','daily','monthly','yearly']:
            actual=path(self.prices[['AAA','BBB']],[.6,.4],mode)
            expected=self.reference(self.prices[['AAA','BBB']],[.6,.4],mode)
            np.testing.assert_allclose(actual,expected,rtol=1e-11)
        none=path(self.prices[['AAA','BBB']],[.6,.4],'none')
        np.testing.assert_allclose(none,(self.prices[['AAA','BBB']]/self.prices[['AAA','BBB']].iloc[0]).dot([.6,.4]),rtol=1e-11)
        daily=path(self.prices[['AAA','BBB']],[.6,.4],'daily')
        expected=np.r_[1,(1+self.prices[['AAA','BBB']].pct_change().dropna().dot([.6,.4])).cumprod()]
        np.testing.assert_allclose(daily,expected,rtol=1e-11)
    def test_all_metrics_all_windows(self):
        cases=[('months',1,None,None),('months',3,None,None),('months',12,None,None),('months',60,None,None),('ytd',12,None,None),('max',12,None,None),('custom',12,'2020-03-01','2024-04-30')]
        for mode,months,start,end in cases:
            for rebalance in ['none','monthly','yearly','daily']:
                result=compute(self.prices,self.allocations,'SPY',mode,months,start,end,rebalance,.03,10000)
                frame=self.prices.loc[result['start']:result['end']];bench=frame.SPY/frame.SPY.iloc[0]
                br=bench.pct_change().dropna();years=(frame.index[-1]-frame.index[0]).days/365.25
                self.assertEqual(result['curve'][0]['date'],result['start']);self.assertEqual(result['curve'][-1]['date'],result['end'])
                for i,a in enumerate(self.allocations):
                    symbols=[p['ticker'] for p in a['positions']];weights=[p['weight']/100 for p in a['positions']]
                    level=self.reference(frame[symbols],weights,rebalance);r=level.pct_change().dropna();perf=level.iloc[-1]-1
                    expected={'performance':perf,'annualized':level.iloc[-1]**(1/years)-1,'volatility':r.std()*np.sqrt(252),'beta':r.cov(br)/br.var(),'sharpe':r.mean()/r.std()*np.sqrt(252),'real':level.iloc[-1]/1.03**years-1,'drawdown':(level/level.cummax()-1).min(),'final_value':10000*level.iloc[-1],'gain':10000*perf}
                    for key,value in expected.items():self.assertAlmostEqual(result['portfolios'][i]['metrics'][key],value,places=7)
                    np.testing.assert_allclose([p['portfolios'][i] for p in result['curve']],level-1,rtol=1e-10,atol=1e-10)
                    self.assertAlmostEqual(np.prod([1+x['performance'] for x in result['portfolios'][i]['calendar']]),level.iloc[-1],places=9)
    def test_amount_scaling_and_benchmark(self):
        a=compute(self.prices,self.allocations,'SPY','months',60,None,None,'monthly',.03,10000)
        b=compute(self.prices,self.allocations,'SPY','months',60,None,None,'monthly',.03,20000)
        for x,y in zip(a['portfolios'],b['portfolios']):
            self.assertAlmostEqual(y['metrics']['final_value'],2*x['metrics']['final_value'])
            self.assertAlmostEqual(y['metrics']['gain'],2*x['metrics']['gain'])
            self.assertEqual(y['metrics']['performance'],x['metrics']['performance'])
        c=compute(self.prices,self.allocations,'QQQ','months',60,None,None,'monthly',.03,10000)
        self.assertNotEqual(a['portfolios'][0]['metrics']['beta'],c['portfolios'][0]['metrics']['beta'])
    def test_common_window_and_errors(self):
        shortened=self.prices.loc['2024-01-01':]
        result=compute(shortened,self.allocations,'SPY','months',60,None,None,'none',.03,10000)
        self.assertTrue(result['warnings']);self.assertEqual(result['start'],shortened.index[0].strftime('%Y-%m-%d'))
        with self.assertRaises(ValueError):compute(self.prices,self.allocations,'SPY','custom',12,'2025-01-01','2024-01-01','none',.03,10000)
        with self.assertRaises(ValueError):compute(self.prices,self.allocations,'SPY','custom',12,'2026-10-26','2026-10-27','none',.03,10000)
        constant=self.prices.copy();constant['SPY']=100
        result=compute(constant,self.allocations,'SPY','months',12,None,None,'none',.03,10000)
        self.assertIsNone(result['portfolios'][0]['metrics']['beta'])

if __name__=='__main__':unittest.main()
