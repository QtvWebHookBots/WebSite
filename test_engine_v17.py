import unittest
import numpy as np
import pandas as pd
from engine_v17 import path,compute
class EngineV17Tests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2020-01-02','2026-10-27'); t=np.arange(len(self.dates))
        self.prices=pd.DataFrame({s:100*np.cumprod(1+.0001*k+.003*np.sin(t/(7+k))) for s,k in [('AAA',1),('BBB',2),('SPY',4)]},index=self.dates)
        self.a=[{'id':'a','name':'A','positions':[{'ticker':'AAA','weight':60},{'ticker':'BBB','weight':40}]}]
    def test_total_path(self):
        for mode in ['none','daily','monthly','yearly']:
            x=path(self.prices[['AAA','BBB']],[.6,.4],mode); self.assertEqual(len(x),len(self.prices)); self.assertAlmostEqual(x.iloc[0],1)
    def test_windows_and_metrics(self):
        for mode,months,start,end in [('months',12,None,None),('ytd',12,None,None),('max',12,None,None),('custom',12,'2022-01-01','2024-01-01')]:
            r=compute(self.prices,self.a,'SPY',mode,months,start,end,'monthly',.03,10000)
            self.assertIn('curve',r); self.assertEqual(len(r['portfolios']),1); self.assertGreater(r['portfolios'][0]['metrics']['final_value'],0)
    def test_bad_dates(self):
        with self.assertRaises(ValueError):compute(self.prices,self.a,'SPY','custom',12,'2025-01-01','2024-01-01','none',.03,10000)
if __name__=='__main__':unittest.main()

class TransactionTWRTests(unittest.TestCase):
    def _frames(self):
        idx=pd.bdate_range('2024-01-02',periods=5)
        return {'AAA':pd.DataFrame({'Close':[100,110,110,121,121],'Adj Close':[100,110,110,121,121],'Dividends':[0,0,0,0,0]},index=idx)}
    def test_deposit_does_not_create_performance(self):
        from app_v17 import transaction_performance
        p={'transactions':[{'date':'2024-01-02','action':'DEPOSIT','amount':1000},{'date':'2024-01-02','action':'BUY','ticker':'AAA','name':'AAA','quantity':10,'price':100,'fees':0},{'date':'2024-01-04','action':'DEPOSIT','amount':500}]}
        r=transaction_performance(p,self._frames(),'2024-01-02','2024-01-08')
        self.assertTrue(r['available']);self.assertAlmostEqual(r['performance'],0.175625,places=6)
        self.assertEqual(r['index_start'],100.0);self.assertEqual(r['inception_date'],'2024-01-02')
    def test_first_transactions_exposed(self):
        from app_v17 import transaction_performance
        p={'transactions':[{'date':'2024-01-02','action':'BUY','ticker':'AAA','name':'Action AAA','quantity':10,'price':100,'fees':0}]}
        r=transaction_performance(p,self._frames(),'2024-01-02','2024-01-08')
        self.assertEqual(r['first_transactions'][0]['name'],'Action AAA')
