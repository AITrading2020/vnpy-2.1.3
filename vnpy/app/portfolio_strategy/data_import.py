import numpy as np
import pandas as pd
import re
from datetime import timedelta


class DataImport:   
    def __init__(self, days, base_dir):
        self.d = days
        self.base_dir = base_dir
        self.get_strategy_data()
        
    def opt_adjust(self, x):
        if re.search('A', x):
            return 1
        else:
            return 0

    def get_all_opt_info(self):
        opt_contract_info = pd.read_csv(self.base_dir + "option_contract_info.csv")
        opt_contract_info = opt_contract_info[opt_contract_info.columns[2:-4]].set_index('symbol')
        return opt_contract_info

    def get_opt_info(self, date):
        opt_contract_info = self.get_all_opt_info()
        opt_contract_info.listed_date = pd.to_datetime(opt_contract_info.listed_date)
        opt_contract_info.de_listed_date = pd.to_datetime(opt_contract_info.de_listed_date)
        opt_contract_info['adjust'] = opt_contract_info['product_name'].apply(lambda x: self.opt_adjust(x))
        exist_opt = opt_contract_info[(opt_contract_info['listed_date'] <= date) & (opt_contract_info['de_listed_date'] >= date)]
        return exist_opt

    def pcr_quantile(self, pcr_type, end_date, period, quantile=[50, 80]):
        start_date = end_date - timedelta(days=int(period * 365))
        pcr = self.etf50[(self.etf50.index >= start_date) & (self.etf50.index <= end_date)]

        pcr_q = []
        for q in quantile:
            pcr_q.append(pcr[pcr_type].quantile(q / 100))

        return pcr_q

    def get_hv_quantile(self, end_date, period, window=30, quantile=[10, 90]):
        start_date = end_date - timedelta(days=int(period * 365))
        etf = self.etf50_p[(self.etf50_p.index >= start_date) & (self.etf50_p.index <= end_date)]
        etf['std'] = np.sqrt(252) * etf['log_rtn'].rolling(window).std()

        q_list = []
        for q in quantile:
            q_list.append(100 * etf['std'].quantile(q / 100))

        return q_list

    def get_etf50(self):
        etf50 = pd.read_csv(self.base_dir + '510050_none.csv', index_col=0)
        etf50.index = pd.to_datetime(etf50.index)
        etf50 = etf50[etf50.index >= pd.to_datetime('2015-02-09')]
        return etf50

    def get_opt_type(self, opt_name):
        opt_contract = self.get_all_opt_info()
        opt_type = opt_contract[opt_contract['order_book_id'] == int(opt_name)]['option_type'].values[0]
        return opt_type

    def get_trade_dates(self):
        my_etf50 = self.get_etf50()
        return my_etf50.index

    def get_none_adjust_opt(self, date):
        exist_opt = self.get_opt_info(date)
        a_opt_list = exist_opt[exist_opt['adjust'] == 1]['product_name'].values.tolist()
        m_opt_list = exist_opt[exist_opt['adjust'] == 0]['product_name'].values.tolist()
        for a_opt in a_opt_list:
          if a_opt.replace('A', 'M') not in m_opt_list:
            m_opt_list.append(a_opt)
        none_adjust_opt = exist_opt[exist_opt['product_name'].isin(m_opt_list)]
        return none_adjust_opt

    def available_opt(self, date, posit):

        """
        find available options 
        """
        # get rid of adjusted options
        exist_opt = self.get_none_adjust_opt(date)

        # calculate strike and days-to-maturity
        exist_opt['product_name'] = exist_opt['product_name'].apply(lambda x: x.replace('A', 'M'))
        exist_opt['strike'] = exist_opt['product_name'].apply(lambda x: int(x[-4:])/1000)
        exist_opt['days'] = ((exist_opt['de_listed_date'] - date)/np.timedelta64(1, 'D')).astype(int)
        exist_opt = exist_opt[exist_opt['days'] > self.d]
        exist_opt = exist_opt[exist_opt['days'] == exist_opt['days'].min()]

        # get ETF50 price at that date
        etf50 = self.get_etf50()
        etf50_price = etf50[etf50.index == date]['open'].values[0]

        # get the minimum difference between strike and ETF50 price
        opt_c, opt_p = [], []
        # OTM
        if posit > 0:
            for d, f in exist_opt.groupby('days'):
                f['diff'] = f['strike'].apply(lambda x: x - etf50_price)
                f['abs_diff'] = abs(f['diff'])
                f_c = f[(f['option_type'] == 'C') & (f['diff'] >= 0)]
                f_p = f[(f['option_type'] == 'P') & (f['diff'] <= 0)]
                if f_c.empty:
                    opt_c = opt_c + f[(f['option_type'] == 'C') & (f['abs_diff'] == min(f['abs_diff']))]['product_name'].values.tolist()
                elif f_c.shape[0] < posit:
                    c_abs_diff = max(f_c['abs_diff'].values)
                    opt_c = opt_c + f_c[f_c['abs_diff'] == c_abs_diff]['product_name'].values.tolist()
                else:
                    c_abs_diff = sorted(f_c['abs_diff'].values)[posit - 1]
                    opt_c = opt_c + f_c[f_c['abs_diff'] == c_abs_diff]['product_name'].values.tolist()
                if f_p.empty:
                    opt_p = opt_p + f[(f['option_type'] == 'P') & (f['abs_diff'] == min(f['abs_diff']))]['product_name'].values.tolist()
                elif f_p.shape[0] < posit:
                    p_abs_diff = max(f_p['abs_diff'].values)
                    opt_p = opt_p + f_p[f_p['abs_diff'] == p_abs_diff]['product_name'].values.tolist()
                else:
                    p_abs_diff = sorted(f_p['abs_diff'].values)[posit - 1]
                    opt_p = opt_p + f_p[f_p['abs_diff'] == p_abs_diff]['product_name'].values.tolist()
        # ITM
        else:
            for d, f in exist_opt.groupby('days'):
                f['diff'] = f['strike'].apply(lambda x: x - etf50_price)
                f['abs_diff'] = abs(f['diff'])
                f_c = f[(f['option_type'] == 'C') & (f['diff'] <= 0)]
                f_p = f[(f['option_type'] == 'P') & (f['diff'] >= 0)]
                if f_c.empty:
                    opt_c = opt_c + f[(f['option_type'] == 'C') & (f['abs_diff'] == min(f['abs_diff']))]['product_name'].values.tolist()
                elif f_c.shape[0] < -posit:
                    c_abs_diff = max(f_c['abs_diff'].values)
                    opt_c = opt_c + f_c[f_c['abs_diff'] == c_abs_diff]['product_name'].values.tolist()
                else:
                    c_abs_diff = sorted(f_c['abs_diff'].values)[-posit - 1]
                    opt_c = opt_c + f_c[f_c['abs_diff'] == c_abs_diff]['product_name'].values.tolist()
                if f_p.empty:
                    opt_p = opt_p + f[(f['option_type'] == 'P') & (f['abs_diff'] == min(f['abs_diff']))]['product_name'].values.tolist()
                elif f_p.shape[0] < -posit:
                    p_abs_diff = max(f_p['abs_diff'].values)
                    opt_p = opt_p + f_p[f_p['abs_diff'] == p_abs_diff]['product_name'].values.tolist()
                else:
                    p_abs_diff = sorted(f_p['abs_diff'].values)[-posit - 1]
                    opt_p = opt_p + f_p[f_p['abs_diff'] == p_abs_diff]['product_name'].values.tolist()

        opt = opt_c + opt_p
        tradable_opt = exist_opt[exist_opt['product_name'].isin(opt)]
        ttm = tradable_opt["days"].values[0]
        return tradable_opt, ttm

    def get_strategy_data(self):
        etf50 = pd.read_csv(self.base_dir + '510050_none.csv', index_col=0)
        etf50.index = pd.to_datetime(etf50.index)
        etf50_p = etf50
        etf50_p['log_rtn'] = np.log(etf50_p['close']).diff()
        etf50 = etf50[(etf50.index >= pd.to_datetime('2014-12-09')) & (etf50.index <= pd.to_datetime('2020-03-31'))]

        etf50_min = pd.read_csv(self.base_dir + '510050_minute.csv', index_col=0)
        etf50_min.index = pd.to_datetime(etf50_min.index)
        etf50_min = etf50_min[etf50_min.index >= pd.to_datetime('2010-12-09')]
        etf50_min['rtn'] = np.log(etf50_min.close).diff()

        def get_rv(date):
            log_rtn = etf50_min[(etf50_min.index > date) & (etf50_min.index < date + timedelta(days=1))].rtn
            rv = np.sqrt(np.power(log_rtn, 2).sum() * 252)
            return rv

        etf50['date'] = etf50.index
        etf50['close_ma20'] = etf50['close'].rolling(20).mean()
        etf50['close_std'] = etf50['close'].rolling(20).std()
        etf50['rv'] = etf50['date'].apply(lambda x: 100 * get_rv(x))
        etf50['rv_std'] = etf50['rv'].rolling(20).std()
        etf50 = etf50[(etf50.index >= pd.to_datetime('2015-02-09')) & (etf50.index <= pd.to_datetime('2020-03-30'))]

        vix = pd.read_csv(self.base_dir + 'vix_data/vix_close_ex.csv', index_col=0)
        vix.index = pd.to_datetime(vix.index)
        vix['etf50'] = etf50['close']
        vix['date'] = vix.index
        vix['rv'] = vix['date'].apply(lambda x: 100 * get_rv(x))
        vix['rv_std'] = etf50['rv_std'].values.tolist()

        vix['ma20'] = vix['vix'].rolling(20).mean()
        for i in vix.index.values[:20]:
            vix.loc[i, 'ma20'] = vix['vix'][:i + 1].mean()
        vix['ma5'] = vix['vix'].rolling(5).mean()
        for i in vix.index.values[:5]:
            vix.loc[i, 'ma5'] = vix['vix'][:i + 1].mean()

        pcr_v = pd.read_csv(self.base_dir + 'pcr_data/pcr_v.csv', index_col=0)
        pcr_v.index = pd.to_datetime(pcr_v.index)
        etf50['pcr_v'] = pcr_v['volume']
        etf50['v_rate'] = etf50['pcr_v'] / etf50['pcr_v'].shift()
        etf50['pcr_v20'] = etf50['pcr_v'].rolling(20).mean()
        for i in etf50.index.values[:20]:
            etf50.loc[i, 'pcr_v20'] = etf50['pcr_v'][:i + 1].mean()
        etf50['pcr_v10'] = etf50['pcr_v'].rolling(10).mean()
        for i in etf50.index.values[:10]:
            etf50.loc[i, 'pcr_v10'] = etf50['pcr_v'][:i + 1].mean()
        etf50['pcr_v5'] = etf50['pcr_v'].rolling(5).mean()
        for i in etf50.index.values[:5]:
            etf50.loc[i, 'pcr_v5'] = etf50['pcr_v'][:i + 1].mean()
        etf50['pcr_v2'] = etf50['pcr_v'].rolling(2).mean()
        for i in etf50.index.values[:2]:
            etf50.loc[i, 'pcr_v2'] = etf50['pcr_v'][:i + 1].mean()

        pcr_oi = pd.read_csv(self.base_dir + 'pcr_data/pcr_oi.csv', index_col=0)
        pcr_oi.index = pd.to_datetime(pcr_oi.index)
        etf50['pcr_oi'] = pcr_oi['open_interest']
        etf50['pcr_oi5'] = etf50['pcr_oi'].rolling(5).mean()
        for i in etf50.index.values[:5]:
            etf50.loc[i, 'pcr_oi5'] = etf50['pcr_oi'][:i + 1].mean()
        etf50['pcr_oi20'] = etf50['pcr_oi'].rolling(20).mean()
        for i in etf50.index.values[:20]:
            etf50.loc[i, 'pcr_oi20'] = etf50['pcr_oi'][:i + 1].mean()

        pcr_tt = pd.read_csv(self.base_dir + 'pcr_data/pcr_tt.csv', index_col=0)
        pcr_tt.index = pd.to_datetime(pcr_tt.index)
        etf50['pcr_tt'] = pcr_tt['total_turnover']
        pcr_tt['rate'] = pcr_tt['total_turnover'] / pcr_tt['total_turnover'].shift()
        etf50['tt_rate'] = pcr_tt['rate']

        self.etf50=etf50
        self.etf50_p=etf50_p
        self.vix=vix