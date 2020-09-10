import numpy as np
import pandas as pd
import re

UNDERLYING_DICT = {
    "510050": "510050", "510300": "510300",
    "CU": "CU889", "AU": "AU889", "RU": "RU99",
    "C": "C889", "M": "M889", "I": "I889", "PG": "PG889",
    "MA": "MA99", "CF": "CF99", "RM": "RM99", "SR": "SR99", "TA": "TA99"
}

OPTION_DOMINANT_MONTH = {
    "510050": [], "510300": [],
    "CU": [], "AU": [6, 12], "RU": [1, 5, 9],
    "C": [1, 5, 9], "M": [1, 5, 9], "I": [1, 5, 9], "PG": [],
    "MA": [1, 5, 9], "CF": [1, 5, 9], "RM": [1, 5, 9], "SR": [1, 5, 9], "TA": [1, 5, 9]
}

class DataImport:

    def __init__(self, days, underlying_symbol, base_dir, dominant):

        self.d = days
        self.base_dir = base_dir

        dominant_months = OPTION_DOMINANT_MONTH[underlying_symbol]
        if underlying_symbol.startswith("510"):
            underlying_symbol = underlying_symbol + ".XSHG"
        self.underlying_symbol = underlying_symbol
        self.underlying_dir = self.underlying_symbol.split(".")[0]
        self.underlying_base_dir = self.base_dir + self.underlying_dir + "/"

        opt_contract_info = pd.read_csv(self.base_dir + "opt_contract_info.csv")
        opt_contract_info = opt_contract_info[opt_contract_info["underlying_symbol"] == self.underlying_symbol]
        self.opt_contract_info = opt_contract_info.set_index('symbol')

        if (not dominant) or (len(dominant_months) == 0):
            pass
        else:
            dominant_month_list = []
            for dm in dominant_months:
                if dm < 10:
                    dominant_month_list.append("0" + str(dm))
                else:
                    dominant_month_list.append(str(dm))

            self.opt_contract_info["dominant"] = self.opt_contract_info["underlying_order_book_id"].apply(
                lambda x: str(x)[-2:] in dominant_month_list)
            self.opt_contract_info = self.opt_contract_info[self.opt_contract_info["dominant"] == True]
            self.opt_contract_info = self.opt_contract_info.drop(columns=["dominant"])

    def opt_adjust(self, x):
        if re.search('A', x):
            return 1
        else:
            return 0

    def get_opt_info(self, date):
        self.opt_contract_info.listed_date = pd.to_datetime(self.opt_contract_info.listed_date)
        self.opt_contract_info.de_listed_date = pd.to_datetime(self.opt_contract_info.de_listed_date)
        if self.underlying_symbol.startswith("510"):
            self.opt_contract_info['adjust'] = self.opt_contract_info['product_name'].apply(
                lambda x: self.opt_adjust(x))
        exist_opt = self.opt_contract_info[
            (self.opt_contract_info['listed_date'] <= date) & (self.opt_contract_info['de_listed_date'] >= date)]
        return exist_opt

    def get_underlying(self):
        # underlying_dir = self.underlying_symbol.split(".")[0]
        # current_dir = self.base_dir + underlying_dir + "/"
        if not self.underlying_dir.startswith("510"):
            underlying_fn = UNDERLYING_DICT[self.underlying_dir]
        else:
            underlying_fn = self.underlying_dir
        underlying_info = pd.read_csv(self.underlying_base_dir + underlying_fn + ".csv", index_col=0)
        underlying_info.index = pd.to_datetime(underlying_info.index)
        return underlying_info

    def get_underlying_data(self, order_book_id):
        """
        Get underlying order book id
        """

        if ("." in order_book_id):
            underlying_filename = order_book_id.split(".")[0] + ".csv"
        else:
            underlying_filename = order_book_id + ".csv"

        print(underlying_filename)
        underlying_data_dir = self.underlying_base_dir + "underlying_price_1d/"
        underlying_data = pd.read_csv(underlying_data_dir + underlying_filename, index_col=0)
        underlying_data.index = pd.to_datetime(underlying_data.index)
        return underlying_data

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
        if self.underlying_symbol.startswith("510"):
            exist_opt = self.get_none_adjust_opt(date)
            exist_opt['product_name'] = exist_opt['product_name'].apply(lambda x: x.replace('A', 'M'))
            exist_opt['strike'] = exist_opt['product_name'].apply(lambda x: int(x[-4:]) / 1000)
        else:
            exist_opt = self.get_opt_info(date)
            exist_opt['strike'] = exist_opt['strike_price']

        # calculate strike and days-to-maturity
        exist_opt['days'] = ((exist_opt['de_listed_date'] - date) / np.timedelta64(1, 'D')).astype(int)
        exist_opt = exist_opt[exist_opt['days'] > self.d]
        exist_opt = exist_opt[exist_opt['days'] == exist_opt['days'].min()]

        # get ETF50 or Future price at that date
        underlying_order_book_id = exist_opt["underlying_order_book_id"].values[0]
        underlying_data = self.get_underlying_data(underlying_order_book_id)
        underlying_price = underlying_data[underlying_data.index == date]['close'].values[0]

        # get the minimum difference between strike and ETF50 price
        opt_c, opt_p = [], []
        # OTM
        if posit > 0:
            for d, f in exist_opt.groupby('days'):
                f['diff'] = f['strike'].apply(lambda x: x - underlying_price)
                f['abs_diff'] = abs(f['diff'])
                f_c = f[(f['option_type'] == 'C') & (f['diff'] >= 0)]
                f_p = f[(f['option_type'] == 'P') & (f['diff'] <= 0)]
                if f_c.empty:
                    opt_c = opt_c + f[(f['option_type'] == 'C') & (f['abs_diff'] == min(f['abs_diff']))][
                        'product_name'].values.tolist()
                elif f_c.shape[0] < posit:
                    c_abs_diff = max(f_c['abs_diff'].values)
                    opt_c = opt_c + f_c[f_c['abs_diff'] == c_abs_diff]['product_name'].values.tolist()
                else:
                    c_abs_diff = sorted(f_c['abs_diff'].values)[posit - 1]
                    opt_c = opt_c + f_c[f_c['abs_diff'] == c_abs_diff]['product_name'].values.tolist()
                if f_p.empty:
                    opt_p = opt_p + f[(f['option_type'] == 'P') & (f['abs_diff'] == min(f['abs_diff']))][
                        'product_name'].values.tolist()
                elif f_p.shape[0] < posit:
                    p_abs_diff = max(f_p['abs_diff'].values)
                    opt_p = opt_p + f_p[f_p['abs_diff'] == p_abs_diff]['product_name'].values.tolist()
                else:
                    p_abs_diff = sorted(f_p['abs_diff'].values)[posit - 1]
                    opt_p = opt_p + f_p[f_p['abs_diff'] == p_abs_diff]['product_name'].values.tolist()
        # ITM
        else:
            for d, f in exist_opt.groupby('days'):
                f['diff'] = f['strike'].apply(lambda x: x - underlying_price)
                f['abs_diff'] = abs(f['diff'])
                f_c = f[(f['option_type'] == 'C') & (f['diff'] <= 0)]
                f_p = f[(f['option_type'] == 'P') & (f['diff'] >= 0)]
                if f_c.empty:
                    opt_c = opt_c + f[(f['option_type'] == 'C') & (f['abs_diff'] == min(f['abs_diff']))][
                        'product_name'].values.tolist()
                elif f_c.shape[0] < -posit:
                    c_abs_diff = max(f_c['abs_diff'].values)
                    opt_c = opt_c + f_c[f_c['abs_diff'] == c_abs_diff]['product_name'].values.tolist()
                else:
                    c_abs_diff = sorted(f_c['abs_diff'].values)[-posit - 1]
                    opt_c = opt_c + f_c[f_c['abs_diff'] == c_abs_diff]['product_name'].values.tolist()
                if f_p.empty:
                    opt_p = opt_p + f[(f['option_type'] == 'P') & (f['abs_diff'] == min(f['abs_diff']))][
                        'product_name'].values.tolist()
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

    def get_delta(self, date, opt_symbol):
        greek_base_dir = self.underlying_base_dir + "opt_greeks_1d/"
        opt_greeks = pd.read_csv(greek_base_dir + opt_symbol + ".csv").set_index("trading_date")
        opt_greeks.index = pd.to_datetime(opt_greeks.index)
        delta = opt_greeks[opt_greeks.index == date]["delta"].values[0]
        return delta

    def delta_opt(self, date, delta_level):
        """
        find options by considering delta
        """
        # get rid of adjusted options
        if self.underlying_symbol.startswith("510"):
            exist_opt = self.get_none_adjust_opt(date)
            exist_opt['product_name'] = exist_opt['product_name'].apply(lambda x: x.replace('A', 'M'))
            exist_opt['strike'] = exist_opt['product_name'].apply(lambda x: int(x[-4:]) / 1000)
        else:
            exist_opt = self.get_opt_info(date)
            exist_opt['strike'] = exist_opt['strike_price']

        # calculate strike and days-to-maturity
        exist_opt['days'] = ((exist_opt['de_listed_date'] - date) / np.timedelta64(1, 'D')).astype(int)
        exist_opt = exist_opt[exist_opt['days'] > self.d]
        exist_opt = exist_opt[exist_opt['days'] == exist_opt['days'].min()]

        # get ETF50 price at that date
        # underlying_info = self.get_underlying()
        # underlying_price = underlying_info[underlying_info.index == date]['open'].values[0]
        delta_table = []
        for symbol in exist_opt["order_book_id"].values:
            delta = self.get_delta(date, symbol)
            delta_table.append([symbol, delta])
        delta_table = pd.DataFrame(delta_table, columns=["order_book_id", "delta"])
        delta_table["diff"] = delta_table["delta"] - delta_level
        delta_table["abs_diff"] = abs(delta_table["diff"])
        opt_symbol = delta_table[delta_table["abs_diff"] == delta_table["abs_diff"].min()]["order_book_id"].values[0]

        tradable_opt = exist_opt[exist_opt["order_book_id"] == opt_symbol]
        ttm = tradable_opt["days"].values[0]
        return tradable_opt, ttm
