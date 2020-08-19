from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Set, Tuple
from functools import lru_cache
import traceback
from dataclasses import dataclass
import pandas as pd
import re
import os
from .option_data import DataImport

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from vnpy.trader.constant import Direction, Offset, Interval, Status
from vnpy.trader.option_database import database_manager
from vnpy.trader.option_object import OrderData, TradeData, BarData
from vnpy.trader.option_utility import round_to, extract_vt_symbol

from .template import StrategyTemplate
from .backtesting import BacktestingEngine, ContractDailyResult, PortfolioDailyResult

# Set seaborn style
sns.set_style("whitegrid")

INTERVAL_DELTA_MAP = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}

OPTION_EXCHANGE_DICT = {
    "510050": "SHFE", "510300": "SHFE",
    "CU": "SHFE", "AU": "SHFE", "RU": "SHFE",
    "C": "DCE", "M": "DCE", "I": "DCE", "PG": "DCE",
    "MA": "CFFEX", "CF": "CFFEX", "RM": "CFFEX", "SR": "CFFEX", "TA": "CFFEX"
}


@dataclass
class OptionBacktestingInfo:
    """
    record underlying, base_dir, exchange_days, posit or delta_level, dominant and columns 
    """

    underlying: str
    base_dir: str
    exchange_days: int = 10
    option_select: Tuple = ("posit", 1)
    dominant: bool = True
    columns: Tuple = None


@dataclass
class OptionTradeInfo:
    """
    record 
    rate, slippage, size, and pricetick
    """

    rate: float = 0
    slippage: float = 0
    size: int = 1
    pricetick: float = 0


class OptionBacktestingEngine(BacktestingEngine):
    """"""

    gateway_name = "OPTION_BACKTESTING"

    def __init__(self, option_backtesting_info, option_trade_info):
        """"""
        super().__init__()
        self.option_select = option_backtesting_info.option_select
        self.pre_opt_symbols = []
        self.underlying = option_backtesting_info.underlying
        self.dominant = option_backtesting_info.dominant
        self.option_exchange = OPTION_EXCHANGE_DICT[self.underlying]
        self.base_dir = option_backtesting_info.base_dir

        self.opt_info = DataImport(option_backtesting_info.exchange_days, self.underlying, self.base_dir, self.dominant)
        self.opt_data = DataImport(0, self.underlying, self.base_dir, self.dominant)
        self.option_trade_info = option_trade_info
        self.columns = option_backtesting_info.columns
        database_manager.set_columns(self.columns)

    def set_parameters(
            self,
            vt_symbols: List[str],
            interval: Interval,
            start: datetime,
            rates: Dict[str, float],
            slippages: Dict[str, float],
            sizes: Dict[str, float],
            priceticks: Dict[str, float],
            capital: int = 0,
            end: datetime = None
    ) -> None:
        """"""
        self.vt_symbols = vt_symbols
        self.interval = interval

        self.rates = rates
        self.slippages = slippages
        self.sizes = sizes
        self.priceticks = priceticks

        self.start = start
        self.end = end
        self.capital = capital

        # add basic vt symbols
        self.basic_vt_symbols = vt_symbols

    # override the load_data() in backtesting
    def load_data(self) -> None:
        """"""
        self.output("开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return

        # Clear previously loaded history data
        self.history_data.clear()
        self.dts.clear()

        # Load 30 days of data each time and allow for progress update
        progress_delta = timedelta(days=30)
        total_delta = self.end - self.start
        interval_delta = INTERVAL_DELTA_MAP[self.interval]

        for vt_symbol in self.vt_symbols:
            start = self.start
            end = self.start + progress_delta
            progress = 0

            data_count = 0
            while start < self.end:
                end = min(end, self.end)  # Make sure end time stays within set range

                data = load_bar_data(
                    vt_symbol,
                    self.interval,
                    start,
                    end,
                    self.columns,
                )

                for bar in data:
                    self.dts.add(bar.datetime)
                    self.history_data[(bar.datetime, vt_symbol)] = bar
                    data_count += 1

                progress += progress_delta / total_delta
                progress = min(progress, 1)
                # progress_bar = "#" * int(progress * 10)
                # self.output(f"{vt_symbol}加载进度：{progress_bar} [{progress:.0%}]")

                start = end + interval_delta
                end += (progress_delta + interval_delta)

            self.output(f"{vt_symbol}历史数据加载完成，数据量：{data_count}")

        self.output("所有历史数据加载完成")

    # load option data on_bars() function
    def load_data_new_bars(self, opt_symbols):
        self.output("load_data_new_bars开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return

        # Load 30 days of data each time and allow for progress update
        progress_delta = timedelta(days=30)
        interval_delta = INTERVAL_DELTA_MAP[self.interval]

        for vt_symbol in opt_symbols:
            start = self.start
            end = self.start + progress_delta

            data_count = 0
            while start < self.end:
                end = min(end, self.end)  # Make sure end time stays within set range

                data = load_bar_data(
                    vt_symbol,
                    self.interval,
                    start,
                    end,
                    self.columns,
                )

                for bar in data:
                    self.dts.add(bar.datetime)
                    self.history_data[(bar.datetime, vt_symbol)] = bar
                    data_count += 1

                start = end + interval_delta
                end += (progress_delta + interval_delta)

            self.output(f"{vt_symbol}历史数据加载完成，数据量：{data_count}")

    def new_bars(self, dt: datetime) -> None:
        """"""
        self.datetime = dt

        self.bars.clear()

        date = pd.to_datetime(str(dt)[:10])
        self.select_type, self.select_value = self.option_select[0], self.option_select[1]
        if self.select_type == "posit":
            tradable_opt, ttm = self.opt_info.available_opt(date, self.select_value)
        elif self.select_type == "delta":
            tradable_opt, ttm = self.opt_info.delta_opt(date, self.select_value)

        opt_symbols = []
        for opt_symbol in tradable_opt["order_book_id"].values:
            opt_symbols.append(str(opt_symbol) + "." + self.option_exchange)

        new_symbols = list(set(opt_symbols) - set(self.vt_symbols))
        if len(new_symbols) > 0:
            for vt_symbol in new_symbols:
                self.rates[vt_symbol] = self.option_trade_info.rate
                self.slippages[vt_symbol] = self.option_trade_info.slippage
                self.sizes[vt_symbol] = self.option_trade_info.size
                self.priceticks[vt_symbol] = self.option_trade_info.pricetick
            self.vt_symbols = self.vt_symbols + new_symbols
            self.load_data_new_bars(new_symbols)
        exsit_opt = self.opt_data.get_opt_info(date)
        trade_symbols = [trade.vt_symbol for trade in self.trades.values()] + opt_symbols + self.pre_opt_symbols
        exist_symbols = list(set(self.vt_symbols).intersection(
            set(list(exsit_opt["order_book_id"].apply(lambda x: str(x) + "." + self.option_exchange).values))))
        exist_symbols = exist_symbols + self.basic_vt_symbols
        for vt_symbol in exist_symbols:
            bar = self.history_data.get((dt, vt_symbol), None)
            if bar:
                self.bars[vt_symbol] = bar
            else:
                dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                self.output(f"数据缺失：{dt_str} {vt_symbol}")

        self.pre_opt_symbols = opt_symbols
        self.cross_limit_order()
        self.strategy.on_bars(self.bars, tradable_opt, ttm, self.option_exchange)

        self.update_daily_close(self.bars, dt)

    def cross_limit_order(self) -> None:
        """
        Cross limit order with last bar/tick data.
        """
        for order in list(self.active_limit_orders.values()):
            bar = self.bars[order.vt_symbol]

            long_cross_price = bar.low_price
            short_cross_price = bar.high_price
            long_best_price = bar.open_price
            short_best_price = bar.open_price

            # Push order update with status "not traded" (pending).
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.update_order(order)

            # Check whether limit orders can be filled.
            long_cross = (
                    order.direction == Direction.LONG
                    and order.price >= long_cross_price
                    and long_cross_price > 0
            )

            short_cross = (
                    order.direction == Direction.SHORT
                    and order.price <= short_cross_price
                    and short_cross_price > 0
            )

            if not long_cross and not short_cross:
                continue

            # Push order update with status "all traded" (filled).
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.update_order(order)

            self.active_limit_orders.pop(order.vt_orderid)

            # Push trade update
            self.trade_count += 1

            if long_cross:
                trade_price = min(order.price, long_best_price)
            else:
                trade_price = max(order.price, short_best_price)

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )
            trade.datetime = self.datetime

            self.strategy.update_trade(trade)
            self.trades[trade.vt_tradeid] = trade


@lru_cache(maxsize=999)
def load_bar_data(
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
        columns=None,
):
    """"""
    symbol, exchange = extract_vt_symbol(vt_symbol)

    return database_manager.load_bar_data(
        symbol, exchange, interval, start, end, columns,
    )
