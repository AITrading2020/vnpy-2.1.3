from datetime import timedelta
from typing import List, Optional
from pytz import timezone

from numpy import ndarray
from rqdatac import init as rqdata_init
from rqdatac.services.basic import all_instruments as rqdata_all_instruments
from rqdatac.services.get_price import get_price as rqdata_get_price
from rqdatac.share.errors import AuthenticationFailed

from .setting import SETTINGS
from .constant import Exchange, Interval
from .object import BarData, HistoryRequest
import pandas as pd

# 代码改动点：
# 1. 存放和读取数据，从rqdata获取的数据依旧保持pandas格式，并且用apply变为进行BarData和原始数据类型的转换，减少了至少2个for循环
# 2. 为了便于检查数据库存放的数据，取消了CHINA_TZ 时区参数，目前读取数据的时区，将不受外部传入CHINA_TZ时区参数的影响，默认为UTC时区
# 3. 具体改动的函数为query_history函数

INTERVAL_VT2RQ = {
    Interval.MINUTE: "1m",
    Interval.HOUR: "60m",
    Interval.DAILY: "1d",
}

INTERVAL_ADJUSTMENT_MAP = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta()         # no need to adjust for daily bar
}

CHINA_TZ = timezone("Asia/Shanghai")

def to_bar(df: pd.DataFrame):
    """
    Generate BarData object from each row of input DataFrame
    相比官方版，新增函数
    """
    bar = BarData(
        symbol=df['symbol'],
        exchange=df['exchange'],
        interval=df['interval'],
        # 在经过apply之后，datetime字段的类型，由之前的datetime变为了Timestamp，
        # 如果想再变为datetime类型，可以让df['datetime']再调用.to_pydatetime()
        datetime=df['datetime'].to_pydatetime(),
        open_price=df['open'],
        high_price=df['high'],
        low_price=df['low'],
        close_price=df['close'],
        volume=df['volume'],
        open_interest=df['open_interest'],
        gateway_name="RQ",
    )
    return bar

class RqdataClient:
    """
    Client for querying history data from RQData.
    """

    def __init__(self):
        """"""
        self.username: str = SETTINGS["rqdata.username"]
        self.password: str = SETTINGS["rqdata.password"]

        self.inited: bool = False
        self.symbols: ndarray = None

    def init(self, username: str = "", password: str = "") -> bool:
        """"""
        if self.inited:
            return True

        if username and password:
            self.username = username
            self.password = password

        if not self.username or not self.password:
            return False

        try:
            rqdata_init(
                self.username,
                self.password,
                ("rqdatad-pro.ricequant.com", 16011),
                use_pool=True,
                max_pool_size=1
            )

            df = rqdata_all_instruments()
            self.symbols = df["order_book_id"].values
        except (RuntimeError, AuthenticationFailed):
            return False

        self.inited = True
        return True

    def to_rq_symbol(self, symbol: str, exchange: Exchange) -> str:
        """
        CZCE product of RQData has symbol like "TA1905" while
        vt symbol is "TA905.CZCE" so need to add "1" in symbol.
        """
        # Equity
        if exchange in [Exchange.SSE, Exchange.SZSE]:
            if exchange == Exchange.SSE:
                rq_symbol = f"{symbol}.XSHG"
            else:
                rq_symbol = f"{symbol}.XSHE"
        # Futures and Options
        elif exchange in [Exchange.SHFE, Exchange.CFFEX, Exchange.DCE, Exchange.CZCE, Exchange.INE]:
            for count, word in enumerate(symbol):
                if word.isdigit():
                    break

            product = symbol[:count]
            time_str = symbol[count:]

            # Futures
            if time_str.isdigit():
                if exchange is not Exchange.CZCE:
                    return symbol.upper()

                # Check for index symbol
                if time_str in ["88", "888", "99"]:
                    return symbol

                year = symbol[count]
                month = symbol[count + 1:]

                if year == "9":
                    year = "1" + year
                else:
                    year = "2" + year

                rq_symbol = f"{product}{year}{month}".upper()
            # Options
            else:
                if exchange in [Exchange.CFFEX, Exchange.DCE, Exchange.SHFE]:
                    rq_symbol = symbol.replace("-", "").upper()
                elif exchange == Exchange.CZCE:
                    year = symbol[count]
                    suffix = symbol[count + 1:]

                    if year == "9":
                        year = "1" + year
                    else:
                        year = "2" + year

                    rq_symbol = f"{product}{year}{suffix}".upper()
        else:
            rq_symbol = f"{symbol}.{exchange.value}"

        return rq_symbol

    def query_history(self, req: HistoryRequest) -> Optional[List[BarData]]:
        """
        Query history bar data from RQData.
        官方版的函数，从RQData里读取的为DataFrame，但为了把每行数据转换为BarData，加入了一个循环，并返回的是list类型
        改进版的函数，在DataFrame内部批量将每行数据转换为BarData，减少了一个循环，并由DataFrame类型转换为list类型返回
        本次改动取消了获取数据的时区标志（原因：1.rqdata获取的均是国内数据，不需要时区也可以；
        2.rqdata获取的数据入库时，在原版的数据库datebase_influx.py中的数据入库时，也将时区信息清空了，
        否则写入influxDB为utc的时间显示，不利于后面数据库的维护和校对）
        """
        if self.symbols is None:
            return None

        symbol = req.symbol
        exchange = req.exchange
        interval = req.interval
        start = req.start
        end = req.end

        rq_symbol = self.to_rq_symbol(symbol, exchange)
        if rq_symbol not in self.symbols:
            return None

        rq_interval = INTERVAL_VT2RQ.get(interval)
        if not rq_interval:
            return None

        # For adjust timestamp from bar close point (RQData) to open point (VN Trader)
        adjustment = INTERVAL_ADJUSTMENT_MAP[interval]

        # For querying night trading period data
        end += timedelta(1)

        # Only query open interest for futures contract
        fields = ["open", "high", "low", "close", "volume"]
        # 如果symbol.isdigit()为false，则一般多为下载的股票类数据，不存在open_interest这个字段,
        # 但由于BarData中存在open_interest这个字段，因此该字段应该用0.0 float类型填充。
        if not symbol.isdigit():
            fields.append("open_interest")

        df = rqdata_get_price(
            rq_symbol,
            frequency=rq_interval,
            fields=fields,
            start_date=start,
            end_date=end,
            adjust_type="none"
        )

        df['symbol'] = symbol
        df['exchange'] = exchange
        df['interval'] = interval
        # 将RQdata里面的时间戳由Bar收盘的时间点转变为VN trader中Bar开盘的时间点,再转换为datetime
        df['datetime'] = (df.index - adjustment)#.tz_localize(CHINA_TZ)
        # 若为非期货品种，open_interest获取值可能为None, 此时应该用0.0 float类型替换
        if 'open_interest' not in fields:
            df['open_interest'] = 0.0

        bar_df = df.apply(to_bar, axis=1)

        return bar_df.tolist()

        # return df.tolist()

    # def query_history(self, req: HistoryRequest) -> Optional[List[BarData]]:
    #     """
    #     Query history bar data from RQData.
    #     """
    #     if self.symbols is None:
    #         return None
    #
    #     symbol = req.symbol
    #     exchange = req.exchange
    #     interval = req.interval
    #     start = req.start
    #     end = req.end
    #
    #     rq_symbol = self.to_rq_symbol(symbol, exchange)
    #     if rq_symbol not in self.symbols:
    #         return None
    #
    #     rq_interval = INTERVAL_VT2RQ.get(interval)
    #     if not rq_interval:
    #         return None
    #
    #     # For adjust timestamp from bar close point (RQData) to open point (VN Trader)
    #     adjustment = INTERVAL_ADJUSTMENT_MAP[interval]
    #
    #     # For querying night trading period data
    #     end += timedelta(1)
    #
    #     # Only query open interest for futures contract
    #     fields = ["open", "high", "low", "close", "volume"]
    #     if not symbol.isdigit():
    #         fields.append("open_interest")
    #
    #     df = rqdata_get_price(
    #         rq_symbol,
    #         frequency=rq_interval,
    #         fields=fields,
    #         start_date=start,
    #         end_date=end,
    #         adjust_type="none"
    #     )
    #
    #     data: List[BarData] = []
    #
    #     if df is not None:
    #         for ix, row in df.iterrows():
    #             dt = row.name.to_pydatetime() - adjustment
    #             dt = CHINA_TZ.localize(dt)
    #
    #             bar = BarData(
    #                 symbol=symbol,
    #                 exchange=exchange,
    #                 interval=interval,
    #                 datetime=dt,
    #                 open_price=row["open"],
    #                 high_price=row["high"],
    #                 low_price=row["low"],
    #                 close_price=row["close"],
    #                 volume=row["volume"],
    #                 open_interest=row.get("open_interest", 0),
    #                 gateway_name="RQ"
    #             )
    #
    #             data.append(bar)
    #
    #     return data


rqdata_client = RqdataClient()
