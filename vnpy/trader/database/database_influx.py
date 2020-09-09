from datetime import datetime
from typing import Optional, Sequence, List

from influxdb import InfluxDBClient, DataFrameClient

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.utility import generate_vt_symbol

from .database import BaseDatabaseManager, Driver, DB_TZ
import pandas as pd

# 代码改动点：
# 1. 存放和读取数据，全部改为了pandas批量化读取，并且用apply变为进行BarData和原始数据类型的转换，减少了至少4个for循环
# 2. 为了便于检查数据库存放的数据，取消了DB_TZ 时区参数，目前读取数据的时区，将不受外部传入DB_TZ时区参数的影响，默认为UTC时区

influx_database = ""
influx_client = None
dfclient_database = ""
dfclient=None

def init(_: Driver, settings: dict):
    database = settings["database"]
    host = settings["host"]
    port = settings["port"]
    username = settings["user"]
    password = settings["password"]

    if not username:  # if username == '' or None, skip username
        username = "root"
        password = "root"

    global influx_client
    global influx_database
    global dfclient
    global dfclient_database

    influx_database = database
    influx_client = InfluxDBClient(host, port, username, password, database)
    influx_client.create_database(database)

    dfclient_database = database
    dfclient = DataFrameClient(host, port, username, password, database)
    dfclient.create_database(database)
    
    return InfluxManager()

def row_to_bar(df: pd.DataFrame):
    """
    Generate BarData object from each row of input DataFrame
    """
    bar = BarData(
        symbol=df['symbol'],
        exchange=df['exchange'],
        # 在经过apply之后，datetime字段的类型，由之前的datetime变为了Timestamp，
        # 如果想再变为datetime类型，可以让df['datetime']再调用.to_pydatetime()
        # datetime=df.name.to_pydatetime().replace(tzinfo=DB_TZ),
        datetime=df.name.to_pydatetime(),
        interval=Interval(df['interval']),
        volume=df['volume'],
        open_price=df['open_price'],
        high_price=df['high_price'],
        open_interest=df['open_interest'],
        low_price=df['low_price'],
        close_price=df['close_price'],
        gateway_name="DB",
    )
    return bar

def bar_to_row(df: pd.Series):
    """
    将pd.Series存放的BarData转换为pd.DataFrame（按行操作）
    """
    BarData = df.array[0]

    price_dict = {'datetime': BarData.datetime,
    # price_dict = {'datetime': BarData.datetime.replace(tzinfo=DB_TZ),
                    'vt_symbol': generate_vt_symbol(BarData.symbol, BarData.exchange),
                    'interval': BarData.interval.value,
                    'open_price': BarData.open_price,
                    'high_price': BarData.high_price,
                    'low_price': BarData.low_price,
                    'close_price': BarData.close_price,
                    'volume': BarData.volume,
                    'open_interest': BarData.open_interest}

    return pd.Series(price_dict)

class InfluxManager(BaseDatabaseManager):

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> Sequence[BarData]:

        if isinstance(start, datetime):
            start = start.date()

        if isinstance(end, datetime):
            end = end.date()

        query = (
            "select * from bar_data"
            " where vt_symbol=$vt_symbol"
            " and interval=$interval"
            f" and time >= '{start.isoformat()}'"
            f" and time <= '{end.isoformat()}';"
        )

        bind_params = {
            "vt_symbol": generate_vt_symbol(symbol, exchange),
            "interval": interval.value
        }

        df = dfclient.query(query,
                            bind_params=bind_params,
                            chunked=True,         # 使用chunked=True可以避免out-of-memory(OOM)
                            chunk_size=20000,     # 在没有out-of-memory(OOM)时，可以尽可能的大一些。
                            )

        bar_df = df['bar_data']

        bar_df['exchange'] = exchange
        bar_df['symbol'] = symbol
        # bar_df.index = bar_df.index.tz_convert(DB_TZ)
        bar_type_df = bar_df.apply(row_to_bar, axis=1)

        return bar_type_df.tolist()

    def load_tick_data(
        self, symbol: str, exchange: Exchange, start: datetime, end: datetime
    ) -> Sequence[TickData]:
        pass

    def save_bar_data(self, data: Sequence[BarData]):
        # data_df里每行存放的是BarData
        data_df = pd.DataFrame(data)

        # new_df里存放的是原始数据类型,调用bar_to_row函数，返回原始的数据类型
        new_df = data_df.apply(bar_to_row, axis=1)

        # new_df['datetime'] = pd.to_datetime(new_df['datetime'])

        # new_df['datetime'] = new_df['datetime'].tz_convert(DB_TZ)

        # 放入influxDB数据库前，必须保证index为time类型
        # 该步骤set_index会使得datetime的时区错乱，因此需要额外调整时区
        new_df.set_index('datetime', inplace=True)

        # new_df.index = new_df.index.tz_localize(DB_TZ)

        dfclient.write_points(dataframe=new_df,
                              measurement="bar_data",
                              tag_columns=["vt_symbol", "interval"],
                              time_precision='s',    # 在influxDB中存在最粗粒度的时间精度可以显著提高数据压缩效率
                              numeric_precision=2,   # 价格数据保留小数点两位即可
                              )

    def save_tick_data(self, df: pd.DataFrame):
        """
        必须保证数值数据写入的类型为float，否则写入后会自动创建新类型，导致数据库类型错乱
        """

        dfclient.write_points(dataframe=df,
                              measurement="tick_data")

        pass

        json_body = []

        tick = data[0]
        vt_symbol = tick.vt_symbol
        
        for tick in data:
            dt = tick.datetime.astimezone(DB_TZ)
            dt = dt.replace(tzinfo=None)

            d = {
                "measurement": "tick_data",
                "tags":{
                    "vt_symbol": vt_symbol,
                },
                "time": dt.isoformat(),
                "fields": {
                    #"exchange": tick.exchange.value,
                    "bid_price_1": tick.bid_price_1,
                    "bid_volume_1": tick.bid_volume_1,
                    "ask_price_1": tick.ask_price_1,
                    "ask_volume_1": tick.ask_volume_1,
                    "open_price": tick.open_price,
                    "high_price": tick.high_price,
                    "low_price": tick.low_price,
                    "last_price": tick.last_price,
                    "volume": tick.volume,
                    "open_interest": tick.open_interest,
                    #"limit_up": tick.limit_up,
                    #"limit_down": tick.limit_down,
                    #"pre_close": tick.pre_close
                }
            }
            json_body.append(d)

        influx_client.write_points(json_body)

    def get_newest_bar_data(
        self, symbol: str, exchange: "Exchange", interval: "Interval"
    ) -> Optional["BarData"]:
        query = (
            "select last(close_price), * from bar_data"
            " where vt_symbol=$vt_symbol"
            " and interval=$interval"
        )

        bind_params = {
            "vt_symbol": generate_vt_symbol(symbol, exchange),
            "interval": interval.value
        }

        result = influx_client.query(query, bind_params=bind_params)
        points = result.get_points()

        bar = None
        for d in points:
            dt = datetime.strptime(d["time"], "%Y-%m-%dT%H:%M:%SZ")

            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                datetime=dt.replace(tzinfo=DB_TZ),
                open_price=d["open_price"],
                high_price=d["high_price"],
                low_price=d["low_price"],
                close_price=d["close_price"],
                volume=d["volume"],
                open_interest=d["open_interest"],
                gateway_name="DB"
            )

        return bar

    def get_oldest_bar_data(
        self, symbol: str, exchange: "Exchange", interval: "Interval"
    ) -> Optional["BarData"]:
        query = (
            "select first(close_price), * from bar_data"
            " where vt_symbol=$vt_symbol"
            " and interval=$interval"
        )

        bind_params = {
            "vt_symbol": generate_vt_symbol(symbol, exchange),
            "interval": interval.value
        }

        result = influx_client.query(query, bind_params=bind_params)
        points = result.get_points()

        bar = None
        for d in points:
            dt = datetime.strptime(d["time"], "%Y-%m-%dT%H:%M:%SZ")

            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                datetime=dt.replace(tzinfo=DB_TZ),
                open_price=d["open_price"],
                high_price=d["high_price"],
                low_price=d["low_price"],
                close_price=d["close_price"],
                volume=d["volume"],
                open_interest=d["open_interest"],
                gateway_name="DB"
            )

        return bar

    def get_newest_tick_data(
        self, symbol: str, exchange: "Exchange"
    ) -> Optional["TickData"]:
        pass

    def get_bar_data_statistics(self) -> List:
        query = "select count(close_price) from bar_data group by *"
        result = influx_client.query(query)

        r = []
        for k, v in result.items():
            tags = k[1]
            statistics = list(v)[0]

            symbol, exchange = tags["vt_symbol"].split(".")

            r.append({
                "symbol": symbol,
                "exchange": exchange,
                "interval": tags["interval"],
                "count": statistics["count"]
            })

        return r

    def delete_bar_data(
        self,
        symbol: str,
        exchange: "Exchange",
        interval: "Interval"
    ) -> int:
        """
        Delete all bar data with given symbol + exchange + interval.
        """
        bind_params = {
            "vt_symbol": generate_vt_symbol(symbol, exchange),
            "interval": interval.value
        }

        # Query data count
        query1 = (
            "select count(close_price) from bar_data"
            " where vt_symbol=$vt_symbol"
            " and interval=$interval"
        )
        result = influx_client.query(query1, bind_params=bind_params)
        points = result.get_points()

        for d in points:
            count = d["count"]

        # Delete data
        query2 = (
            "drop series from bar_data"
            " where vt_symbol=$vt_symbol"
            " and interval=$interval"
        )
        influx_client.query(query2, bind_params=bind_params)

        return count

    def clean(self, symbol: str):
        pass
