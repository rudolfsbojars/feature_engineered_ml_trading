
from binance_historical_data import BinanceDataDumper

def import_binance_data():
    dumper = BinanceDataDumper(
        path_dir_where_to_dump="data/raw",
        asset_class="spot",
        data_type="klines",
        data_frequency="1m",
    )

    dumper.dump_data(
        tickers=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        date_start=None,
        is_to_update_existing=False
    )
    