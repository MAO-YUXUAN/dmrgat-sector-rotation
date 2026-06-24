from __future__ import annotations

from pathlib import Path

import akshare as ak
import pandas as pd

from .config import ProjectConfig
from .utils import ensure_dir


def fetch_sw_industries() -> pd.DataFrame:
    industries = ak.sw_index_first_info()
    if industries is None or industries.empty:
        raise RuntimeError("AKShare sw_index_first_info returned no rows.")
    industries = industries.rename(
        columns={
            "行业代码": "index_code",
            "行业名称": "industry_name",
        }
    )
    return industries


def fetch_sw_daily(industry_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date, format="%Y%m%d")
    end_ts = pd.to_datetime(end_date, format="%Y%m%d")
    frames: list[pd.DataFrame] = []

    for index_code in industry_codes:
        symbol = index_code.split(".")[0]
        df = ak.index_hist_sw(symbol=symbol, period="day")
        if df is None or df.empty:
            continue
        df = df.rename(
            columns={
                "代码": "ts_code",
                "日期": "trade_date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "vol",
                "成交额": "amount",
            }
        )
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= start_ts) & (df["trade_date"] <= end_ts)].copy()
        if df.empty:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["ts_code"] = index_code
        df["pre_close"] = df["close"].shift(1)
        df["change"] = df["close"] - df["pre_close"]
        df["pct_chg"] = df["close"].pct_change() * 100.0
        frames.append(
            df[
                [
                    "ts_code",
                    "trade_date",
                    "open",
                    "low",
                    "high",
                    "close",
                    "pre_close",
                    "change",
                    "pct_chg",
                    "vol",
                    "amount",
                ]
            ]
        )

    if not frames:
        raise RuntimeError("AKShare index_hist_sw returned no usable rows for the SW level-1 universe.")
    return pd.concat(frames, ignore_index=True)


def fetch_hs300_daily(benchmark_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    symbol = f"sh{benchmark_code.split('.')[0]}"
    start_ts = pd.to_datetime(start_date, format="%Y%m%d")
    end_ts = pd.to_datetime(end_date, format="%Y%m%d")
    df = ak.stock_zh_index_daily(symbol=symbol)
    if df is None or df.empty:
        raise RuntimeError(f"AKShare stock_zh_index_daily returned no rows for benchmark {benchmark_code}.")
    df = df.rename(columns={"date": "trade_date", "volume": "vol"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[(df["trade_date"] >= start_ts) & (df["trade_date"] <= end_ts)].copy()
    df["ts_code"] = benchmark_code
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["pre_close"] = df["close"].shift(1)
    df["change"] = df["close"] - df["pre_close"]
    df["pct_chg"] = df["close"].pct_change() * 100.0
    df["amount"] = pd.NA
    keep_cols = [
        "ts_code",
        "trade_date",
        "open",
        "low",
        "high",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]
    return df[keep_cols]


def download_all(config: ProjectConfig) -> dict[str, Path]:
    raw_dir = ensure_dir(config.raw_dir)

    industries = fetch_sw_industries()
    industry_codes = sorted(industries["index_code"].dropna().unique().tolist())
    sw_daily = fetch_sw_daily(industry_codes, config.start_date, config.end_date)
    hs300 = fetch_hs300_daily(config.benchmark_code, config.start_date, config.end_date)

    industry_path = raw_dir / "sw_l1_industries.csv"
    sw_daily_path = raw_dir / "sw_daily.csv"
    hs300_path = raw_dir / "hs300_daily.csv"

    industries.to_csv(industry_path, index=False, encoding="utf-8-sig")
    sw_daily.to_csv(sw_daily_path, index=False, encoding="utf-8-sig")
    hs300.to_csv(hs300_path, index=False, encoding="utf-8-sig")

    return {
        "industries": industry_path,
        "sw_daily": sw_daily_path,
        "hs300_daily": hs300_path,
    }
