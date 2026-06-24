from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .utils import ensure_dir


@dataclass
class PreparedDataset:
    feature_frame: pd.DataFrame
    benchmark_frame: pd.DataFrame
    industries: pd.DataFrame


def _standardize_columns(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    existing_map = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=existing_map)


def load_raw_data(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(config.raw_dir)
    industries = pd.read_csv(raw_dir / "sw_l1_industries.csv")
    sw_daily = pd.read_csv(raw_dir / "sw_daily.csv")
    hs300 = pd.read_csv(raw_dir / "hs300_daily.csv")
    return industries, sw_daily, hs300


def build_feature_frame(config: ProjectConfig) -> PreparedDataset:
    industries, sw_daily, hs300 = load_raw_data(config)

    sw_daily = _standardize_columns(
        sw_daily,
        {
            "index_code": "ts_code",
            "trade_date": "trade_date",
            "pct_chg": "pct_chg",
            "pct_change": "pct_chg",
            "amount": "amount",
            "vol": "vol",
            "close": "close",
            "pre_close": "pre_close",
        },
    )
    hs300 = _standardize_columns(
        hs300,
        {
            "ts_code": "ts_code",
            "trade_date": "trade_date",
            "pct_chg": "pct_chg",
            "pct_change": "pct_chg",
            "close": "close",
        },
    )

    sw_daily["trade_date"] = pd.to_datetime(sw_daily["trade_date"])
    hs300["trade_date"] = pd.to_datetime(hs300["trade_date"])
    sw_daily = sw_daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    hs300 = hs300.sort_values("trade_date").reset_index(drop=True)

    code_col = "index_code" if "index_code" in industries.columns else "ts_code"
    name_candidates = ["industry_name", "行业名称", "name", "industry"]
    name_col = next((col for col in name_candidates if col in industries.columns), None)
    if name_col is not None and config.core_industries:
        industries = industries[industries[name_col].isin(config.core_industries)].copy()
        selected_codes = industries[code_col].dropna().unique().tolist()
        sw_daily = sw_daily[sw_daily["ts_code"].isin(selected_codes)].copy()

    sw_daily["ret_1d"] = sw_daily.groupby("ts_code")["close"].pct_change()
    sw_daily["amount"] = sw_daily["amount"].astype(float)
    sw_daily["amount_chg"] = sw_daily.groupby("ts_code")["amount"].pct_change().replace([np.inf, -np.inf], np.nan)

    for window in config.feature_lookbacks:
        sw_daily[f"mom_{window}"] = sw_daily.groupby("ts_code")["close"].pct_change(window)
        sw_daily[f"ret_std_{window}"] = (
            sw_daily.groupby("ts_code")["ret_1d"].rolling(window).std().reset_index(level=0, drop=True)
        )
        sw_daily[f"amt_avg_{window}"] = (
            sw_daily.groupby("ts_code")["amount"].rolling(window).mean().reset_index(level=0, drop=True)
        )

    short_w, long_w = min(config.feature_lookbacks), max(config.feature_lookbacks)
    sw_daily["ma_short"] = sw_daily.groupby("ts_code")["close"].rolling(short_w).mean().reset_index(level=0, drop=True)
    sw_daily["ma_long"] = sw_daily.groupby("ts_code")["close"].rolling(long_w).mean().reset_index(level=0, drop=True)
    sw_daily["ma_gap"] = sw_daily["ma_short"] / sw_daily["ma_long"] - 1.0
    sw_daily["amount_z20"] = sw_daily.groupby("ts_code")["amount"].transform(
        lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()
    )

    benchmark = hs300[["trade_date", "close"]].copy()
    benchmark["benchmark_ret_1d"] = benchmark["close"].pct_change()
    benchmark[f"benchmark_fwd_{config.horizon}"] = benchmark["close"].shift(-config.horizon) / benchmark["close"] - 1.0

    sw_daily = sw_daily.merge(
        benchmark[["trade_date", "benchmark_ret_1d", f"benchmark_fwd_{config.horizon}"]],
        on="trade_date",
        how="left",
    )
    sw_daily["excess_ret_1d"] = sw_daily["ret_1d"] - sw_daily["benchmark_ret_1d"]
    sw_daily[f"fwd_ret_{config.horizon}"] = sw_daily.groupby("ts_code")["close"].shift(-config.horizon) / sw_daily["close"] - 1.0
    sw_daily["future_excess_ret"] = sw_daily[f"fwd_ret_{config.horizon}"] - sw_daily[f"benchmark_fwd_{config.horizon}"]
    sw_daily["target"] = sw_daily["future_excess_ret"]

    sw_daily["ret_rank_pct"] = sw_daily.groupby("trade_date")["ret_1d"].rank(pct=True)
    sw_daily["excess_rank_pct"] = sw_daily.groupby("trade_date")["excess_ret_1d"].rank(pct=True)
    sw_daily[f"mom_rank_pct_{short_w}"] = sw_daily.groupby("trade_date")[f"mom_{short_w}"].rank(pct=True)
    sw_daily[f"mom_rank_pct_{long_w}"] = sw_daily.groupby("trade_date")[f"mom_{long_w}"].rank(pct=True)
    sw_daily["amount_rank_pct"] = sw_daily.groupby("trade_date")["amount_chg"].rank(pct=True)
    sw_daily["vol_rank_pct"] = sw_daily.groupby("trade_date")[f"ret_std_{long_w}"].rank(pct=True)

    if config.label_epsilon > 0:
        sw_daily = sw_daily[sw_daily["future_excess_ret"].abs() >= config.label_epsilon].copy()

    feature_columns = [
        "ret_1d",
        "excess_ret_1d",
        f"ret_std_{short_w}",
        f"ret_std_{long_w}",
        "amount_chg",
        "amount_z20",
        f"mom_{short_w}",
        f"mom_{long_w}",
        "ma_gap",
        "ret_rank_pct",
        "excess_rank_pct",
        f"mom_rank_pct_{short_w}",
        f"mom_rank_pct_{long_w}",
        "amount_rank_pct",
        "vol_rank_pct",
    ]
    keep_columns = ["ts_code", "trade_date", "close", "amount", "target"] + feature_columns
    feature_frame = sw_daily[keep_columns].copy()

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.dropna().reset_index(drop=True)

    return PreparedDataset(
        feature_frame=feature_frame,
        benchmark_frame=benchmark,
        industries=industries,
    )


def save_feature_frame(dataset: PreparedDataset, config: ProjectConfig) -> Path:
    processed_dir = ensure_dir(config.processed_dir)
    output_path = processed_dir / "features.csv"
    dataset.feature_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    dataset.industries.to_csv(processed_dir / "industries.csv", index=False, encoding="utf-8-sig")
    return output_path
