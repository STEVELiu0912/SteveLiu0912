
# ── Cell #1 (markdown) ──
# 趋势判定：trend = sign(EMA_gap); trend=+1 视为上涨趋势段; trend=-1 视为下跌趋势段

# ── Cell #2 (code) ──
# -*- coding: utf-8 -*-
"""
Reversal models backtest (multi-symbol, daily, open-to-open execution)

Logic
- Use current-trend filter from EMA_gap sign:
    +1 (uptrend)  -> query LONG model (predict "reversal down"); if prob >= thr_long -> SHORT entry
    -1 (downtrend)-> query SHORT model (predict "reversal up");  if prob >= thr_short-> LONG entry
- Enter/exit on next day's OPEN (T+1), hold until opposite model flips the side.
- No lookahead: signals at t are traded at t+1.
- Equal-weight portfolio: average daily returns across all active symbol positions (cash when no positions).

Outputs
- Per code CSV with: total_return, CAGR, Sharpe, MaxDD, WinRate, Trades, AvgRet, LongAvgRet, ShortAvgRet
- Portfolio-wide summary (same metrics) printed and saved.

Requirements: pandas, numpy, lightgbm, pyarrow/fastparquet, ta
"""

import os, glob, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==== User paths (you already defined these; keep them consistent) ====
PARQUET_DIR = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\PARQUET美股\PARQUET"

LONG_TXT  = r"D:\models_generic\long_trend_rev_20250820_172531.txt"
LONG_META = r"D:\models_generic\long_trend_rev_20250820_172531_meta.json"
SHORT_TXT = r"D:\models_generic\short_trend_rev_20250821_112322.txt"
SHORT_META= r"D:\models_generic\short_trend_rev_20250821_112322_meta.json"

# ==== Global feature config (match your feature builder expectations) ====
ANNUALIZE = 252
VOL_COL   = "σ20"        # internal canonical vol feature name
MA_COL    = "ma20_126"
STD_COL   = "std20_126"
MA126, STD126 = MA_COL, STD_COL
PAST      = 10

# ===================== Utilities & feature helpers (self-contained) =====================
def pct_rank_last(s: pd.Series) -> float:
    """Percentile rank of the last value within the window [0,1] (exclude itself)."""
    if s.size < 2: return np.nan
    last = s.iloc[-1]
    rank = (s <= last).sum() - 1
    return rank / (s.size - 1) if s.size > 1 else np.nan

def rolling_linreg_slope(s: pd.Series) -> float:
    """Slope of y ~ a + b*t on the window (t=0..n-1)."""
    y = s.to_numpy()
    n = y.size
    if n < 2 or np.all(np.isnan(y)): return np.nan
    t = np.arange(n, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2: return np.nan
    t = t[mask]; y = y[mask]
    t = t - t.mean()
    y = y - y.mean()
    denom = np.sum(t*t)
    if denom == 0: return np.nan
    return float(np.sum(t*y) / denom)
import os, re, json, numpy as np, pandas as pd, lightgbm as lgb

class LGBBoosterWrapper:
    """Fixed threshold + fixed feature order; 2-col predict_proba."""
    def __init__(self, booster: lgb.Booster, threshold: float, features: list[str]):
        self.booster = booster
        self.threshold = float(threshold)
        self.features = list(features)  # training order

    def build_X(self, df_feat: pd.DataFrame) -> np.ndarray:
        # select in training order; fill missing with NaN
        idx = df_feat.index
        cols = []
        for c in self.features:
            if c in df_feat.columns:
                s = df_feat[c]
            else:
                s = pd.Series(np.nan, index=idx)
            cols.append(s.astype("float32"))
        X = np.column_stack(cols).astype("float32")
        # sanity: dimension match
        exp = self.booster.num_feature()
        if X.shape[1] != exp:
            raise RuntimeError(f"Feature dim mismatch: X has {X.shape[1]}, booster expects {exp}")
        return X

    def predict_proba(self, X_like) -> np.ndarray:
        if isinstance(X_like, pd.DataFrame):
            X = self.build_X(X_like)
        else:
            X = np.asarray(X_like, dtype="float32")
        n_it = getattr(self.booster, "best_iteration", None) or None
        p = self.booster.predict(X, raw_score=False, num_iteration=n_it)
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        return p

def _looks_like_column_i(names: list[str]) -> bool:
    if not names: return True
    return all(re.fullmatch(r"Column_\d+", str(n)) for n in names[:min(5, len(names))])

def load_booster_with_meta(txt_path: str, meta_path: str | None = None) -> LGBBoosterWrapper:
    booster = lgb.Booster(model_file=txt_path)

    # threshold
    thr = 0.5
    feats = None
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            thr = meta.get("threshold") or meta.get("best_threshold") or meta.get("thr") or 0.5
            # accept multiple keys for features
            for k in ("features","feature_cols","feature_order","feat_names"):
                if k in meta and isinstance(meta[k], list) and meta[k]:
                    feats = list(map(str, meta[k]))
                    break
        except Exception:
            pass

    if not feats:
        # fallback to model file only if it's NOT Column_i
        names = booster.feature_name()
        if not names or _looks_like_column_i(names):
            raise ValueError(
                "Cannot determine feature order: meta lacks features and model exposes generic Column_i. "
                "Retrain saving feature list to meta or supply it explicitly."
            )
        feats = list(map(str, names))

    # dedup, keep order
    seen, clean_feats = set(), []
    for c in feats:
        if c not in seen:
            seen.add(c); clean_feats.append(c)

    # optional: assert count
    exp = booster.num_feature()
    if len(clean_feats) != exp:
        # Not fatal, but warn / normalize if needed
        # raise ValueError(...)
        pass

    return LGBBoosterWrapper(booster, float(thr), clean_feats)

# Load models
LONG_MODEL  = load_booster_with_meta(LONG_TXT,  LONG_META)
SHORT_MODEL = load_booster_with_meta(SHORT_TXT, SHORT_META)

long_clf,  long_thr  = LONG_MODEL,  LONG_MODEL.threshold
short_clf, short_thr = SHORT_MODEL, SHORT_MODEL.threshold

feature_cols_long  = LONG_MODEL.features
feature_cols_short = SHORT_MODEL.features
need_cols = sorted(set(feature_cols_long) | set(feature_cols_short))

# ---- Your feature builders (as given/assumed) ----
def add_all_features(df: pd.DataFrame, need_cols: list[str]) -> pd.DataFrame:
    import ta
    eps = 1e-9
    df = df.copy()

    # Basic cleaning
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Returns
    df["ret"]    = df["close"].pct_change()
    df["logret"] = np.log1p(df["ret"])

    # Averages / EMA / slopes
    df["MA20"]   = df["close"].rolling(20).mean()
    df["MA50"]   = df["close"].rolling(50).mean()
    df["MA100"]  = df["close"].rolling(100).mean()
    df["EMA20"]  = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"]  = df["close"].ewm(span=50, adjust=False).mean()

    # Gap & distances
    df["EMA_gap"]       = df["EMA20"] - df["EMA50"]
    df["close_to_MA20"] = df["close"] - df["MA20"]
    df["close_to_MA50"] = df["close"] - df["MA50"]

    # Slopes (both names)
    df["MA20_slope"]   = df["MA20"].diff(5)
    df["slope_ma_20"]  = df["MA20"].diff()
    df["slope_ma_100"] = df["MA100"].diff()
    df["slope50"]      = df["MA50"].diff()

    # TA indicators
    df["RSI14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14)
    df["ATR14"] = atr.average_true_range()

    stoch = ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3)
    df["StochRSI_%K"] = stoch.stochrsi_k()
    df["StochRSI_%D"] = stoch.stochrsi_d()
    df["Stoch_K_minus_D"] = df["StochRSI_%K"] - df["StochRSI_%D"]

    # ADX / DI
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["ADX14"] = adx.adx()
    df["_DIp"]  = adx.adx_pos()
    df["_DIn"]  = adx.adx_neg()
    df["dip_min_din"] = df["_DIp"] - df["_DIn"]
    df["adx_down5"]   = df["ADX14"] - df["ADX14"].shift(5)

    # RSI derivatives
    df["RSI14_slope5"]   = df["RSI14"].diff(5)
    df["RSI14_pct_rank"] = df["RSI14"].rolling(window=20, min_periods=5).apply(pct_rank_last, raw=False)
    rsi_mean20 = df["RSI14"].rolling(20).mean()
    rsi_std20  = df["RSI14"].rolling(20).std(ddof=0)
    df["RSI_sigma"] = (df["RSI14"] - rsi_mean20) / (rsi_std20 + 1e-9)

    # Volatility core (annualized σ20)
    WIN = 20
    df[VOL_COL] = df["ret"].rolling(WIN).std(ddof=0) * np.sqrt(ANNUALIZE)
    df["σ20"] = df[VOL_COL]  # ensure literal column

    # Long windows on VOL_COL
    for lb in (126, 252):
        df[f"ma20_{lb}"]  = df[VOL_COL].rolling(lb).mean()
        df[f"std20_{lb}"] = df[VOL_COL].rolling(lb).std(ddof=0)

    # Short vol stats & transforms
    df["vol_ma20"]  = df[VOL_COL].rolling(WIN).mean()
    df["vol_std20"] = df[VOL_COL].rolling(WIN).std(ddof=0)
    df["vol_ma_past"]     = df[VOL_COL].rolling(PAST, min_periods=PAST).mean()
    df["vol_ma_past_chg"] = df["vol_ma_past"].pct_change()
    df["σ20_z"]     = (df[VOL_COL] - df[MA126]) / (df[STD126] + 1e-9)
    df["vol_of_vol10"] = df[VOL_COL].rolling(10).std(ddof=0)
    df["vol_ratio60"]   = df[VOL_COL] / (df[VOL_COL].rolling(60).median().replace(0, np.nan))

    # Bollinger & Keltner
    ma20_close = df["close"].rolling(WIN).mean()
    sd20_close = df["close"].rolling(WIN).std(ddof=0)
    bb_up = ma20_close + 2.0 * sd20_close
    bb_lo = ma20_close - 2.0 * sd20_close
    df["bb_width"] = (bb_up - bb_lo) / (ma20_close + 1e-9)

    k_up = df["EMA20"] + 2.0 * df["ATR14"]
    k_lo = df["EMA20"] - 2.0 * df["ATR14"]
    df["keltner_w"]   = (2.0 * df["ATR14"]) / (df["EMA20"] + 1e-9)
    df["squeeze_on"]  = ((bb_up < k_up) & (bb_lo > k_lo)).astype("int8")

    # ATR-normalized shapes & proximities
    upper_wick = (df["high"] - np.maximum(df["open"], df["close"])) / (df["ATR14"] + 1e-9)
    lower_wick = (np.minimum(df["open"], df["close"]) - df["low"]) / (df["ATR14"] + 1e-9)
    body       = (np.abs(df["close"] - df["open"])) / (df["ATR14"] + 1e-9)
    df["upper_wick3"] = upper_wick.rolling(3).mean()
    df["lower_wick3"] = lower_wick.rolling(3).mean()
    df["body3"]       = body.rolling(3).mean()

    df["dist_ma20_atr"]   = (df["close"] - df["MA20"]) / (df["ATR14"] + 1e-9)
    df["price_diff5_atr"] = (df["close"] - df["close"].shift(5)) / (df["ATR14"] + 1e-9)
    df["ma_cross_prox"]   = (np.abs(df["MA20"] - df["MA50"])) / (df["ATR14"] + 1e-9)

    # Linear-regression slopes on short windows
    df["StochRSI_TrendSlope"] = df["StochRSI_%K"].rolling(5, min_periods=5).apply(rolling_linreg_slope, raw=False)
    df["ATR_slope"]           = df["ATR14"].rolling(5, min_periods=5).apply(rolling_linreg_slope, raw=False)

    # Local return context
    df["past_ret"]      = df["ret"].rolling(PAST, min_periods=PAST).sum()
    df["run_ret_hist"]  = df["past_ret"]

    # State-like features
    mu100 = df["close"].rolling(100).mean()
    sd100 = df["close"].rolling(100).std(ddof=0)
    df["z_close_100"] = (df["close"] - mu100) / (sd100 + 1e-9)

    roll_peak = df["close"].cummax()
    df["dd_from_peak_hist"] = (df["close"] / (roll_peak + 1e-9)) - 1.0

    # Trend age: bars since last sign change of EMA_gap
    sig = np.sign(df["EMA_gap"].fillna(0.0)).astype(int)
    age = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if sig.iat[i] == 0:
            age[i] = 0
        elif i > 0 and sig.iat[i] == sig.iat[i-1]:
            age[i] = age[i-1] + 1
        else:
            age[i] = 1
    df["trend_age_hist"] = pd.Series(age, index=df.index)

    # r2_60 on log(close)
    def _r2_win(x: pd.Series) -> float:
        y = np.log(np.maximum(x.to_numpy(), 1e-12))
        n = y.size
        if n < 2 or np.all(np.isnan(y)): return np.nan
        t = np.arange(n, dtype=float)
        ym, tm = np.nanmean(y), np.mean(t)
        cov = np.nansum((t - tm) * (y - ym))
        var_t = np.sum((t - tm) ** 2) + 1e-9
        b = cov / var_t
        yhat = ym + b * (t - tm)
        ss_res = np.nansum((y - yhat) ** 2)
        ss_tot = np.nansum((y - ym) ** 2) + 1e-9
        return 1.0 - ss_res / ss_tot
    df["r2_60"] = df["close"].rolling(60, min_periods=60).apply(_r2_win, raw=False)

    # beta60 placeholder (needs benchmark to be non-NaN)
    if "beta60" not in df.columns:
        df["beta60"] = np.nan

    # Minimal backfill for legacy need_cols
    need_set = set(need_cols)
    def want(col: str) -> bool:
        return (col in need_set) and (col not in df.columns)

    if want("rsi14") and "RSI14" in df.columns:
        df["rsi14"] = df["RSI14"]

    if want("dist_ma50_atr") and {"MA50","ATR14"}.issubset(df.columns):
        df["dist_ma50_atr"] = (df["close"] - df["MA50"]) / (df["ATR14"] + 1e-9)

    if want("don_pos20"):
        hh = df["high"].rolling(20).max()
        ll = df["low"].rolling(20).min()
        df["don_pos20"] = (df["close"] - ll) / (hh - ll + 1e-9)

    df.drop(columns=[c for c in ["_DIp","_DIn"] if c in df.columns], inplace=True)
    return df

def compute_features_if_needed(df: pd.DataFrame, need_cols: list[str]) -> pd.DataFrame:
    base = {"open","high","low","close"}
    if not base.issubset(df.columns):
        miss = sorted(base - set(df.columns))
        raise ValueError(f"Missing OHLC columns: {miss}")
    df = add_all_features(df, need_cols)
    core_need = {"ret", "RSI14", VOL_COL, MA126, STD126}
    miss_core = [c for c in core_need if c not in df.columns]
    if miss_core:
        raise ValueError(f"Missing core columns after feature build: {miss_core}")
    # index normalization
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()

def load_model_and_meta(txt_path: str, meta_path: str):
    booster = lgb.Booster(model_file=txt_path)
    thr = 0.5
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # try common keys
        for k in ["thr","threshold","best_threshold","opt_thr"]:
            if k in meta:
                thr = float(meta[k]); break
    except Exception:
        meta = {}
    return booster, thr, meta

def align_features_for(booster: lgb.Booster, df_feat: pd.DataFrame) -> pd.DataFrame:
    feat_names = booster.feature_name()
    X = pd.DataFrame(index=df_feat.index)
    for c in feat_names:
        X[c] = df_feat[c] if c in df_feat.columns else np.nan
    return X

def compute_metrics_from_equity(equity: pd.Series, daily_ret: pd.Series) -> dict:
    if equity.empty or daily_ret.dropna().empty:
        return dict(total_return=np.nan, CAGR=np.nan, Sharpe=np.nan, MaxDD=np.nan,
                    WinRate=np.nan, Trades=0, AvgRet=np.nan, LongAvgRet=np.nan, ShortAvgRet=np.nan)
    total_return = equity.iloc[-1] - 1.0
    n_days = daily_ret.dropna().shape[0]
    CAGR = float(equity.iloc[-1]**(ANNUALIZE/max(1,n_days)) - 1.0)
    vol = daily_ret.std(ddof=0)
    Sharpe = float(np.sqrt(ANNUALIZE) * daily_ret.mean() / vol) if vol and not np.isnan(vol) and vol>0 else np.nan
    # Max drawdown on equity
    peak = equity.cummax()
    dd = (equity/peak - 1.0)
    MaxDD = float(dd.min()) if not dd.empty else np.nan
    return dict(total_return=float(total_return), CAGR=CAGR, Sharpe=Sharpe, MaxDD=MaxDD)

def summarize_trades(trades: list[dict]) -> dict:
    if not trades:
        return dict(WinRate=np.nan, Trades=0, AvgRet=np.nan, LongAvgRet=np.nan, ShortAvgRet=np.nan)
    rets = np.array([t["ret"] for t in trades], dtype=float)
    longs  = np.array([t["ret"] for t in trades if t["side"]==+1], dtype=float)
    shorts = np.array([t["ret"] for t in trades if t["side"]==-1], dtype=float)
    winrate = float((rets > 0).mean()) if rets.size else np.nan
    avg_all = float(np.nanmean(rets)) if rets.size else np.nan
    avg_long = float(np.nanmean(longs)) if longs.size else np.nan
    avg_short= float(np.nanmean(shorts)) if shorts.size else np.nan
    return dict(WinRate=winrate, Trades=int(len(trades)), AvgRet=avg_all,
                LongAvgRet=avg_long, ShortAvgRet=avg_short)

# ===================== Load models =====================
model_long, thr_long, meta_long   = load_model_and_meta(LONG_TXT, LONG_META)
model_short, thr_short, meta_short= load_model_and_meta(SHORT_TXT, SHORT_META)

# Union of features needed by both boosters for minimal backfill
NEED_COLS = sorted(set(model_long.feature_name()) | set(model_short.feature_name()))

# ===================== Single-symbol backtest =====================
def backtest_symbol(df_raw: pd.DataFrame,
                    model_long: lgb.Booster, thr_long: float,
                    model_short: lgb.Booster, thr_short: float,
                    start_date=None, end_date=None,
                    cost_bps: float = 0.0) -> tuple[pd.Series, pd.Series, list[dict]]:
    """
    Returns:
      daily_ret  : Series of daily returns (open-to-open) under our position process
      equity     : 1 * cumprod(1 + daily_ret)
      trades     : list of trade dicts with entry/exit and PnL
    """
    # Normalize columns to lowercase OHLCV if needed
    rename_map = {c: c.lower() for c in df_raw.columns}
    df = df_raw.rename(columns=rename_map).copy()

    # Ensure datetime index
    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        # try an index-based parse
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(subset=[df.index.name])

    df = df.sort_index()
    if start_date: df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:   df = df[df.index <= pd.to_datetime(end_date)]
    if df.shape[0] < 200:  # need enough bars for features
        return pd.Series(dtype=float), pd.Series(dtype=float), []

    # Build features (includes EMA_gap etc.)
    feat = compute_features_if_needed(df, NEED_COLS)

    # Trend filter from EMA_gap sign
    trend = np.sign(feat["EMA_gap"].fillna(0.0)).astype(int)
    feat["trend"] = trend

    idx_up = feat.index[np.sign(feat["EMA_gap"].fillna(0)).astype(int)==+1]
    idx_dn = feat.index[np.sign(feat["EMA_gap"].fillna(0)).astype(int)==-1]

    p_long  = pd.Series(np.nan, index=feat.index)
    p_short = pd.Series(np.nan, index=feat.index)

    if len(idx_up):
        p = long_clf.predict_proba(feat.loc[idx_up, :])[:, 1]
        p_long.loc[idx_up] = p

    if len(idx_dn):
        p = short_clf.predict_proba(feat.loc[idx_dn, :])[:, 1]
        p_short.loc[idx_dn] = p

    # Trading signals for desired position at time t (executed at t+1 open)
    desired = pd.Series(0, index=feat.index, dtype=int)
    desired.loc[idx_up]  = np.where((p_long.loc[idx_up]  >= thr_long),  -1, 0)  # short when long-model flags reversal
    desired.loc[idx_dn]  = np.where((p_short.loc[idx_dn] >= thr_short), +1, 0)  # long  when short-model flags reversal

    # Convert desired -> held position with stop-and-reverse; only change when desired != 0
    pos = pd.Series(0, index=desired.index, dtype=int)
    for i in range(1, len(desired)):
        pos.iat[i] = pos.iat[i-1]
        if desired.iat[i-1] != 0 and desired.iat[i-1] != pos.iat[i-1]:
            pos.iat[i] = desired.iat[i-1]  # switch at today's open? No: we execute at t+1 open -> position becomes active today
        elif desired.iat[i-1] != 0 and desired.iat[i-1] == pos.iat[i-1]:
            # hold as is
            pass

    # Daily open-to-open returns
    open_px = df["open"].reindex(pos.index).astype(float)
    open_ret = open_px.pct_change()

    # Apply position; include simple round-trip costs (bps) when we change position
    daily_ret = pos.shift(0).fillna(0).astype(float) * open_ret
    if cost_bps and cost_bps > 0:
        # charge cost when |pos_t - pos_{t-1}| > 0 (entry/flip/exit)
        turns = (pos.diff().fillna(0) != 0).astype(int)
        daily_ret = daily_ret - turns * (cost_bps * 1e-4)

    equity = (1.0 + daily_ret.fillna(0)).cumprod()

    # Build trade list from position changes (executed at next open)
    trades = []
    cur_side, entry_idx, entry_px = 0, None, None
    for i in range(1, len(pos)):
        prev, now = pos.iat[i-1], pos.iat[i]
        ts_prev, ts_now = pos.index[i-1], pos.index[i]
        # entry at i when pos turns from 0 to ±1
        if prev == 0 and now != 0:
            cur_side = int(now)
            entry_idx = ts_now
            entry_px  = float(open_px.loc[ts_now])
        # exit/flip when pos turns from ±1 to 0 or opposite
        elif prev != 0 and now != prev:
            exit_idx = ts_now
            exit_px  = float(open_px.loc[ts_now])
            signed_ret = cur_side * (exit_px / entry_px - 1.0)
            trades.append(dict(entry=entry_idx, exit=exit_idx, side=cur_side,
                               entry_px=entry_px, exit_px=exit_px, ret=signed_ret))
            # if flip, immediately re-enter on same bar at open with new side
            if now != 0:
                cur_side = int(now)
                entry_idx = ts_now
                entry_px  = float(open_px.loc[ts_now])
            else:
                cur_side, entry_idx, entry_px = 0, None, None

    # Final open equity close out (optional): ignore to keep pure signal-to-signal trades

    return daily_ret.fillna(0.0), equity, trades

# ===================== Batch over folder & portfolio aggregation =====================
results = []
per_code_daily = {}   # code -> daily return series
per_code_trades = {}  # code -> list of trade dicts

# Only last 10 years
END = None
START = (pd.Timestamp.today().normalize() - pd.Timedelta(days=365*10+7)).date()

files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
for fp in files:
    code = Path(fp).stem
    try:
        df_raw = pd.read_parquet(fp)
    except Exception as e:
        print(f"[WARN] {code}: failed to read parquet ({e}); skip.")
        continue

    dr, eq, trades = backtest_symbol(
        df_raw, model_long, thr_long, model_short, thr_short,
        start_date=START, end_date=END, cost_bps=0.0  # set cost_bps if you want fees
    )
    if dr.empty:
        print(f"[WARN] {code}: not enough data for features; skip.")
        continue

    # Metrics
    m_core = compute_metrics_from_equity((1+dr).cumprod(), dr)
    m_trd  = summarize_trades(trades)
    row = dict(code=code, **m_core, **m_trd)
    results.append(row)

    per_code_daily[code]  = dr
    per_code_trades[code] = trades

# Per-code summary
summary_df = pd.DataFrame(results).sort_values("CAGR", ascending=False)
summary_df.reset_index(drop=True, inplace=True)

# ---- Portfolio aggregation (equal weight on active positions across codes) ----
if not per_code_daily:
    raise SystemExit("No symbols produced results.")

panel = pd.DataFrame(per_code_daily).sort_index().fillna(0.0)
port_ret = panel.mean(axis=1)              # equal-weight across codes each day
port_eq  = (1.0 + port_ret).cumprod()

# Portfolio metrics + trade stats aggregated
port_core = compute_metrics_from_equity(port_eq, port_ret)
all_trades = [t for tr in per_code_trades.values() for t in tr]
port_trd  = summarize_trades(all_trades)

# 组合行：code 用固定标识，字段与 per-code 对齐
portfolio_row = dict(code="PORTFOLIO", **port_core, **port_trd)

# 确保列顺序与 summary_df 一致（缺的列补 NaN，多的列丢弃）
for col in summary_df.columns:
    portfolio_row.setdefault(col, np.nan)
portfolio_row = {k: portfolio_row[k] for k in summary_df.columns}

# 追加到 summary_df
summary_df = pd.concat([summary_df, pd.DataFrame([portfolio_row])], ignore_index=True)

display(summary_df.head(10))
display(summary_df.tail(10))
OUT_DIR = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\result"
out1 = os.path.join(OUT_DIR, "backtest_summary_by_code.csv")
summary_df.to_csv(out1, index=False)
print(f"[OK] Saved summary (with portfolio) -> {out1}")



# ── Cell #3 (markdown) ──
# 不再用 EMA_gap 判趋势，而是每天同时跑 long/short 两个模型；谁先给信号就按谁开仓；开仓后只有出现“相反方向”的信号才允许平仓/反手；无信号则一直持有。

# ── Cell #4 (code) ──
import os, glob, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==== User paths (you already defined these; keep them consistent) ====
PARQUET_DIR = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\PARQUET美股\PARQUET"

LONG_TXT  = r"D:\models_generic\long_trend_rev_20250820_172531.txt"
LONG_META = r"D:\models_generic\long_trend_rev_20250820_172531_meta.json"
SHORT_TXT = r"D:\models_generic\short_trend_rev_20250821_112322.txt"
SHORT_META= r"D:\models_generic\short_trend_rev_20250821_112322_meta.json"
# ==== Global feature config (match your feature builder expectations) ====
ANNUALIZE = 252
VOL_COL   = "σ20"        # internal canonical vol feature name
MA_COL    = "ma20_126"
STD_COL   = "std20_126"
MA126, STD126 = MA_COL, STD_COL
PAST      = 10

# ===================== Utilities & feature helpers (self-contained) =====================
def pct_rank_last(s: pd.Series) -> float:
    """Percentile rank of the last value within the window [0,1] (exclude itself)."""
    if s.size < 2: return np.nan
    last = s.iloc[-1]
    rank = (s <= last).sum() - 1
    return rank / (s.size - 1) if s.size > 1 else np.nan

def rolling_linreg_slope(s: pd.Series) -> float:
    """Slope of y ~ a + b*t on the window (t=0..n-1)."""
    y = s.to_numpy()
    n = y.size
    if n < 2 or np.all(np.isnan(y)): return np.nan
    t = np.arange(n, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2: return np.nan
    t = t[mask]; y = y[mask]
    t = t - t.mean()
    y = y - y.mean()
    denom = np.sum(t*t)
    if denom == 0: return np.nan
    return float(np.sum(t*y) / denom)
import os, re, json, numpy as np, pandas as pd, lightgbm as lgb

class LGBBoosterWrapper:
    """Fixed threshold + fixed feature order; 2-col predict_proba."""
    def __init__(self, booster: lgb.Booster, threshold: float, features: list[str]):
        self.booster = booster
        self.threshold = float(threshold)
        self.features = list(features)  # training order

    def build_X(self, df_feat: pd.DataFrame) -> np.ndarray:
        # select in training order; fill missing with NaN
        idx = df_feat.index
        cols = []
        for c in self.features:
            if c in df_feat.columns:
                s = df_feat[c]
            else:
                s = pd.Series(np.nan, index=idx)
            cols.append(s.astype("float32"))
        X = np.column_stack(cols).astype("float32")
        # sanity: dimension match
        exp = self.booster.num_feature()
        if X.shape[1] != exp:
            raise RuntimeError(f"Feature dim mismatch: X has {X.shape[1]}, booster expects {exp}")
        return X

    def predict_proba(self, X_like) -> np.ndarray:
        if isinstance(X_like, pd.DataFrame):
            X = self.build_X(X_like)
        else:
            X = np.asarray(X_like, dtype="float32")
        n_it = getattr(self.booster, "best_iteration", None) or None
        p = self.booster.predict(X, raw_score=False, num_iteration=n_it)
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        return p

def _looks_like_column_i(names: list[str]) -> bool:
    if not names: return True
    return all(re.fullmatch(r"Column_\d+", str(n)) for n in names[:min(5, len(names))])

def load_booster_with_meta(txt_path: str, meta_path: str | None = None) -> LGBBoosterWrapper:
    booster = lgb.Booster(model_file=txt_path)

    # threshold
    thr = 0.5
    feats = None
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            thr = meta.get("threshold") or meta.get("best_threshold") or meta.get("thr") or 0.5
            # accept multiple keys for features
            for k in ("features","feature_cols","feature_order","feat_names"):
                if k in meta and isinstance(meta[k], list) and meta[k]:
                    feats = list(map(str, meta[k]))
                    break
        except Exception:
            pass

    if not feats:
        # fallback to model file only if it's NOT Column_i
        names = booster.feature_name()
        if not names or _looks_like_column_i(names):
            raise ValueError(
                "Cannot determine feature order: meta lacks features and model exposes generic Column_i. "
                "Retrain saving feature list to meta or supply it explicitly."
            )
        feats = list(map(str, names))

    # dedup, keep order
    seen, clean_feats = set(), []
    for c in feats:
        if c not in seen:
            seen.add(c); clean_feats.append(c)

    # optional: assert count
    exp = booster.num_feature()
    if len(clean_feats) != exp:
        # Not fatal, but warn / normalize if needed
        # raise ValueError(...)
        pass

    return LGBBoosterWrapper(booster, float(thr), clean_feats)

# Load models
LONG_MODEL  = load_booster_with_meta(LONG_TXT,  LONG_META)
SHORT_MODEL = load_booster_with_meta(SHORT_TXT, SHORT_META)

long_clf,  long_thr  = LONG_MODEL,  LONG_MODEL.threshold
short_clf, short_thr = SHORT_MODEL, SHORT_MODEL.threshold

feature_cols_long  = LONG_MODEL.features
feature_cols_short = SHORT_MODEL.features
need_cols = sorted(set(feature_cols_long) | set(feature_cols_short))

# ---- Your feature builders (as given/assumed) ----
def add_all_features(df: pd.DataFrame, need_cols: list[str]) -> pd.DataFrame:
    import ta
    eps = 1e-9
    df = df.copy()

    # Basic cleaning
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Returns
    df["ret"]    = df["close"].pct_change()
    df["logret"] = np.log1p(df["ret"])

    # Averages / EMA / slopes
    df["MA20"]   = df["close"].rolling(20).mean()
    df["MA50"]   = df["close"].rolling(50).mean()
    df["MA100"]  = df["close"].rolling(100).mean()
    df["EMA20"]  = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"]  = df["close"].ewm(span=50, adjust=False).mean()

    # Gap & distances
    df["EMA_gap"]       = df["EMA20"] - df["EMA50"]
    df["close_to_MA20"] = df["close"] - df["MA20"]
    df["close_to_MA50"] = df["close"] - df["MA50"]

    # Slopes (both names)
    df["MA20_slope"]   = df["MA20"].diff(5)
    df["slope_ma_20"]  = df["MA20"].diff()
    df["slope_ma_100"] = df["MA100"].diff()
    df["slope50"]      = df["MA50"].diff()

    # TA indicators
    df["RSI14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14)
    df["ATR14"] = atr.average_true_range()

    stoch = ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3)
    df["StochRSI_%K"] = stoch.stochrsi_k()
    df["StochRSI_%D"] = stoch.stochrsi_d()
    df["Stoch_K_minus_D"] = df["StochRSI_%K"] - df["StochRSI_%D"]

    # ADX / DI
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["ADX14"] = adx.adx()
    df["_DIp"]  = adx.adx_pos()
    df["_DIn"]  = adx.adx_neg()
    df["dip_min_din"] = df["_DIp"] - df["_DIn"]
    df["adx_down5"]   = df["ADX14"] - df["ADX14"].shift(5)

    # RSI derivatives
    df["RSI14_slope5"]   = df["RSI14"].diff(5)
    df["RSI14_pct_rank"] = df["RSI14"].rolling(window=20, min_periods=5).apply(pct_rank_last, raw=False)
    rsi_mean20 = df["RSI14"].rolling(20).mean()
    rsi_std20  = df["RSI14"].rolling(20).std(ddof=0)
    df["RSI_sigma"] = (df["RSI14"] - rsi_mean20) / (rsi_std20 + 1e-9)

    # Volatility core (annualized σ20)
    WIN = 20
    df[VOL_COL] = df["ret"].rolling(WIN).std(ddof=0) * np.sqrt(ANNUALIZE)
    df["σ20"] = df[VOL_COL]  # ensure literal column

    # Long windows on VOL_COL
    for lb in (126, 252):
        df[f"ma20_{lb}"]  = df[VOL_COL].rolling(lb).mean()
        df[f"std20_{lb}"] = df[VOL_COL].rolling(lb).std(ddof=0)

    # Short vol stats & transforms
    df["vol_ma20"]  = df[VOL_COL].rolling(WIN).mean()
    df["vol_std20"] = df[VOL_COL].rolling(WIN).std(ddof=0)
    df["vol_ma_past"]     = df[VOL_COL].rolling(PAST, min_periods=PAST).mean()
    df["vol_ma_past_chg"] = df["vol_ma_past"].pct_change()
    df["σ20_z"]     = (df[VOL_COL] - df[MA126]) / (df[STD126] + 1e-9)
    df["vol_of_vol10"] = df[VOL_COL].rolling(10).std(ddof=0)
    df["vol_ratio60"]   = df[VOL_COL] / (df[VOL_COL].rolling(60).median().replace(0, np.nan))

    # Bollinger & Keltner
    ma20_close = df["close"].rolling(WIN).mean()
    sd20_close = df["close"].rolling(WIN).std(ddof=0)
    bb_up = ma20_close + 2.0 * sd20_close
    bb_lo = ma20_close - 2.0 * sd20_close
    df["bb_width"] = (bb_up - bb_lo) / (ma20_close + 1e-9)

    k_up = df["EMA20"] + 2.0 * df["ATR14"]
    k_lo = df["EMA20"] - 2.0 * df["ATR14"]
    df["keltner_w"]   = (2.0 * df["ATR14"]) / (df["EMA20"] + 1e-9)
    df["squeeze_on"]  = ((bb_up < k_up) & (bb_lo > k_lo)).astype("int8")

    # ATR-normalized shapes & proximities
    upper_wick = (df["high"] - np.maximum(df["open"], df["close"])) / (df["ATR14"] + 1e-9)
    lower_wick = (np.minimum(df["open"], df["close"]) - df["low"]) / (df["ATR14"] + 1e-9)
    body       = (np.abs(df["close"] - df["open"])) / (df["ATR14"] + 1e-9)
    df["upper_wick3"] = upper_wick.rolling(3).mean()
    df["lower_wick3"] = lower_wick.rolling(3).mean()
    df["body3"]       = body.rolling(3).mean()

    df["dist_ma20_atr"]   = (df["close"] - df["MA20"]) / (df["ATR14"] + 1e-9)
    df["price_diff5_atr"] = (df["close"] - df["close"].shift(5)) / (df["ATR14"] + 1e-9)
    df["ma_cross_prox"]   = (np.abs(df["MA20"] - df["MA50"])) / (df["ATR14"] + 1e-9)

    # Linear-regression slopes on short windows
    df["StochRSI_TrendSlope"] = df["StochRSI_%K"].rolling(5, min_periods=5).apply(rolling_linreg_slope, raw=False)
    df["ATR_slope"]           = df["ATR14"].rolling(5, min_periods=5).apply(rolling_linreg_slope, raw=False)

    # Local return context
    df["past_ret"]      = df["ret"].rolling(PAST, min_periods=PAST).sum()
    df["run_ret_hist"]  = df["past_ret"]

    # State-like features
    mu100 = df["close"].rolling(100).mean()
    sd100 = df["close"].rolling(100).std(ddof=0)
    df["z_close_100"] = (df["close"] - mu100) / (sd100 + 1e-9)

    roll_peak = df["close"].cummax()
    df["dd_from_peak_hist"] = (df["close"] / (roll_peak + 1e-9)) - 1.0

    # Trend age: bars since last sign change of EMA_gap
    sig = np.sign(df["EMA_gap"].fillna(0.0)).astype(int)
    age = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if sig.iat[i] == 0:
            age[i] = 0
        elif i > 0 and sig.iat[i] == sig.iat[i-1]:
            age[i] = age[i-1] + 1
        else:
            age[i] = 1
    df["trend_age_hist"] = pd.Series(age, index=df.index)

    # r2_60 on log(close)
    def _r2_win(x: pd.Series) -> float:
        y = np.log(np.maximum(x.to_numpy(), 1e-12))
        n = y.size
        if n < 2 or np.all(np.isnan(y)): return np.nan
        t = np.arange(n, dtype=float)
        ym, tm = np.nanmean(y), np.mean(t)
        cov = np.nansum((t - tm) * (y - ym))
        var_t = np.sum((t - tm) ** 2) + 1e-9
        b = cov / var_t
        yhat = ym + b * (t - tm)
        ss_res = np.nansum((y - yhat) ** 2)
        ss_tot = np.nansum((y - ym) ** 2) + 1e-9
        return 1.0 - ss_res / ss_tot
    df["r2_60"] = df["close"].rolling(60, min_periods=60).apply(_r2_win, raw=False)

    # beta60 placeholder (needs benchmark to be non-NaN)
    if "beta60" not in df.columns:
        df["beta60"] = np.nan

    # Minimal backfill for legacy need_cols
    need_set = set(need_cols)
    def want(col: str) -> bool:
        return (col in need_set) and (col not in df.columns)

    if want("rsi14") and "RSI14" in df.columns:
        df["rsi14"] = df["RSI14"]

    if want("dist_ma50_atr") and {"MA50","ATR14"}.issubset(df.columns):
        df["dist_ma50_atr"] = (df["close"] - df["MA50"]) / (df["ATR14"] + 1e-9)

    if want("don_pos20"):
        hh = df["high"].rolling(20).max()
        ll = df["low"].rolling(20).min()
        df["don_pos20"] = (df["close"] - ll) / (hh - ll + 1e-9)

    df.drop(columns=[c for c in ["_DIp","_DIn"] if c in df.columns], inplace=True)
    return df

def compute_features_if_needed(df: pd.DataFrame, need_cols: list[str]) -> pd.DataFrame:
    base = {"open","high","low","close"}
    if not base.issubset(df.columns):
        miss = sorted(base - set(df.columns))
        raise ValueError(f"Missing OHLC columns: {miss}")
    df = add_all_features(df, need_cols)
    core_need = {"ret", "RSI14", VOL_COL, MA126, STD126}
    miss_core = [c for c in core_need if c not in df.columns]
    if miss_core:
        raise ValueError(f"Missing core columns after feature build: {miss_core}")
    # index normalization
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()

def load_model_and_meta(txt_path: str, meta_path: str):
    booster = lgb.Booster(model_file=txt_path)
    thr = 0.5
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # try common keys
        for k in ["thr","threshold","best_threshold","opt_thr"]:
            if k in meta:
                thr = float(meta[k]); break
    except Exception:
        meta = {}
    return booster, thr, meta

def align_features_for(booster: lgb.Booster, df_feat: pd.DataFrame) -> pd.DataFrame:
    feat_names = booster.feature_name()
    X = pd.DataFrame(index=df_feat.index)
    for c in feat_names:
        X[c] = df_feat[c] if c in df_feat.columns else np.nan
    return X

def compute_metrics_from_equity(equity: pd.Series, daily_ret: pd.Series) -> dict:
    if equity.empty or daily_ret.dropna().empty:
        return dict(total_return=np.nan, CAGR=np.nan, Sharpe=np.nan, MaxDD=np.nan,
                    WinRate=np.nan, Trades=0, AvgRet=np.nan, LongAvgRet=np.nan, ShortAvgRet=np.nan)
    total_return = equity.iloc[-1] - 1.0
    n_days = daily_ret.dropna().shape[0]
    CAGR = float(equity.iloc[-1]**(ANNUALIZE/max(1,n_days)) - 1.0)
    vol = daily_ret.std(ddof=0)
    Sharpe = float(np.sqrt(ANNUALIZE) * daily_ret.mean() / vol) if vol and not np.isnan(vol) and vol>0 else np.nan
    # Max drawdown on equity
    peak = equity.cummax()
    dd = (equity/peak - 1.0)
    MaxDD = float(dd.min()) if not dd.empty else np.nan
    return dict(total_return=float(total_return), CAGR=CAGR, Sharpe=Sharpe, MaxDD=MaxDD)

def summarize_trades(trades: list[dict]) -> dict:
    if not trades:
        return dict(WinRate=np.nan, Trades=0, AvgRet=np.nan, LongAvgRet=np.nan, ShortAvgRet=np.nan)
    rets = np.array([t["ret"] for t in trades], dtype=float)
    longs  = np.array([t["ret"] for t in trades if t["side"]==+1], dtype=float)
    shorts = np.array([t["ret"] for t in trades if t["side"]==-1], dtype=float)
    winrate = float((rets > 0).mean()) if rets.size else np.nan
    avg_all = float(np.nanmean(rets)) if rets.size else np.nan
    avg_long = float(np.nanmean(longs)) if longs.size else np.nan
    avg_short= float(np.nanmean(shorts)) if shorts.size else np.nan
    return dict(WinRate=winrate, Trades=int(len(trades)), AvgRet=avg_all,
                LongAvgRet=avg_long, ShortAvgRet=avg_short)

# ===================== Load models =====================
model_long, thr_long, meta_long   = load_model_and_meta(LONG_TXT, LONG_META)
model_short, thr_short, meta_short= load_model_and_meta(SHORT_TXT, SHORT_META)

# Union of features needed by both boosters for minimal backfill
NEED_COLS = sorted(set(model_long.feature_name()) | set(model_short.feature_name()))

# ===================== Single-symbol backtest =====================
def backtest_symbol(df_raw: pd.DataFrame,
                    model_long: lgb.Booster, thr_long: float,
                    model_short: lgb.Booster, thr_short: float,
                    start_date=None, end_date=None,
                    cost_bps: float = 0.0) -> tuple[pd.Series, pd.Series, list[dict]]:
    """
    Returns:
      daily_ret  : Series of daily returns (open-to-open)
      equity     : 1 * cumprod(1 + daily_ret)
      trades     : list of trade dicts with entry/exit and PnL
    """
    # --- 0) 规范化输入 ---
    rename_map = {c: c.lower() for c in df_raw.columns}
    df = df_raw.rename(columns=rename_map).copy()

    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(subset=[df.index.name])

    df = df.sort_index()
    if start_date: df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:   df = df[df.index <= pd.to_datetime(end_date)]
    if df.shape[0] < 200:
        return pd.Series(dtype=float), pd.Series(dtype=float), []

    # --- 1) 计算特征（不做趋势门控） ---
    feat = compute_features_if_needed(df, NEED_COLS)

    # --- 2) 两个模型同时预测（使用你的 wrapper long_clf/short_clf） ---
    # long 模型的“做空概率”
    p_long_all  = pd.Series(long_clf.predict_proba(feat)[:, 1],  index=feat.index)
    # short 模型的“做多概率”
    p_short_all = pd.Series(short_clf.predict_proba(feat)[:, 1], index=feat.index)

    # 处理 NaN：把 NaN margin 视为“未触发”
    m_long  = (p_long_all  - thr_long).astype(float)
    m_short = (p_short_all - thr_short).astype(float)
    m_long[p_long_all.isna()]   = -np.inf
    m_short[p_short_all.isna()] = -np.inf

    # --- 3) 生成当日 desired(t) ---
    desired = pd.Series(0, index=feat.index, dtype=int)

    # 仅 long 触发 -> 做空
    mask_short_only = (m_short < 0) & (m_long >= 0)
    desired.loc[mask_short_only] = -1

    # 仅 short 触发 -> 做多
    mask_long_only  = (m_long  < 0) & (m_short >= 0)
    desired.loc[mask_long_only]  = +1

    # 同时触发 -> 谁 margin 大选谁；相等时偏向反向（这里默认偏向“做空”）
    both = (m_long >= 0) & (m_short >= 0)
    if both.any():
        pick_short = (m_short.loc[both] > m_long.loc[both])  # True -> 多；False -> 空
        desired.loc[both] = np.where(pick_short, +1, -1)

    # （可选）加入触发带宽抑制抖动，例如 margin 至少 >= 0.01 才算触发：
    # eps_margin = 0.00
    # desired[(m_long.abs() < eps_margin) & (m_short.abs() < eps_margin)] = 0

    # --- 4) 仓位更新（T+1 开盘生效；仅相反信号才改变仓位；无信号/同向信号忽略） ---
    pos = pd.Series(0, index=desired.index, dtype=int)
    for i in range(1, len(desired)):
        pos.iat[i] = pos.iat[i-1]
        # 只有当昨日日志望的方向非0 且 与昨日持仓不同 -> 今日开盘切换
        if desired.iat[i-1] != 0 and desired.iat[i-1] != pos.iat[i-1]:
            pos.iat[i] = desired.iat[i-1]

    # --- 5) 收益（开盘到开盘） + 交易成本 ---
    open_px  = df["open"].reindex(pos.index).astype(float)
    open_ret = open_px.pct_change()

    daily_ret = pos.astype(float) * open_ret
    if cost_bps and cost_bps > 0:
        turns = (pos.diff().fillna(0) != 0).astype(int)
        daily_ret = daily_ret - turns * (cost_bps * 1e-4)

    equity = (1.0 + daily_ret.fillna(0)).cumprod()

    # --- 6) 交易列表（按仓位变化记录） ---
    trades = []
    cur_side, entry_idx, entry_px = 0, None, None
    for i in range(1, len(pos)):
        prev, now = pos.iat[i-1], pos.iat[i]
        ts_now = pos.index[i]
        if prev == 0 and now != 0:
            # 建仓（在 i 时刻的开盘）
            cur_side = int(now)
            entry_idx = ts_now
            entry_px  = float(open_px.loc[ts_now])
        elif prev != 0 and now != prev:
            # 平/反手：先记录上一笔，再（若 now!=0）立即以同一开盘价作为新建仓起点
            exit_idx = ts_now
            exit_px  = float(open_px.loc[ts_now])
            signed_ret = cur_side * (exit_px / entry_px - 1.0)
            trades.append(dict(entry=entry_idx, exit=exit_idx, side=cur_side,
                               entry_px=entry_px, exit_px=exit_px, ret=signed_ret))
            if now != 0:
                cur_side = int(now)
                entry_idx = ts_now
                entry_px  = float(open_px.loc[ts_now])
            else:
                cur_side, entry_idx, entry_px = 0, None, None

    return daily_ret.fillna(0.0), equity, trades


# ===================== Batch over folder & portfolio aggregation =====================
results = []
per_code_daily = {}   # code -> daily return series
per_code_trades = {}  # code -> list of trade dicts

# Only last 10 years
END = None
START = (pd.Timestamp.today().normalize() - pd.Timedelta(days=365*10+7)).date()

files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
for fp in files:
    code = Path(fp).stem
    try:
        df_raw = pd.read_parquet(fp)
    except Exception as e:
        print(f"[WARN] {code}: failed to read parquet ({e}); skip.")
        continue

    dr, eq, trades = backtest_symbol(
        df_raw, model_long, thr_long, model_short, thr_short,
        start_date=START, end_date=END, cost_bps=0.0  # set cost_bps if you want fees
    )
    if dr.empty:
        print(f"[WARN] {code}: not enough data for features; skip.")
        continue

    # Metrics
    m_core = compute_metrics_from_equity((1+dr).cumprod(), dr)
    m_trd  = summarize_trades(trades)
    row = dict(code=code, **m_core, **m_trd)
    results.append(row)

    per_code_daily[code]  = dr
    per_code_trades[code] = trades

# Per-code summary
summary_df = pd.DataFrame(results).sort_values("CAGR", ascending=False)
summary_df.reset_index(drop=True, inplace=True)

# ---- Portfolio aggregation (equal weight on active positions across codes) ----
if not per_code_daily:
    raise SystemExit("No symbols produced results.")

panel = pd.DataFrame(per_code_daily).sort_index().fillna(0.0)
port_ret = panel.mean(axis=1)              # equal-weight across codes each day
port_eq  = (1.0 + port_ret).cumprod()

# Portfolio metrics + trade stats aggregated
port_core = compute_metrics_from_equity(port_eq, port_ret)
all_trades = [t for tr in per_code_trades.values() for t in tr]
port_trd  = summarize_trades(all_trades)

# 组合行：code 用固定标识，字段与 per-code 对齐
portfolio_row = dict(code="PORTFOLIO", **port_core, **port_trd)

# 确保列顺序与 summary_df 一致（缺的列补 NaN，多的列丢弃）
for col in summary_df.columns:
    portfolio_row.setdefault(col, np.nan)
portfolio_row = {k: portfolio_row[k] for k in summary_df.columns}

# 追加到 summary_df
summary_df = pd.concat([summary_df, pd.DataFrame([portfolio_row])], ignore_index=True)

# 显示与保存（只保存一个文件，包含组合行）
display(summary_df.head(10))
display(summary_df.tail(10))
OUT_DIR = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\result"
out1 = os.path.join(OUT_DIR, "backtest_summary_by_code.csv")
summary_df.to_csv(out1, index=False)
print(f"[OK] Saved summary (with portfolio) -> {out1}")


# ── Cell #5 (code) ──

