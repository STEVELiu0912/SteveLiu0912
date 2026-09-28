# ── Cell #1 (code) ──
import os, glob, numpy as np, pandas as pd
from pathlib import Path
from joblib import load
import warnings
import ta
warnings.filterwarnings('ignore')


# ── Cell #2 (markdown) ──
# 实际回测（2024 - 2025）


# ── Cell #3 (markdown) ──
# 目标：用“波动率布林带 + 趋势过滤 + RSI 反转”捕捉波动扩张后的单边机会，并在波动回落/过热或触发风控时退出。
# 
# 开仓条件（满足全部条件）：
# 1. 20日波动率上穿其126日均线。
# 2. 过去7日累计收益率在指定区间[0.5%, 10%]。
# 
# 平仓条件:
# 1. 波动率跌回均线以下或突破上轨（趋势结束）。
# 2. 达到止盈。
# 3. 达到止损。
# 
# 趋势判断:
# 1. 过去7日累计收益率 > 0 ⇒ 判断为上涨趋势（做多）。
# 2. 过去7日累计收益率 < 0 ⇒ 判断为下跌趋势（做空）。
# 3. 若不在以上区间，判为无明显趋势，不开仓。
# 4. 若触发做多信号时RSI > 65，则反转方向为做空，若触发做空信号时RSI < 30，则反转方向为做多。

# ── Cell #4 (code) ──
# ───────────── Global config ─────────────
DATA_DIR   = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\PARQUET美股\PARQUET"
OUT_DIR    = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\result"
START_DATE = pd.Timestamp("2000-01-01")
END_DATE   = pd.Timestamp("2025-08-05")

# Parameters
WIN        = 20
LOOKBACK   = 126
SIGMA_LVL  = 2
STOP_LOSS_PCT  = 2
TAKE_PROFIT_PCT = 2

MIN_THRES  = 0.015
MAX_THRES  = 0.15
PAST       = 10
# === 动态阈值 ===
CAL_WIN = 126
Q_LOW, Q_HIGH = 0.2, 0.95

VOL_COL  = f"σ{WIN}"
MA_COL   = f"ma{WIN}_{LOOKBACK}"
STD_COL  = f"std{WIN}_{LOOKBACK}"
REQUIRED_COLS = {"close", "ret", "RSI14", VOL_COL, MA_COL, STD_COL}

SLOTS = 10
MANUAL_FEATURES = [
        "EMA_gap","close_to_MA20","close_to_MA50",
        "RSI14","RSI14_slope5","RSI14_pct_rank",
        "StochRSI_%K","StochRSI_%D","Stoch_K_minus_D","StochRSI_TrendSlope",
        "σ20","σ20_z","bb_width","vol_ma20","vol_std20","vol_ma_past_chg",
        "ma20_126","std20_126","ma20_252","std20_252",
        "ATR14","ATR_slope","upper_wick3","lower_wick3","body3",
        "past_ret","ret",
        "beta60","r2_60","ADX14","dip_min_din","slope50",
        "dist_ma20_atr","keltner_w","squeeze_on",
        "trend_age_hist","run_ret_hist","dd_from_peak_hist",
        "z_close_100","slope_ma_20","slope_ma_100",
        "adx_down5","price_diff5_atr",
        "vol_of_vol10","vol_ratio60","ma_cross_prox",
    ]
LONG_TXT  = r"D:\models_generic\long_trend_rev_20250820_172531.txt"
LONG_META = r"D:\models_generic\long_trend_rev_20250820_172531_meta.json"
SHORT_TXT = r"D:\models_generic\short_trend_rev_20250821_112322.txt"
SHORT_META= r"D:\models_generic\short_trend_rev_20250821_112322_meta.json"

# ── Cell #5 (code) ──
import pandas as pd
import glob, os

folder = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\PARQUET美股\PARQUET"

for fp in glob.glob(os.path.join(folder, "*.parquet")):
    try:
        df = pd.read_parquet(fp)
        df.columns = [c.lower() for c in df.columns]   # 转小写
        df.to_parquet(fp, index=True)                  # 覆盖保存
        print(f"[OK] fixed {os.path.basename(fp)}")
    except Exception as e:
        print(f"[FAIL] {fp}: {e}")


# ── Cell #6 (code) ──
import os, json, numpy as np, lightgbm as lgb

def _sanitize_path(p: str):
    if p is None: return None
    p = str(p).strip().strip('"').strip("'")     # 去掉引号/空格/换行
    p = os.path.normpath(p)
    if not os.path.isabs(p):
        p = os.path.abspath(p)                   # 统一绝对路径
    # Windows 超长路径兜底
    if os.name == "nt" and len(p) > 240 and not p.startswith("\\\\?\\"):
        p = "\\\\?\\" + p
    return p

def load_booster_with_meta(txt_path: str, meta_path: str | None = None):
    txt_path = _sanitize_path(txt_path)
    print("[CHK] model path =", repr(txt_path))
    ok = os.path.exists(txt_path)
    size = os.path.getsize(txt_path) if ok else -1
    print(f"[CHK] exists={ok} size={size}")

    booster = None
    # 1) 先尝试常规方式
    if ok and size > 0:
        try:
            booster = lgb.Booster(model_file=txt_path)
            print("[OK] loaded by model_file")
        except Exception as e:
            print("[WARN] model_file load failed:", e)

    # 2) 失败就用 model_str 兜底（避免路径/编码怪异）
    if booster is None and ok and size > 0:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            model_str = f.read()
        booster = lgb.Booster(model_str=model_str)
        print("[OK] loaded by model_str fallback")

    if booster is None:
        raise FileNotFoundError(f"Cannot open model: {txt_path}")

    # 读 meta：阈值 + 特征
    thr, feats = 0.5, None
    meta_path = _sanitize_path(meta_path) if meta_path else None
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            thr   = meta.get("threshold") or meta.get("best_threshold") or meta.get("thr") or 0.5
            feats = meta.get("features") or meta.get("feature_cols")
        except Exception as e:
            print("[WARN] read meta failed:", e)

    if not feats:
        bn = booster.feature_name()
        if bn and len(bn) > 0:
            feats = list(bn)
        else:
            raise ValueError("meta 无 features，模型也没记录 feature_name()，无法确定顺序")

    # 去重保序
    seen, clean_feats = set(), []
    for c in feats:
        c = str(c)
        if c not in seen:
            seen.add(c)
            clean_feats.append(c)

    class _Wrap:
        def __init__(self, booster, thr, feats):
            self.booster = booster
            self.threshold = float(thr)
            self.features = list(feats)
        def predict_proba(self, X):
            if hasattr(X, "to_numpy"): X = X.to_numpy()
            n_it = getattr(self.booster, "best_iteration", None) or None
            p = self.booster.predict(X, raw_score=False, num_iteration=n_it)
            if p.ndim == 1: p = np.column_stack([1.0-p, p])
            return p

    print("[OK] features =", len(clean_feats))
    return _Wrap(booster, thr, clean_feats)

# ───────────── NEW: load LightGBM boosters from .txt + meta.json ─────────────
import os, json
import numpy as np
import lightgbm as lgb

class LGBBoosterWrapper:
    """Minimal wrapper: fixed threshold + fixed feature order + predict_proba."""
    def __init__(self, booster: lgb.Booster, threshold: float, features: list[str]):
        self.booster = booster
        self.threshold = float(threshold)
        self.features = list(features)  # 固定顺序

    def predict_proba(self, X):
        # 这里假定 X 的列顺序已在外部按 self.features 对齐
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        n_it = getattr(self.booster, "best_iteration", None)
        if not n_it or n_it <= 0:
            n_it = None  # 使用全部树
        p = self.booster.predict(X, raw_score=False, num_iteration=n_it)
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        return p

def load_booster_with_meta(txt_path: str, meta_path: str | None = None) -> LGBBoosterWrapper:
    if txt_path is None or not os.path.exists(txt_path):
        raise FileNotFoundError(f"Model file not found: {txt_path}")
    booster = lgb.Booster(model_file=txt_path)

    # 阈值 & 特征列表
    thr = 0.5
    feats = None
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            thr = meta.get("threshold") or meta.get("best_threshold") or meta.get("thr") or 0.5
            feats = meta.get("features") or meta.get("feature_cols")
        except Exception as e:
            print("[WARN] read meta failed:", e)

    # 若 meta 无 features，就尝试模型内的 feature_name
    if not feats:
        bn = booster.feature_name()
        if bn and len(bn) > 0:
            feats = list(bn)
        else:
            raise ValueError("无法确定特征列顺序：meta无features，模型文件无feature_name()")

    # 去重 & 保序
    seen, clean_feats = set(), []
    for c in feats:
        c = str(c)
        if c not in seen:
            seen.add(c)
            clean_feats.append(c)

    return LGBBoosterWrapper(booster, float(thr), clean_feats)

LONG_MODEL  = load_booster_with_meta(LONG_TXT,  LONG_META)
SHORT_MODEL = load_booster_with_meta(SHORT_TXT, SHORT_META)

long_clf,  long_thr  = LONG_MODEL,  LONG_MODEL.threshold
short_clf, short_thr = SHORT_MODEL, SHORT_MODEL.threshold

# 之后预测时务必这样对齐列顺序（不要用 need_cols 送进模型）：
feature_cols_long  = LONG_MODEL.features
feature_cols_short = SHORT_MODEL.features

# 只用于检查覆盖率，不用于预测
need_cols = set(feature_cols_long) | set(feature_cols_short)
print("[INFO] need_cols size =", len(need_cols))
print("[INFO] long features head =", feature_cols_long[:6], "...", len(feature_cols_long))
print("[INFO] short features head=", feature_cols_short[:6], "...", len(feature_cols_short))

# 打印模型里记录的特征名时要防 None
bn_long = LONG_MODEL.booster.feature_name()
bn_short = SHORT_MODEL.booster.feature_name()
print("[INFO] booster long feat sample:", (bn_long[:10] if bn_long else None))
print("[INFO] booster short feat sample:", (bn_short[:10] if bn_short else None))

print("[INFO] LONG_META exists?", os.path.exists(LONG_META), LONG_META)
print("[INFO] SHORT_META exists?", os.path.exists(SHORT_META), SHORT_META)



# ── Cell #7 (code) ──
# ==== constants ====
ANNUALIZE = 252
MA126, STD126 = MA_COL, STD_COL   # 让 add_all_features 里用到的别名与全局一致

# ==== helpers used by feature builder ====
def pct_rank_last(s: pd.Series) -> float:
    """Percentile rank of the last value within the window [0,1]."""
    if s.size < 2:
        return np.nan
    last = s.iloc[-1]
    rank = (s <= last).sum() - 1  # exclude the last itself
    return rank / (s.size - 1)

def rolling_linreg_slope(s: pd.Series) -> float:
    """Slope of y ~ a + b*t on the window (t=0..n-1)."""
    y = s.to_numpy()
    n = y.size
    if n < 2 or np.all(np.isnan(y)):
        return np.nan
    t = np.arange(n, dtype=float)
    # remove NaNs
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    t = t[mask]; y = y[mask]
    # slope b = Cov(t,y)/Var(t)
    vt = np.var(t)
    if vt == 0:
        return 0.0
    return np.cov(t, y, ddof=0)[0, 1] / vt

def add_all_features(df: pd.DataFrame, need_cols: list[str]) -> pd.DataFrame:
    """
    Compute the full current feature set; then minimally backfill only the truly
    missing legacy cols listed in need_cols. No lookahead/leakage.
    """
    import numpy as np
    import ta

    eps = 1e-9
    df = df.copy()

    # ---- Basic cleaning ----
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Returns ----
    df["ret"]    = df["close"].pct_change()
    df["logret"] = np.log1p(df["ret"])

    # ---- Averages / EMA / slopes ----
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
    df["MA20_slope"]  = df["MA20"].diff(5)
    df["slope_ma_20"] = df["MA20"].diff()
    df["slope_ma_100"]= df["MA100"].diff()
    df["slope50"]     = df["MA50"].diff()

    # ---- TA indicators ----
    df["RSI14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14)
    df["ATR14"] = atr.average_true_range()

    stoch = ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3)
    df["StochRSI_%K"] = stoch.stochrsi_k()
    df["StochRSI_%D"] = stoch.stochrsi_d()
    df["Stoch_K_minus_D"] = df["StochRSI_%K"] - df["StochRSI_%D"]

    # ADX / DI (needed for ADX14, dip_min_din, adx_down5)
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["ADX14"] = adx.adx()
    df["_DIp"]  = adx.adx_pos()
    df["_DIn"]  = adx.adx_neg()
    df["dip_min_din"] = df["_DIp"] - df["_DIn"]
    df["adx_down5"]   = df["ADX14"] - df["ADX14"].shift(5)

    # ---- RSI derivatives ----
    df["RSI14_slope5"]  = df["RSI14"].diff(5)
    df["RSI14_pct_rank"] = df["RSI14"].rolling(window=20, min_periods=5).apply(pct_rank_last, raw=False)
    rsi_mean20 = df["RSI14"].rolling(20).mean()
    rsi_std20  = df["RSI14"].rolling(20).std(ddof=0)
    df["RSI_sigma"] = (df["RSI14"] - rsi_mean20) / (rsi_std20 + eps)

    # ---- Volatility core (annualized σ20 by your VOL_COL) ----
    WIN = 20
    df[VOL_COL] = df["ret"].rolling(WIN).std(ddof=0) * np.sqrt(ANNUALIZE)

    # Also expose literal 'σ20' if VOL_COL != 'σ20'
    if VOL_COL != "σ20":
        df["σ20"] = df[VOL_COL]
    else:
        # Ensure the literal column exists (covers both cases)
        df["σ20"] = df[VOL_COL]

    # Long windows on VOL_COL
    for lb in (126, 252):
        df[f"ma20_{lb}"]  = df[VOL_COL].rolling(lb).mean()
        df[f"std20_{lb}"] = df[VOL_COL].rolling(lb).std(ddof=0)

    # Short vol stats & transforms
    df["vol_ma20"]  = df[VOL_COL].rolling(WIN).mean()
    df["vol_std20"] = df[VOL_COL].rolling(WIN).std(ddof=0)
    df["vol_ma_past"]     = df[VOL_COL].rolling(PAST, min_periods=PAST).mean()
    df["vol_ma_past_chg"] = df["vol_ma_past"].pct_change()
    df["σ20_z"]     = (df[VOL_COL] - df[MA126]) / (df[STD126] + eps)
    df["vol_of_vol10"] = df[VOL_COL].rolling(10).std(ddof=0)
    df["vol_ratio60"]   = df[VOL_COL] / (df[VOL_COL].rolling(60).median().replace(0, np.nan))

    # ---- Price-based Bollinger & Keltner (for bb_width, squeeze_on, keltner_w) ----
    ma20_close = df["close"].rolling(WIN).mean()
    sd20_close = df["close"].rolling(WIN).std(ddof=0)
    bb_up = ma20_close + 2.0 * sd20_close
    bb_lo = ma20_close - 2.0 * sd20_close
    df["bb_width"] = (bb_up - bb_lo) / (ma20_close + eps)

    k_up = df["EMA20"] + 2.0 * df["ATR14"]
    k_lo = df["EMA20"] - 2.0 * df["ATR14"]
    df["keltner_w"] = (2.0 * df["ATR14"]) / (df["EMA20"] + eps)
    df["squeeze_on"] = ((bb_up < k_up) & (bb_lo > k_lo)).astype("int8")

    # ---- ATR-normalized shapes & proximities ----
    upper_wick = (df["high"] - np.maximum(df["open"], df["close"])) / (df["ATR14"] + eps)
    lower_wick = (np.minimum(df["open"], df["close"]) - df["low"]) / (df["ATR14"] + eps)
    body       = (np.abs(df["close"] - df["open"])) / (df["ATR14"] + eps)
    df["upper_wick3"] = upper_wick.rolling(3).mean()
    df["lower_wick3"] = lower_wick.rolling(3).mean()
    df["body3"]       = body.rolling(3).mean()

    df["dist_ma20_atr"]   = (df["close"] - df["MA20"]) / (df["ATR14"] + eps)
    df["price_diff5_atr"] = (df["close"] - df["close"].shift(5)) / (df["ATR14"] + eps)
    df["ma_cross_prox"]   = (np.abs(df["MA20"] - df["MA50"])) / (df["ATR14"] + eps)

    # ---- Linear-regression slopes on short windows ----
    df["StochRSI_TrendSlope"] = df["StochRSI_%K"].rolling(5, min_periods=5).apply(rolling_linreg_slope, raw=False)
    df["ATR_slope"]           = df["ATR14"].rolling(5, min_periods=5).apply(rolling_linreg_slope, raw=False)

    # ---- Local return context ----
    df["past_ret"]      = df["ret"].rolling(PAST, min_periods=PAST).sum()
    df["run_ret_hist"]  = df["past_ret"]  # alias for model

    # ---- State-like features ----
    # z-score of close on 100 bars
    mu100 = df["close"].rolling(100).mean()
    sd100 = df["close"].rolling(100).std(ddof=0)
    df["z_close_100"] = (df["close"] - mu100) / (sd100 + eps)

    # drawdown from running peak
    roll_peak = df["close"].cummax()
    df["dd_from_peak_hist"] = (df["close"] / (roll_peak + eps)) - 1.0

    # trend age: bars since last sign change of EMA_gap (non-negative count)
    sig = np.sign(df["EMA_gap"].fillna(0.0)).astype(int)
    age = np.zeros(len(df), dtype=int)
    last = 0
    for i in range(len(df)):
        if sig.iat[i] == 0:
            age[i] = 0
        elif i > 0 and sig.iat[i] == sig.iat[i-1]:
            age[i] = age[i-1] + 1
        else:
            age[i] = 1
        last = sig.iat[i]
    df["trend_age_hist"] = pd.Series(age, index=df.index)

    # ---- Regressions (costly ones) ----
    # r2_60 on log(close)
    def _r2_win(x: pd.Series) -> float:
        y = np.log(np.maximum(x.to_numpy(), 1e-12))
        n = y.size
        if n < 2 or np.all(np.isnan(y)): return np.nan
        t = np.arange(n, dtype=float)
        ym, tm = np.nanmean(y), np.mean(t)
        cov = np.nansum((t - tm) * (y - ym))
        var_t = np.sum((t - tm) ** 2) + eps
        b = cov / var_t
        yhat = ym + b * (t - tm)
        ss_res = np.nansum((y - yhat) ** 2)
        ss_tot = np.nansum((y - ym) ** 2) + eps
        return 1.0 - ss_res / ss_tot
    df["r2_60"] = df["close"].rolling(60, min_periods=60).apply(_r2_win, raw=False)

    # beta60 requires a benchmark; keep NaN placeholder (as before)
    df["beta60"] = df.get("beta60", np.nan)

    # ---- Minimal backfill for legacy need_cols (only if still missing) ----
    need_set = set(need_cols)
    def want(col: str) -> bool:
        return (col in need_set) and (col not in df.columns)

    if want("rsi14") and "RSI14" in df.columns:
        df["rsi14"] = df["RSI14"]

    if want("dist_ma50_atr") and {"MA50","ATR14"}.issubset(df.columns):
        df["dist_ma50_atr"] = (df["close"] - df["MA50"]) / (df["ATR14"] + eps)

    if want("don_pos20"):
        hh = df["high"].rolling(20).max()
        ll = df["low"].rolling(20).min()
        df["don_pos20"] = (df["close"] - ll) / (hh - ll + eps)

    # Cleanup temp cols
    df.drop(columns=[c for c in ["_DIp","_DIn"] if c in df.columns], inplace=True)

    return df

def compute_features_if_needed(df: pd.DataFrame, need_cols: list[str]) -> pd.DataFrame:
    """
    统一入口：
      1) 校验 OHLC；
      2) 按当前口径计算全套特征；
      3) 仅对模型真实需要且缺失的“老特征”做最小补齐；
      4) 仍缺的列以 NaN 兜底（LightGBM 能处理缺失）；
      5) 返回按时间排序的 DataFrame。
    """
    import numpy as np

    base = {"open","high","low","close"}
    if not base.issubset(df.columns):
        miss = sorted(base - set(df.columns))
        raise ValueError(f"缺少基础OHLC列，无法计算特征: {miss}")

    # 1&2：计算当下口径的全量特征 + 3：按需补齐老特征
    df = add_all_features(df, need_cols)

    # 关键核心列也去空一次（以防万一）
    core_need = {"ret", "RSI14", VOL_COL, MA126, STD126}
    miss_core = [c for c in core_need if c not in df.columns]
    if miss_core:
        # 如果到这步还缺，说明上游数据不完整；抛错让外层处理
        raise ValueError(f"缺少核心列: {miss_core}")

    # 5：时间索引清理与排序
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    return df


# ── Cell #8 (code) ──
def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], utc=False, errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date")
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame 无 'date' 列且索引不是 DatetimeIndex")
    else:
        # 已是 DatetimeIndex，也去掉可能的 tz
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
    out = out.sort_index()
    return out


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If ret missing but close exists, derive it
    if "ret" not in df.columns and "close" in df.columns:
        df["ret"] = df["close"].pct_change()
    return df

def first_fully_valid_start(df: pd.DataFrame, cols: set[str]) -> pd.Timestamp | None:
    ok = df.dropna(subset=list(cols))
    return ok.index.min() if not ok.empty else None

def backtest_one(file_path: str) -> dict | None:
    """
    Backtest a single symbol. Returns metrics dict or None on failure.
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"[WARN] 读取失败 {file_path}: {e}")
        return None

    df = ensure_datetime_index(df)
    df = ensure_derived_columns(df)

    # 先按日期裁剪
    df = df.loc[START_DATE:END_DATE].copy()
    if df.empty:
        print(f"[WARN] 区间内无数据 {file_path}")
        return None

    # 先补齐/重算缺失特征
    try:
        df = compute_features_if_needed(df, need_cols)
    except ValueError as e:
        print(f"[WARN] {Path(file_path).name}: {e}")
        return None
    # 先补齐/重算缺失特征之后，立刻检查
    missing_feats = [c for c in set(feature_cols_long) | set(feature_cols_short) if c not in df.columns]
    if missing_feats:
        print(f"[WARN] {Path(file_path).name} 缺少特征列: {missing_feats[:6]} ...")
        return None
    # 现在再严格检查必需列（基础 + 核心波动带）
    required_now = {"close", "ret", "RSI14", VOL_COL, MA_COL, STD_COL}
    missing = required_now - set(df.columns)
    if missing:
        print(f"[WARN] {Path(file_path).name} 缺少必需列: {sorted(missing)}")
        return None

    if len(df) < 20:
        print(f"[WARN] 有效区间过短 {file_path} (len={len(df)})")
        return None

    # 与进场判定口径一致：使用过去 PAST 天的日收益“和”
    df["ret_past_sum"] = df["ret"].rolling(PAST, min_periods=PAST).sum()
    ret_past_abs = df["ret_past_sum"].abs()

    # 分位数用滚动窗口，并在当前bar用到“前一时刻已知”的分位数（shift(1)）
    df["MIN_dyn"] = ret_past_abs.rolling(CAL_WIN, min_periods=int(CAL_WIN*0.5)).quantile(Q_LOW).shift(1)
    df["MAX_dyn"] = ret_past_abs.rolling(CAL_WIN, min_periods=int(CAL_WIN*0.5)).quantile(Q_HIGH).shift(1)
    # 带宽
    df["upper"] = df[MA_COL] + SIGMA_LVL * df[STD_COL]
    df["lower"] = df[MA_COL] - SIGMA_LVL * df[STD_COL]

    # 核心列最终去空
    df = df.dropna(subset=[VOL_COL, MA_COL, STD_COL, "ret", "RSI14"])
    if df.empty:
        print(f"[WARN] 关键列为空 {file_path}")
        return None

    # ====== 下面交易循环与绩效统计保持你的原逻辑，只把模型取特征处改成 feats_long/feats_short ======
    positions = pd.Series(index=df.index, dtype="float64")
    in_pos, dirn, entry_price = False, 0, None
    entry_rsi = None
    entry_reversed = False
    entry_rev_type = None
    trade_records: list[tuple[float, int, float, bool, str | None]] = []

    for i in range(1, len(df)):
        vol_prev, vol_curr = df[VOL_COL].iat[i - 1], df[VOL_COL].iat[i]
        ma_prev,  ma_curr  = df[MA_COL].iat[i - 1], df[MA_COL].iat[i]
        close_curr         = df["close"].iat[i]
        upper_curr         = df["upper"].iat[i]

        if not in_pos:
            crossed_up = (vol_prev <= ma_prev) and (vol_curr > ma_curr)
            if crossed_up:
                # 计算过去 PAST 天收益（与你原逻辑一致：求和）
                start = max(0, i - PAST + 1)
                past_ret = df["ret"].iloc[start : i + 1]
                net_ret  = float(past_ret.sum())
                '''
                # 取本bar的动态阈值（缺数据时回退到全局常量）
                min_i = df["MIN_dyn"].iat[i]
                max_i = df["MAX_dyn"].iat[i]
                if not np.isfinite(min_i) or not np.isfinite(max_i) or min_i <= 0 or max_i <= 0 or min_i >= max_i:
                    min_i, max_i = MIN_THRES, MAX_THRES  # fallback

                # 用动态阈值做方向判定
                if  min_i <= net_ret <= max_i:
                    dirn_base = 1
                elif -max_i <= net_ret <= -min_i:
                    dirn_base = -1
                else:
                    dirn_base = 0
                '''
                if MIN_THRES <= net_ret <= MAX_THRES:
                    dirn_base = 1
                elif -MAX_THRES <= net_ret <= -MIN_THRES:
                    dirn_base = -1
                else:
                    dirn_base = 0
                rsi_now = df["RSI14"].iat[i]
                dirn, is_rev, rev_type = dirn_base, False, None

                if dirn_base == 1 and rsi_now > 55:
                    X = df.loc[[df.index[i]], feature_cols_long].to_numpy()
                    proba = long_clf.predict_proba(X)[0, 1]
                    if proba >= long_thr:
                        dirn, is_rev, rev_type = -1, True, "LS"

                elif dirn_base == -1 and rsi_now < 45:
                    X = df.loc[[df.index[i]], feature_cols_short].to_numpy()
                    proba = short_clf.predict_proba(X)[0, 1]
                    if proba >= short_thr:
                        dirn, is_rev, rev_type = 1, True, "SL"

                if dirn != 0:
                    in_pos = True
                    entry_price    = close_curr
                    entry_rsi      = rsi_now
                    entry_reversed = is_rev
                    entry_rev_type = rev_type
                    positions.iat[i] = dirn
        else:
            exit_vol = (vol_curr < ma_curr) or (vol_curr > upper_curr)
            stop_loss = (dirn == 1 and close_curr < entry_price * (1 - STOP_LOSS_PCT)) or \
                        (dirn == -1 and close_curr > entry_price * (1 + STOP_LOSS_PCT))
            take_profit = (dirn == 1 and close_curr > entry_price * (1 + TAKE_PROFIT_PCT)) or \
                          (dirn == -1 and close_curr < entry_price * (1 - TAKE_PROFIT_PCT))

            if exit_vol or stop_loss or take_profit:
                trade_pl = (close_curr - entry_price) / entry_price * dirn
                trade_records.append((float(trade_pl), int(dirn), float(entry_rsi),
                                      bool(entry_reversed), entry_rev_type))
                in_pos, dirn, entry_price, entry_rsi = False, 0, None, None
                entry_reversed, entry_rev_type = False, None
                positions.iat[i] = 0
            else:
                positions.iat[i] = dirn

    positions = positions.ffill().fillna(0)
    df["strategy_ret"] = df["ret"] * positions.shift(1).fillna(0)
    df["equity"] = (1 + df["strategy_ret"]).cumprod()
    active_series = (positions != 0).astype(int)
    if df["equity"].empty:
        return None

    # ── 单标的绩效 ──
    total_ret = float(df["equity"].iat[-1] - 1)
    annual_ret = (df["equity"].iloc[-1]) ** (252 / len(df)) - 1

    ret_mean = float(df["strategy_ret"].mean())
    ret_std  = float(df["strategy_ret"].std(ddof=0))
    sharpe   = np.sqrt(252) * ret_mean / ret_std if ret_std > 0 else np.nan
    mdd      = max_drawdown(df["equity"])

    # ── 单标的交易统计（含 RSI 分组均值） ──
    trade_rets = np.array([t[0] for t in trade_records], dtype="float64")
    trade_dirs = np.array([t[1] for t in trade_records], dtype="int8")
    trade_rsi  = np.array([t[2] for t in trade_records], dtype="float64")

    win_mask   = trade_rets > 0
    long_mask  = trade_dirs > 0
    short_mask = trade_dirs < 0

    def _mean_safe(arr):
        return float(np.mean(arr)) if arr.size else np.nan

    win_rate  = float(win_mask.mean()) if trade_rets.size else np.nan
    avg_trade = _mean_safe(trade_rets)
    long_avg  = _mean_safe(trade_rets[long_mask])
    short_avg = _mean_safe(trade_rets[short_mask])

    avg_rsi_long_win   = _mean_safe(trade_rsi[long_mask  & win_mask])
    avg_rsi_long_loss  = _mean_safe(trade_rsi[long_mask  & ~win_mask & (trade_rets < 0)])
    avg_rsi_short_win  = _mean_safe(trade_rsi[short_mask & win_mask])
    avg_rsi_short_loss = _mean_safe(trade_rsi[short_mask & ~win_mask & (trade_rets < 0)])

    trade_isrv = np.array([t[3] for t in trade_records], dtype=bool)
    trade_rtyp = np.array([t[4] for t in trade_records], dtype=object)

    rev_mask = trade_isrv
    ls_mask  = rev_mask & (trade_rtyp == "LS")
    sl_mask  = rev_mask & (trade_rtyp == "SL")

    def _win_rate(mask):
        n = int(mask.sum())
        return float((win_mask & mask).mean()) if n > 0 else np.nan

    rev_count        = int(rev_mask.sum())
    rev_avg_pl       = _mean_safe(trade_rets[rev_mask])
    rev_win_numerator= int((win_mask & rev_mask).sum())
    rev_win_rate     = (rev_win_numerator / rev_count if rev_count else np.nan)
    rev_LS_count     = int(ls_mask.sum())
    rev_LS_avg_pl    = _mean_safe(trade_rets[ls_mask])
    rev_SL_count     = int(sl_mask.sum())
    rev_SL_avg_pl    = _mean_safe(trade_rets[sl_mask])

    return {
        "code"               : Path(file_path).stem,
        "总收益率"             : total_ret,
        "年化收益率"           : float(annual_ret),
        "夏普比率"             : float(sharpe),
        "最大回撤"             : mdd,
        "胜率"                 : win_rate,
        "交易次数"             : int(trade_rets.size),
        "平均收益"             : avg_trade,
        "多头平均收益"         : long_avg,
        "空头平均收益"         : short_avg,
        "多头盈利RSI均值"     : avg_rsi_long_win,
        "多头亏损RSI均值"     : avg_rsi_long_loss,
        "空头盈利RSI均值"     : avg_rsi_short_win,
        "空头亏损RSI均值"     : avg_rsi_short_loss,

        # 反转相关指标
        "RSI反转次数"          : rev_count,
        "RSI反转成功次数"       : rev_win_numerator,
        "RSI反转胜率"          : rev_win_rate,
        "RSI反转平均收益"      : rev_avg_pl,
        "多转空次数"           : rev_LS_count,
        "多转空平均收益"       : rev_LS_avg_pl,
        "空转多次数"           : rev_SL_count,
        "空转多平均收益"       : rev_SL_avg_pl,

        "_daily_ret"          : df["strategy_ret"].copy(),
        "_trade_details"      : trade_records,   # (pl, dir, rsi, is_rev, rev_type)
        "_active"             : active_series,
    }


# ── Cell #9 (code) ──
# ───────────────────────── 辅助函数 ─────────────────────────
def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（负数）。"""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())
def check_required_columns(df: pd.DataFrame, file_path: str) -> None:
    """严格校验所需列，不做兜底计算。缺失直接报错。"""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{Path(file_path).name} 缺少必需列: {sorted(missing)}")

def simulate_with_slots_d(ret_map: dict[str, pd.Series],
                               active_map: dict[str, pd.Series],
                               slots: int):
    """
    返回：
      slot_rets   : DataFrame(index=日期, columns=slot_1..slot_k)，每个槽位的日收益(空闲=0)
      slot_equity : 同上，每个槽位的净值曲线（初始=1）
      port_ret    : 组合日收益 = 10 槽位日收益的等权平均
      alloc_codes : DataFrame(index=日期, columns=slot_1..slot_k)，每槽位当日所跟的代码(空闲=None)
    规则：
      - 入场判定：active 从0→1的日子视为“入场”。若当日空闲槽位不足，超出的入场被丢弃（直到其下一次出现入场）。
      - 收益口径：使用你回测返回的 strategy_ret（即入场当日收益=0，次日开始计）。
    """
    # 统一索引
    all_idx = None
    for s in list(ret_map.values()) + list(active_map.values()):
        all_idx = s.index if all_idx is None else all_idx.union(s.index)
    all_idx = all_idx.sort_values()

    codes = sorted(active_map.keys())
    ret_mat    = pd.concat([ret_map[c].reindex(all_idx).fillna(0.0) for c in codes], axis=1)
    active_mat = pd.concat([active_map[c].reindex(all_idx).fillna(0).astype(int) for c in codes], axis=1)
    ret_mat.columns = active_mat.columns = codes

    # 初始化
    slot_codes = [None] * slots        # 每槽位当前跟随的代码
    prev_active = pd.Series(0, index=codes, dtype=int)

    slot_rets_rows = []
    alloc_rows = []

    for t in all_idx:
        today_active = active_mat.loc[t]

        # 1) 释放已平仓（从1→0）
        for si, code in enumerate(slot_codes):
            if code is not None and today_active[code] == 0:
                slot_codes[si] = None

        # 2) 新入场（0→1），按代码字典序，填入空槽
        entrants = [c for c in codes if (prev_active[c] == 0 and today_active[c] == 1)]
        free_idx = [i for i, c in enumerate(slot_codes) if c is None]
        for si, c in zip(free_idx, entrants):
            slot_codes[si] = c
            # 若空槽不够，剩余 entrants 被丢弃（即资金受限不执行）

        # 3) 计算当日每槽位收益（跟谁就取谁的 strategy_ret；空闲为 0）
        slot_ret = []
        alloc = []
        for code in slot_codes:
            if code is None:
                slot_ret.append(0.0)
                alloc.append(None)
            else:
                slot_ret.append(float(ret_mat.loc[t, code]))
                alloc.append(code)

        slot_rets_rows.append(slot_ret)
        alloc_rows.append(alloc)
        prev_active = today_active
    slot_cols = [f"slot_{i+1}" for i in range(slots)]
    slot_rets   = pd.DataFrame(slot_rets_rows, index=all_idx, columns=slot_cols)
    alloc_codes = pd.DataFrame(alloc_rows,   index=all_idx, columns=slot_cols)

    # 槽位净值与组合
    slot_equity = (1 + slot_rets).cumprod()
    port_ret    = slot_rets.mean(axis=1)  # 等权求平均 = （总资金=10份）日收益

    return slot_rets, slot_equity, port_ret, alloc_codes

def simulate_with_slots(ret_map, active_map, slots:int) -> pd.Series:
    all_idx = None
    for s in list(ret_map.values()) + list(active_map.values()):
        all_idx = s.index if all_idx is None else all_idx.union(s.index)
    all_idx = all_idx.sort_values()
    codes = sorted(active_map.keys())
    ret_mat    = pd.concat([ret_map[c].reindex(all_idx).fillna(0.0) for c in codes], axis=1)
    active_mat = pd.concat([active_map[c].reindex(all_idx).fillna(0).astype(int) for c in codes], axis=1)
    ret_mat.columns = active_mat.columns = codes

    allocated = set()
    prev_active = pd.Series(0, index=codes, dtype=int)
    out = []

    for t in all_idx:
        today_active = active_mat.loc[t]

        # 释放
        for c in list(allocated):
            if today_active[c] == 0:
                allocated.remove(c)

        # 新增（FIFO：同日超限按代码序）
        entrants = [c for c in codes if (prev_active[c] == 0 and today_active[c] == 1)]
        free = slots - len(allocated)
        if free > 0:
            for c in entrants[:free]:
                allocated.add(c)

        # 组合当日收益 = （已分配标的的 strategy_ret 之和）/ 槽位数
        day_sum = ret_mat.loc[t, list(allocated)].sum() if allocated else 0.0
        out.append(day_sum / slots)

        prev_active = today_active

    return pd.Series(out, index=all_idx, name=f"port_ret_slots_{slots}")

# ── Cell #10 (code) ──
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows_for_table = []
    all_trades: list[tuple[float, int, float]] = []  # (pl, dirn, rsi)
    daily_rets = {}
    active_map = {}

    for fp in glob.glob(os.path.join(DATA_DIR, "*.parquet")):
        res = backtest_one(fp)
        if res is None:
            continue

        # 收集组合用的数据
        all_trades.extend(res["_trade_details"])
        daily_rets[res["code"]] = res["_daily_ret"]
        active_map[res["code"]] = res["_active"]

        # 丢弃大对象，保留表格列
        row = {k: v for k, v in res.items()
               if not k.startswith("_")}
        rows_for_table.append(row)

    if not rows_for_table:
        raise RuntimeError("没有成功回测的文件！")

    # ——— 单标的结果表 ———
    df_res = (pd.DataFrame(rows_for_table)
                .set_index("code")
                .sort_index())

    # ——— Portfolio: equal-weight daily returns on union of dates ———
    if not daily_rets:
        raise RuntimeError("缺少组合日收益数据！")

    # Union index across all series
    all_idx = None
    for s in daily_rets.values():
        all_idx = s.index if all_idx is None else all_idx.union(s.index)
    all_idx = all_idx.sort_values()

    # Reindex each series to union, fill missing with 0 (no position/no data → 0)
    ret_mat = pd.concat([s.reindex(all_idx).fillna(0.0) for s in daily_rets.values()], axis=1)
    ret_mat.columns = list(daily_rets.keys())

    port_ret    = ret_mat.mean(axis=1, skipna=True)
    port_equity = (1 + port_ret).cumprod()

    total_ret     = float(port_equity.iat[-1] - 1)
    final_equity  = float(port_equity.iat[-1])
    annual_ret    = final_equity ** (252 / len(port_equity)) - 1
    ret_mean      = float(port_ret.mean())
    ret_std       = float(port_ret.std(ddof=0))
    sharpe        = np.sqrt(252) * ret_mean / ret_std if ret_std > 0 else np.nan
    mdd           = max_drawdown(port_equity)

    # Align actives on the same union index for concurrency stats
    active_mat = pd.concat([a.reindex(all_idx).fillna(0).astype(int) for a in active_map.values()], axis=1)
    active_mat.columns = list(active_map.keys())
    concurrent = active_mat.sum(axis=1).astype(int)
    max_concurrent = int(concurrent.max())


    # p95（含零日）：覆盖 95% 全部交易日（包括没有持仓的日子）
    def p95_int(s: pd.Series) -> int:
        vals = np.sort(s.to_numpy())
        k = int(np.ceil(0.95 * len(vals))) - 1
        return int(vals[max(k, 0)])

    # p95（仅持仓日）：只在 concurrent>0 的日子上计算，更贴近“有单时的资金需求”
    pos_days = concurrent[concurrent > 0]
    p95_all_days = p95_int(concurrent)
    p95_pos_days = p95_int(pos_days) if not pos_days.empty else 0

    ret_slots = simulate_with_slots(daily_rets, active_map, p95_all_days)
    equity_slots = (1 + ret_slots.fillna(0)).cumprod()
    total_slots  = float(equity_slots.iat[-1] - 1)
    final_slots_equity = float(equity_slots.iat[-1])
    annual_slots = final_slots_equity ** (252 / len(port_equity)) - 1

    # ——— 组合：胜率与交易分组 RSI ———
    trade_rets  = np.array([t[0] for t in all_trades], dtype="float64")
    trade_dirs  = np.array([t[1] for t in all_trades], dtype="int8")
    trade_rsi   = np.array([t[2] for t in all_trades], dtype="float64")
    trade_isrv  = np.array([t[3] for t in all_trades], dtype=bool)
    trade_rtyp  = np.array([t[4] for t in all_trades], dtype=object)

    win_mask    = trade_rets > 0
    long_mask   = trade_dirs > 0
    short_mask  = trade_dirs < 0
    rev_mask    = trade_isrv
    ls_mask     = rev_mask & (trade_rtyp == "LS")
    sl_mask     = rev_mask & (trade_rtyp == "SL")


    def _mean_safe(a): return float(np.mean(a)) if a.size else np.nan

    rev_win_numerator = int((win_mask & rev_mask).sum())
    port_rev_count     = int(rev_mask.sum())
    port_rev_win_rate = (rev_win_numerator / port_rev_count if port_rev_count else np.nan)
    port_rev_avg_pl    = _mean_safe(trade_rets[rev_mask])

    port_rev_LS_count  = int(ls_mask.sum())
    port_rev_LS_avg_pl = _mean_safe(trade_rets[ls_mask])

    port_rev_SL_count  = int(sl_mask.sum())
    port_rev_SL_avg_pl = _mean_safe(trade_rets[sl_mask])
    port_win_rate   = float(win_mask.mean()) if trade_rets.size else np.nan
    port_trades     = int(trade_rets.size)
    port_avg_trade  = _mean_safe(trade_rets)
    port_long_avg   = _mean_safe(trade_rets[long_mask])
    port_short_avg  = _mean_safe(trade_rets[short_mask])

    port_rsi_long_win   = _mean_safe(trade_rsi[long_mask  & win_mask])
    port_rsi_long_loss  = _mean_safe(trade_rsi[long_mask  & ~win_mask & (trade_rets < 0)])
    port_rsi_short_win  = _mean_safe(trade_rsi[short_mask & win_mask])
    port_rsi_short_loss = _mean_safe(trade_rsi[short_mask & ~win_mask & (trade_rets < 0)])

    port_row = pd.Series({
        "总收益率"            : total_ret,
        "年化收益率"          : float(annual_ret),
        "夏普比率"            : float(sharpe),
        "最大回撤"            : mdd,
        "胜率"                : port_win_rate,
        "交易次数"            : port_trades,
        "平均收益"            : port_avg_trade,
        "多头平均收益"        : port_long_avg,
        "空头平均收益"        : port_short_avg,
        "多头盈利RSI均值"    : port_rsi_long_win,
        "多头亏损RSI均值"    : port_rsi_long_loss,
        "空头盈利RSI均值"    : port_rsi_short_win,
        "空头亏损RSI均值"    : port_rsi_short_loss,
        "并发总收益率"        : total_slots,
        "并发年化收益率"      : float(annual_slots),

        # 组合层反转统计
        "RSI反转次数"            : port_rev_count,
        "RSI反转成功次数":        rev_win_numerator,
        "RSI反转胜率"            : port_rev_win_rate,
        "RSI反转平均收益"        : port_rev_avg_pl,
        "多转空次数"          : port_rev_LS_count,
        "多转空平均收益"      : port_rev_LS_avg_pl,
        "空转多次数"          : port_rev_SL_count,
        "空转多平均收益"      : port_rev_SL_avg_pl,
    }, name="组合")

    # 与单标的结果拼接
    df_out = pd.concat([df_res, port_row.to_frame().T], axis=0)

    # 保存
    out_csv = os.path.join(OUT_DIR, "all_backtests.csv")
    df_out.to_csv(out_csv, float_format="%.6f")
    print(f"\n已保存：{out_csv}")

    # 汇总打印
    print("\n===== Portfolio Aggregate Performance =====")
    for k, v in port_row.items():
        if isinstance(v, int):
            print(f"{k:<18}: {v:7d}")
        elif isinstance(v, float):
                print(f"{k:<18}: {v:7.4f}")
        else:  # 字符串，比如 "8/15"
            print(f"{k:<18}: {v}")


    # 频数分布表（便于直观看范围）
    concurrency_dist = concurrent.value_counts().sort_index()
    concurrency_dist.to_csv(os.path.join(OUT_DIR, "concurrency_distribution.csv"))

    print("\n===== Concurrency (Active Signals) =====")
    print(f"Max concurrent        : {max_concurrent:d}")
    print(f"p95 (all trading days): {p95_all_days:d}")
    print(f"p95 (active-only days): {p95_pos_days:d}")
    print(f"Distribution saved to : {os.path.join(OUT_DIR, 'concurrency_distribution.csv')}")
    slot_rets, slot_equity, port_ret10, alloc_codes = simulate_with_slots_d(daily_rets, active_map, p95_all_days)
    # 画 10 条资金曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 7))
    for c in slot_equity.columns:
        plt.plot(slot_equity.index, slot_equity[c], lw=1)
    plt.title("Equity Curves")
    plt.xlabel("Date"); plt.ylabel("Equity (Start=1)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()


# ── Cell #11 (markdown) ──
# 网格搜索最佳参数（2021 - 2024）

# ── Cell #12 (code) ──
import os, glob, numpy as np, pandas as pd
from pathlib import Path
from itertools import product
from contextlib import contextmanager
from joblib import Parallel, delayed
from tqdm import tqdm

# ── Cell #13 (code) ──
# ============= 固定默认参数（便于阅读与复用） =============
BASE_PARAMS = dict(
    WIN=20, LOOKBACK=126, SIGMA_LVL=2,
    STOP_LOSS_PCT=0.08, TAKE_PROFIT_PCT=0.03,
    MIN_THRES=0.01, MAX_THRES=0.10, PAST=7,
    MIN_RSI=30, MAX_RSI=70
)
DATA_DIR   = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\大宗商品\gooddata"
START_DATE = "2023-01-01"
END_DATE   = "2025-08-05"

# ============= 搜索空间（可按需微调） =============
WIN_LIST        = [20]
LOOKBACK_LIST   = [126]
SIGMA_LIST      = [2.0]
SL_PCT_LIST     = [2]
TP_PCT_LIST     = [2]
MIN_THRES_LIST  = [0.015]
MAX_THRES_LIST  = [0.15]
PAST_LIST       = [10]
MIN_RSI_LIST     = [30]
MAX_RSI_LIST     = [66]

# ── Cell #14 (code) ──
def eval_portfolio(daily_ret_dict: dict[str, pd.Series],
                   trade_details: list[tuple[float,int,float]]) -> dict:
    """
    daily_ret_dict: {code -> Series(strategy_ret)}（对齐日期）
    trade_details : [(pl, dir, rsi), ...]
    返回组合层面的 Sharpe/回撤/胜率/笔数/收益等（基于等权组合）。
    """
    if not daily_ret_dict:
        return dict(score=np.nan)

    ret_mat = pd.concat(daily_ret_dict.values(), axis=1).fillna(0.0)
    port_ret = ret_mat.mean(axis=1)                      # 当日等权
    port_equity = (1 + port_ret).cumprod()

    total_ret = float(port_equity.iat[-1] - 1)
    n_bdays   = pd.bdate_range(START_DATE, END_DATE).size
    final_equity = float(port_equity.iat[-1])  # 期末净值
    annual_ret = final_equity ** (252 / n_bdays) - 1     # ✅ 正确复利年化

    ret_mean = float(port_ret.mean())
    ret_std  = float(port_ret.std(ddof=0))
    sharpe   = np.sqrt(252) * ret_mean / ret_std if ret_std > 0 else np.nan
    mdd      = max_drawdown(port_equity)

    tr = np.array([t[0] for t in trade_details], dtype=float)
    win_rate = float((tr > 0).mean()) if tr.size else np.nan
    trades   = int(tr.size)

    return dict(
        score=sharpe,                 # 评分：组合 Sharpe
        sharpe=sharpe,
        annual_ret=float(annual_ret),
        total_ret=total_ret,
        max_dd=mdd,
        win_rate=win_rate,
        trades=trades
    )

# ============= 工具：参数注入（不改 backtest_one 的情况下最清晰的做法） =============
@contextmanager
def apply_params(param_dict: dict):
    keys = ["WIN","LOOKBACK","SIGMA_LVL","STOP_LOSS_PCT","TAKE_PROFIT_PCT",
            "MIN_THRES","MAX_THRES","PAST","MIN_RSI","MAX_RSI"]
    backup = {k: globals().get(k, None) for k in keys}
    try:
        for k in keys:
            if k in param_dict:
                globals()[k] = param_dict[k]
        yield
    finally:
        for k, v in backup.items():
            if v is None and k in globals():
                del globals()[k]
            else:
                globals()[k] = v

# ============= 单次参数组合的完整回测与评分 =============
def run_once(params: dict) -> dict | None:
    # 合法性：最基本关系约束
    if not (params["MIN_THRES"] < params["MAX_THRES"]):
        return None
    if params["PAST"] <= 0:
        return None
    if params["MIN_RSI"] >= params["MAX_RSI"]:
        return None
    daily_ret_dict = {}
    trade_details  = []
    with apply_params(params):
        for fp in glob.glob(os.path.join(DATA_DIR, "*.parquet")):
            res = backtest_one(fp)              # 使用你现有的函数
            if not res:
                continue
            # 组合需要的两个内部字段（参考我之前版本）
            if "_daily_ret" in res:
                daily_ret_dict[res["code"]] = res["_daily_ret"]
            if "_trade_details" in res:
                trade_details.extend(res["_trade_details"])
    if not daily_ret_dict:
        return None

    met = eval_portfolio(daily_ret_dict, trade_details)
    if np.isnan(met["score"]):
        return None

    return dict(params=params, **met)

# ============= 网格构造 =============
def build_param_grid() -> list[dict]:
    grid = []
    for w, lb, sg, sl, tp, mn, mx, past, minrsi, maxrsi in product(
        WIN_LIST, LOOKBACK_LIST, SIGMA_LIST,
        SL_PCT_LIST, TP_PCT_LIST,
        MIN_THRES_LIST, MAX_THRES_LIST,
        PAST_LIST,
        MIN_RSI_LIST, MAX_RSI_LIST
    ):
        if minrsi >= maxrsi:
            continue  # 排除非法 RSI 范围
        if mn >= mx:
            continue  # 趋势区间非法
        p = BASE_PARAMS.copy()
        p.update(dict(
            WIN=w, LOOKBACK=lb, SIGMA_LVL=sg,
            STOP_LOSS_PCT=sl, TAKE_PROFIT_PCT=tp,
            MIN_THRES=mn,  MAX_THRES=mx,  PAST=past,
            MIN_RSI=minrsi, MAX_RSI=maxrsi
        ))
        grid.append(p)
    return grid

# ============= 进度条兼容（tqdm_joblib） =============
try:
    from tqdm.contrib import tqdm_joblib
except (ImportError, AttributeError):
    # 轻量 fallback：不依赖 tqdm.contrib
    from joblib import parallel as _joblib_parallel
    @contextmanager
    def tqdm_joblib(tqdm_object):
        class _TqdmBatchCallback(_joblib_parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)
        old_cb = _joblib_parallel.BatchCompletionCallBack
        _joblib_parallel.BatchCompletionCallBack = _TqdmBatchCallback
        try:
            yield tqdm_object
        finally:
            _joblib_parallel.BatchCompletionCallBack = old_cb
            tqdm_object.close()

# ============= 主流程：并行网格搜索、排序、输出 Top-N =============
def grid_search(topn: int = 5, n_jobs: int = -1):
    grid = build_param_grid()
    print(f"Total combos: {len(grid)}")

    with tqdm_joblib(tqdm(desc="Grid Search Progress", total=len(grid))) as _:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_once)(p) for p in grid
        )

    results = [r for r in results if r]
    if not results:
        raise RuntimeError("没有有效的参数组合（score 为 NaN 或无交易）")

    # 排序：按组合 Sharpe，其次按年化收益
    results.sort(key=lambda d: (d["score"], d["annual_ret"]), reverse=True)

    best = results[0]
    print("\n==== Best Parameter Set (by portfolio Sharpe) ====")
    p = best["params"]
    print(f"WIN={p['WIN']:<2} LB={p['LOOKBACK']:<4} SG={p['SIGMA_LVL']:<3} "
          f"SL={p['STOP_LOSS_PCT']:.3f} TP={p['TAKE_PROFIT_PCT']:.3f} "
          f"MIN={p['MIN_THRES']:.3f} MAX={p['MAX_THRES']:.3f} PAST={p['PAST']:<2} "
          f"RSI=({p['MIN_RSI']}-{p['MAX_RSI']}) "
          f"| Sharpe={best['sharpe']:.3f} AnnRet={best['annual_ret']:.2%} "
          f"MDD={best['max_dd']:.2%} WinRate={best['win_rate']:.2%} Trades={best['trades']}")
    # 也可返回结果列表，后续导出 CSV
    return results


# ── Cell #15 (code) ──
# 运行
if __name__ == "__main__":
    _ = grid_search(topn=5, n_jobs=-1)


# ── Cell #16 (code) ──

