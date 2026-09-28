# ── Cell #1 (code) ──
# -*- coding: utf-8 -*-
"""
Single-horizon reversal labeling (K=5) — merged features
"""

import os, glob, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm

# ========= Paths =========
GOOD_DIR   = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\PARQUET美股\PARQUET"
OUT_DIR    = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\price data"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV    = os.path.join(OUT_DIR, "reversal_dataset.csv")

# ========= Params =========
ANNUALIZE       = 252.0

# Forward-truth (7-day trend) labeling
H_TREND         = 7
R2_FWD_MIN      = 0.35
K_SIGMA         = 0.75
STAMP_COOLDOWN  = 1

# TTE labeling
TTE_CAP         = 60

# Single horizon for "within K days" classification
K_HORIZON       = 5

# Base sampling policy
SAMPLE_POLICY   = "dense"     # "dense" | "stride" | "stamp"
STRIDE          = 2

# Mild winsorization (upper-tail cap for numeric features)
WINSOR_Q        = 0.995

# ====== Extra params that some features may use ======
PAST      = 10           # local return window
MIN_THRES = 0.015        # not directly used here but kept for consistency
MAX_THRES = 0.15
SIGMA_LVL = 2.0

# ========= Helpers =========
def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for cand in ["date", "Date", "__index_level_0__"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand])
            df = df.set_index(cand)
            return df.sort_index()
    raise ValueError("No datetime index or date column found.")

def roll_ols_beta_r2(logp: pd.Series, L: int):
    n = len(logp)
    beta = np.full(n, np.nan, dtype=float)
    r2   = np.full(n, np.nan, dtype=float)
    x = np.arange(n, dtype=float)
    for i in range(L-1, n):
        y  = logp.values[i-L+1:i+1]
        xi = x[i-L+1:i+1]
        X  = np.c_[np.ones(L), xi]
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ b
        sst  = ((y - y.mean())**2).sum()
        ssr  = ((y - yhat)**2).sum()
        beta[i] = b[1]
        r2[i]   = 1.0 - (ssr/sst if sst > 0 else 0.0)
    return beta, r2

def fwd_ols_beta_r2(logp: pd.Series, H: int):
    n = len(logp)
    beta = np.full(n, np.nan, dtype=float)
    r2   = np.full(n, np.nan, dtype=float)
    t = np.arange(H, dtype=float)
    X = np.c_[np.ones(H), t]
    pinv = np.linalg.pinv(X)
    for i in range(0, n - H + 1):
        y = logp.values[i:i+H]
        b = pinv @ y
        yhat = X @ b
        sst = ((y - y.mean())**2).sum()
        ssr = ((y - yhat)**2).sum()
        beta[i] = b[1]
        r2[i]   = 1.0 - (ssr/sst if sst > 0 else 0.0)
    return beta, r2

def compute_trend_forward(df: pd.DataFrame,
                          H_trend: int = 7,
                          r2_min: float = 0.35,
                          k_sigma: float = 0.75,
                          cooldown: int = 1,
                          col_out_prefix: str = "trend7"):
    logp = np.log(df["close"])
    beta, r2 = fwd_ols_beta_r2(logp, H_trend)
    df[f"{col_out_prefix}_beta"] = beta
    df[f"{col_out_prefix}_r2"]   = r2

    if "σ20" not in df.columns:
        df["logret"] = np.log1p(df["close"].pct_change())
        df["σ20"] = df["logret"].rolling(20).std(ddof=0) * np.sqrt(ANNUALIZE)
    sigma_H = (df["σ20"] / np.sqrt(ANNUALIZE)) * np.sqrt(H_trend)

    delta_log = beta * (H_trend - 1)
    mag = np.abs(np.exp(delta_log) - 1.0)
    th  = k_sigma * sigma_H

    strong = (r2 >= r2_min) & (mag >= th)
    up   = (beta > 0)
    down = (beta < 0)
    trend_raw = np.where(strong & up, 1, np.where(strong & down, -1, 0)).astype("int8")

    if H_trend > 1:
        trend_raw[-(H_trend-1):] = 0

    age = np.zeros(len(trend_raw), dtype=np.int32)
    run = 0
    for i in range(len(trend_raw)):
        if trend_raw[i] != 0 and (i == 0 or trend_raw[i] == trend_raw[i-1]):
            run += 1
        else:
            run = 1 if trend_raw[i] != 0 else 0
        age[i] = run if trend_raw[i] != 0 else 0

    stamp = np.zeros_like(trend_raw, dtype=np.int8)
    last = -10**9
    for i, s in enumerate(trend_raw):
        if s != 0 and (i == 0 or trend_raw[i-1] != s) and (i - last) >= cooldown:
            stamp[i] = s
            last = i

    df[f"{col_out_prefix}_raw"]   = trend_raw
    df[f"{col_out_prefix}_age"]   = age
    df[f"{col_out_prefix}_stamp"] = stamp
    return df

def compute_tte_labels_from_trend(df: pd.DataFrame,
                                  raw_col: str   = "trend7_raw",
                                  stamp_col: str = "trend7_stamp",
                                  tte_cap: int | None = 60):
    raw   = df[raw_col].to_numpy()
    stamp = df[stamp_col].to_numpy()
    n = len(df)

    tte = np.full(n, np.nan, dtype=float)
    evt = np.zeros(n, dtype=int)

    next_pos = np.full(n, -1, dtype=int)
    next_neg = np.full(n, -1, dtype=int)

    last = -1
    for i in range(n-1, -1, -1):
        next_pos[i] = last
        if stamp[i] == 1:
            last = i
    last = -1
    for i in range(n-1, -1, -1):
        next_neg[i] = last
        if stamp[i] == -1:
            last = i

    for i in range(n):
        s = raw[i]
        if s == 0:
            continue
        j = next_neg[i] if s == 1 else next_pos[i]
        if j != -1 and j > i:
            t = j - i
            evt[i] = 1
        else:
            t = (n - 1) - i
            evt[i] = 0
        if tte_cap is not None:
            t = min(t, int(tte_cap))
        tte[i] = float(t)

    return tte, evt

def add_state_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    side_hist = np.where(
        (df["beta60"] > 0) & (df["DIp"] > df["DIn"]),  1,
        np.where((df["beta60"] < 0) & (df["DIp"] < df["DIn"]), -1, 0)
    ).astype(np.int8)

    run = 0
    age = np.zeros(len(side_hist), dtype=int)
    for i, s in enumerate(side_hist):
        if s != 0 and (i == 0 or s == side_hist[i-1]):
            run += 1
        else:
            run = 1 if s != 0 else 0
        age[i] = run if s != 0 else 0
    df["trend_age_hist"] = age

    start_idx = np.full(len(df), -1, dtype=int)
    cur = -1
    for i, s in enumerate(side_hist):
        if s == 0:
            cur = -1
        elif i == 0 or s != side_hist[i-1]:
            cur = i
        start_idx[i] = cur

    run_ret = np.full(len(df), np.nan)
    dd_from_peak = np.full(len(df), np.nan)
    for i in range(len(df)):
        if side_hist[i] == 0 or start_idx[i] == -1:
            continue
        s = start_idx[i]
        seg = df["close"].iloc[s:i+1]
        run_ret[i] = seg.pct_change().add(1).prod() - 1.0
        if side_hist[i] == 1:
            peak = seg.max()
            dd_from_peak[i] = seg.iloc[-1] / (peak + eps) - 1.0
        else:
            trough = seg.min()
            dd_from_peak[i] = seg.iloc[-1] / (trough + eps) - 1.0
    df["run_ret_hist"] = run_ret
    df["dd_from_peak_hist"] = dd_from_peak

    for L in (20, 50, 100):
        ma = df["close"].rolling(L).mean()
        sd = df["close"].rolling(L).std(ddof=0)
        df[f"z_close_{L}"]  = (df["close"] - ma) / (sd + eps)
        df[f"slope_ma_{L}"] = ma.diff()

    df["RSI14_slope5"] = df["RSI14"].diff(5)
    df["RSI14_acc5"]   = df["RSI14_slope5"].diff(5)
    return df

# 计算滚动分位（0~1）
def _roll_percent_rank(arr: np.ndarray) -> float:
    s = np.sort(arr)
    return np.searchsorted(s, arr[-1], side='right') / len(arr)


# ========= Build (single-horizon, is_reverse) =========
rows = []
all_files = glob.glob(os.path.join(GOOD_DIR, "*.parquet"))

for fp in tqdm(all_files, desc="Files"):
    code = Path(fp).stem
    try:
        df = pd.read_parquet(fp)
    except Exception as e:
        print(f"[Skip] {code}: read error {e}")
        continue

    df = _ensure_dtindex(df)
    req_cols = {"open", "high", "low", "close"}
    if not req_cols.issubset(df.columns):
        print(f"[Skip] {code}: missing {req_cols - set(df.columns)}")
        continue

    # ===== historical base =====
    df["ret"]    = df["close"].pct_change()
    df["logret"] = np.log1p(df["ret"])
    df["ret1"]   = df["ret"]

    df["MA20"]    = df["close"].rolling(20).mean()
    df["MA50"]    = df["close"].rolling(50).mean()
    df["slope50"] = df["MA50"].diff()

    import ta
    df["RSI14"]   = ta.momentum.RSIIndicator(df["close"], 14).rsi()                               # 别名

    WIN = 20
    # 非年化波动（新：σ20）
    df["σ20"]     = df["logret"].rolling(WIN).std(ddof=0)

    # 布林带宽度（原有 20/2σ）
    ma20_close = df["close"].rolling(WIN).mean()
    sd20_close = df["close"].rolling(WIN).std(ddof=0)
    upper = ma20_close + 2 * sd20_close; lower = ma20_close - 2 * sd20_close
    df["bb_width"]   = (upper - lower) / ma20_close

    # ATR / Keltner / squeeze
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14)
    df["ATR14"]     = atr.average_true_range()
    df["ATR_slope"] = df["ATR14"].diff()

    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    df["keltner_w"] = (2 * df["ATR14"]) / (ema20.replace(0, np.nan))

    k_up = ema20 + 2 * df["ATR14"]
    k_lo = ema20 - 2 * df["ATR14"]
    df["squeeze_on"] = ((upper < k_up) & (lower > k_lo)).astype("int8")

    # Donchian 位置（保留）
    dc_h = df["high"].rolling(WIN).max()
    dc_l = df["low"].rolling(WIN).min()
    df["don_pos20"] = (df["close"] - dc_l) / (dc_h - dc_l + 1e-9)

    # ADX/DI
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["ADX14"] = adx.adx()
    df["DIp"]   = adx.adx_pos()
    df["DIn"]   = adx.adx_neg()

    # 回归斜率/拟合优度
    beta60, r2_60 = roll_ols_beta_r2(np.log(df["close"]), L=60)
    df["beta60"], df["r2_60"] = beta60, r2_60

    # ===== 新增/补齐：两套手工因子 =====
    eps = 1e-8

    # —— Trend / location
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["EMA_gap"]       = (ema12 - ema26) / (ema26 + eps)
    df["slope_ma_20"]   = df["MA20"].diff()

    # —— Momentum / oscillators
    df["RSI14_slope5"]   = df["RSI14"].diff(5)
    W_pct = int(ANNUALIZE)  # 252
    df["RSI14_pct_rank"]  = df["RSI14"].rolling(W_pct).apply(_roll_percent_rank, raw=True)

    # StochRSI (14, 3)
    RSI14 = df["RSI14"]
    rsi_min = RSI14.rolling(14).min()
    rsi_max = RSI14.rolling(14).max()
    stoch_rsi_k = (RSI14 - rsi_min) / (rsi_max - rsi_min + eps)
    df["StochRSI_%K"] = stoch_rsi_k
    df["StochRSI_%D"] = stoch_rsi_k.rolling(3).mean()
    df["Stoch_K_minus_D"]    = df["StochRSI_%K"] - df["StochRSI_%D"]
    df["StochRSI_TrendSlope"] = df["StochRSI_%K"].diff(5)

    # —— Volatility / channels
    df["vol_ma20"] = df["σ20"].rolling(20).mean()
    df["vol_std20"] = df["σ20"].rolling(20).std(ddof=0)
    df["vol_ma_past_chg"] = df["vol_ma20"] - df["vol_ma20"].shift(PAST)

    # σ20 基于长窗的均值/方差与 z 分数
    df["ma20_126"] = df["σ20"].rolling(126).mean()
    df["std20_126"] = df["σ20"].rolling(126).std(ddof=0)
    df["ma20_252"] = df["σ20"].rolling(252).mean()
    df["std20_252"] = df["σ20"].rolling(252).std(ddof=0)
    df["σ20_z"] = (df["σ20"] - df["ma20_126"]) / (df["std20_126"] + eps)

    # —— Risk / candle shape
    upper_wick = (df["high"] - np.maximum(df["open"], df["close"])) / (df["ATR14"] + eps)
    lower_wick = (np.minimum(df["open"], df["close"]) - df["low"]) / (df["ATR14"] + eps)
    body       = (df["close"] - df["open"]).abs() / (df["ATR14"] + eps)
    df["upper_wick3"] = upper_wick.rolling(3).mean()
    df["lower_wick3"] = lower_wick.rolling(3).mean()
    df["body3"]       = body.rolling(3).mean()

    # —— Local return context
    df["past_ret"] = df["close"].pct_change(PAST)

    # —— 你原脚本中的复合/状态类特征（保留）
    df["dip_min_din"]     = df["DIp"] - df["DIn"]
    df["dist_ma20_atr"]   = (df["close"] - df["MA20"]) / (df["ATR14"] + eps)
    df["dist_ma50_atr"]   = (df["close"] - df["MA50"]) / (df["ATR14"] + eps)
    df["adx_down5"]       = df["ADX14"] - df["ADX14"].shift(5)
    df["price_diff5_atr"] = df["close"].diff(5) / (df["ATR14"] + eps)
    df["vol_of_vol10"]    = df["σ20"].rolling(10).std(ddof=0)
    df["vol_ratio60"]     = df["σ20"] / (df["σ20"].rolling(60).median().replace(0, np.nan))
    df["ma_cross_prox"]   = np.abs(df["MA20"] - df["MA50"]) / (df["ATR14"] + eps)

    # 状态衍生
    df = add_state_features(df)

    # ===== forward truth & TTE/event =====
    df = compute_trend_forward(df, H_trend=H_TREND,
                               r2_min=R2_FWD_MIN, k_sigma=K_SIGMA,
                               cooldown=STAMP_COOLDOWN, col_out_prefix="trend7")
    tte, event = compute_tte_labels_from_trend(df,
                                               raw_col="trend7_raw",
                                               stamp_col="trend7_stamp",
                                               tte_cap=TTE_CAP)
    df["tte"]   = tte
    df["event"] = event

    # ===== sampling =====
    raw = df["trend7_raw"].to_numpy()
    if SAMPLE_POLICY == "stamp":
        idxs = np.where(df["trend7_stamp"].values != 0)[0]
    elif SAMPLE_POLICY == "stride":
        idxs = np.where(raw != 0)[0]
        if STRIDE > 1:
            keep = []
            last_side, last_pos = 0, -10**9
            for i in idxs:
                if raw[i] != last_side:
                    keep.append(i); last_side, last_pos = raw[i], i
                else:
                    if i - last_pos >= STRIDE:
                        keep.append(i); last_pos = i
            idxs = np.array(keep, dtype=int)
    else:
        idxs = np.where(raw != 0)[0]
    idxs = [i for i in idxs if np.isfinite(df["tte"].iat[i]) and raw[i] != 0]

    # ===== merged feature list (union) =====
    FEATS = [
        "EMA_gap","RSI14","RSI14_slope5","RSI14_pct_rank",
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

    # ===== labeling for single horizon K=5 =====
    K = int(K_HORIZON)
    for i in tqdm(idxs, desc=f"{code} samples", leave=False):
        date_str = df.index[i].strftime("%Y-%m-%d")
        signal_i = int(raw[i])                 # metadata only
        tte_i    = int(df["tte"].iat[i])
        evt_i    = int(df["event"].iat[i])

        if evt_i == 1:
            y = 1 if tte_i <= K else 0
        else:
            if K <= tte_i:
                y = 0
            else:
                continue  # unknown -> skip

        row = {"code": code, "date": date_str, "signal": signal_i, "is_reverse": int(y)}
        for f in FEATS:
            row[f] = df[f].iat[i] if f in df.columns else np.nan
        rows.append(row)

# ========= Save =========
if not rows:
    print("[INFO] no samples produced.")
else:
    out_df = pd.DataFrame(rows)
    out_df = out_df.replace([np.inf, -np.inf], np.nan).dropna()

    num_cols = out_df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        caps = out_df[num_cols].quantile(WINSOR_Q)
        for c in num_cols:
            cap = caps[c]
            out_df.loc[out_df[c] > cap, c] = cap

    out_df = out_df.sort_values(["code", "date"])
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    n = len(out_df); pos = int(out_df["is_reverse"].sum())
    print(f"[OK] saved → {OUT_CSV} | rows={n} | positives={pos} ({pos/n:.3%})")

CSV_PATH = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\price data\reversal_dataset.csv"

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def drop_duplicate_rows_and_columns(df, feats, signature_round=6, near_dup_corr=0.95, min_overlap=50):
    df = df.copy()
    sub = _coerce_numeric(df[feats].copy(), feats)

    # 1) drop duplicate rows (in feature space)
    sig_rows = sub.round(signature_round).fillna(1e100)
    mask_unique = ~sig_rows.duplicated()
    dropped_rows = (~mask_unique).sum()
    sub = sub.loc[mask_unique]
    df = df.loc[sub.index].copy()

    # 2a) drop identical columns (value signature)
    sig = sub.fillna(1e100).round(signature_round)
    seen, to_drop_cols = {}, []
    for c in sig.columns:
        key = tuple(sig[c].tolist())
        if key in seen:
            to_drop_cols.append(c)
        else:
            seen[key] = c
    if to_drop_cols:
        sub = sub.drop(columns=to_drop_cols)

    # 2b) drop near-duplicate columns by high |corr|
    if near_dup_corr:
        corr = sub.corr(method="pearson", min_periods=min_overlap)
        cols = list(corr.columns)
        to_drop_highcorr = set()
        for i in range(len(cols)):
            if cols[i] in to_drop_highcorr:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_drop_highcorr:
                    continue
                v = corr.iloc[i, j]
                if pd.notna(v) and abs(v) >= near_dup_corr:
                    to_drop_highcorr.add(cols[j])
        if to_drop_highcorr:
            sub = sub.drop(columns=list(to_drop_highcorr))

    kept_cols = list(sub.columns)
    df = df.drop(columns=[c for c in feats if c not in kept_cols], errors="ignore")
    df[kept_cols] = sub[kept_cols]  # aligned by index

    dropped_cols_all = to_drop_cols + [c for c in feats if c not in kept_cols and c not in to_drop_cols]
    print(f"✅ Dropped duplicate rows: {dropped_rows}")
    print(f"✅ Kept {len(kept_cols)}/{len(feats)} features; dropped: {dropped_cols_all}")
    return df, kept_cols

def print_corr_pairs(df, feats, thresholds=(0.95, 0.90), min_overlap=50):
    sub = _coerce_numeric(df[feats].copy(), feats)
    for method in ("pearson", "spearman"):
        corr = sub.corr(method=method, min_periods=min_overlap)
        print(f"\n=== {method.title()} correlations (min_overlap={min_overlap}) ===")
        for thr in thresholds:
            pairs = []
            cols = corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    v = corr.iloc[i, j]
                    if pd.notna(v) and abs(v) >= thr:
                        pairs.append((cols[i], cols[j], float(v)))
            print(f"> |r| ≥ {thr}: {len(pairs)} pairs")
            for a, b, r in pairs:
                print(f"  {a:<20} {b:<20} r={r:.4f}")

# === run ===
df = pd.read_csv(CSV_PATH)
df_clean, kept_feats = drop_duplicate_rows_and_columns(df, FEATS)
print_corr_pairs(df_clean, kept_feats, thresholds=(0.95, 0.9))
df_clean.to_csv(r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\price data\reversal_dataset_clean.csv", index=False)

# ── Cell #2 (code) ──

