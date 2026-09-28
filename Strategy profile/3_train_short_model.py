# ── Cell #1 (code) ──
"""
Short-trend reversal model (generic, no time/code)
- 数据来自你“新定义 signal/is_reverse + 新因子”的生成脚本
- 仅使用 signal == -1 的样本（short 趋势）
- StratifiedKFold + OOF 评估，阈值=MCC 或 FPR 约束
"""

import os, json, warnings
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve,
    fbeta_score, precision_recall_curve  # <— NEW
)
from lightgbm.callback import early_stopping, log_evaluation

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
CSV_PATH = r"D:\刘致尧的奇怪文件夹\奇怪的三号（大学）\实习\策略组\price data\reversal_dataset_clean.csv"  # 用新脚本产出的文件
MODEL_DIR = r"D:\models_generic"
os.makedirs(MODEL_DIR, exist_ok=True)

SIDE_FILTER = -1  # 只训练“short 趋势反转”
TARGET_COL = "is_reverse"
N_SPLITS = 5
RANDOM_SEED = 42

DO_SHAP = True

FEATS = [
        "EMA_gap","RSI14_slope5","RSI14_pct_rank",
        "StochRSI_%K","StochRSI_%D","Stoch_K_minus_D","StochRSI_TrendSlope","σ20_z","bb_width","vol_ma20","vol_std20","vol_ma_past_chg","std20_126","ma20_252","std20_252",
        "ATR14","ATR_slope","upper_wick3","lower_wick3","body3",
        "past_ret","ret",
        "beta60","r2_60","ADX14","dip_min_din","slope50","keltner_w","squeeze_on",
        "trend_age_hist","run_ret_hist","dd_from_peak_hist",
        "z_close_100","slope_ma_20","slope_ma_100",
        "adx_down5","price_diff5_atr",
        "vol_of_vol10","vol_ratio60","ma_cross_prox",
    ]
# ===================== LOAD & CLEAN =====================
df = pd.read_csv(CSV_PATH,encoding="gbk")

# 1) 方向筛选
if "signal" not in df.columns:
    raise ValueError("CSV 缺少 'signal' 列。请先用新生成脚本产出数据。")
if SIDE_FILTER is not None:
    df = df[df["signal"] == SIDE_FILTER].copy()

# 2) 目标列检查
if TARGET_COL not in df.columns:
    raise ValueError(f"CSV 缺少目标列 {TARGET_COL}")

# 3) 特征列：自动或手动
if all(col in df.columns for col in FEATS):
    feature_cols = FEATS
else:
    # 自动：除了meta列都当作特征
    meta_cols = {"code", "date", "signal", TARGET_COL}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    # 只保留数值型
    feature_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]

# 基础清理
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols + [TARGET_COL]).copy()
if len(df) < 200:
    raise ValueError("样本过少（<200）")

X = df[feature_cols].astype("float32").values
y = df[TARGET_COL].astype(int).values

print(f"[Info] samples={len(df)}  pos%={y.mean():.3f}  n_features={len(feature_cols)}")

# ===================== Stratified K-Fold =====================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
splits = list(skf.split(X, y))


# ===================== Optuna Objective =====================
def map_params(p):
    return dict(
        objective='binary', metric='auc', verbosity=-1,
        learning_rate=p['lr'],
        num_leaves=p['leaves'],
        max_depth=p['depth'],
        min_child_samples=p['min_child'],
        feature_fraction=p['feat_frac'],
        bagging_fraction=p['bag_frac'],
        bagging_freq=1,
        lambda_l1=p['l1'],
        lambda_l2=p['l2']
    )


def objective(trial):
    p = {
        "lr": trial.suggest_float("lr", 0.005, 0.3, log=True),
        "leaves": trial.suggest_int("leaves", 31, 255),
        "depth": trial.suggest_int("depth", 3, 12),
        "min_child": trial.suggest_int("min_child", 5, 120),
        "feat_frac": trial.suggest_float("feat_frac", 0.4, 1.0),
        "bag_frac": trial.suggest_float("bag_frac", 0.5, 1.0),
        "l1": trial.suggest_float("l1", 0.0, 10.0),
        "l2": trial.suggest_float("l2", 0.0, 20.0),
    }
    params = map_params(p)

    oof_pred = np.full(len(y), np.nan, dtype=float)

    for tr_idx, te_idx in splits:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        pos_ratio = (y_tr == 1).mean()
        scale_pos = (1 - pos_ratio) / max(pos_ratio, 1e-6)
        params_fold = params | {"scale_pos_weight": scale_pos}

        dtr = lgb.Dataset(X_tr, label=y_tr)
        dte = lgb.Dataset(X_te, label=y_te, reference=dtr)

        booster = lgb.train(
            params_fold, dtr,
            num_boost_round=5000,
            valid_sets=[dte], valid_names=["val"],
            callbacks=[
                early_stopping(stopping_rounds=200, first_metric_only=True, verbose=False),
                log_evaluation(period=0)
            ]
        )
        oof_pred[te_idx] = booster.predict(X_te, num_iteration=booster.best_iteration)

    mask = ~np.isnan(oof_pred)
    return roc_auc_score(y[mask], oof_pred[mask])

# ======= 3) functions：在原“Threshold selection”前加入三个函数 =======
def choose_threshold_fpr(y_true, p_prob, fpr_cap=0.05):
    fpr, tpr, thr = roc_curve(y_true, p_prob)
    idx = np.where(fpr <= fpr_cap)[0]
    return float(thr[idx[-1]]) if len(idx) else 0.99


def choose_threshold_fbeta(y_true, p_prob, beta=2.0):
    ps, rs, ths = precision_recall_curve(y_true, p_prob)
    # 注意：ths 长度 = len(ps) - 1
    best, best_t = -1.0, 0.5
    for t in ths:
        y_pred = (p_prob >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta)
        if score > best:
            best, best_t = score, t
    return float(best_t), float(best)


def choose_topk_threshold(p_prob, k=0.05):
    # 取概率 Top-k% 作为正类
    t = np.quantile(p_prob, 1.0 - k)
    return float(t)


# （保留原有的 MCC；把你原来的 by_fpr 当别名保留，避免旧调用断裂）
def choose_threshold_mcc(y_true, p_prob):
    ths = np.linspace(0.05, 0.95, 91)
    mccs = [matthews_corrcoef(y_true, (p_prob >= t).astype(int)) for t in ths]
    return float(ths[int(np.argmax(mccs))]), float(np.max(mccs))


def pick_k_for_precision(y, p, target_prec=0.50, ks=None):
    if ks is None:
        ks = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08]
    rows = []
    for k in ks:
        thr = np.quantile(p, 1.0 - k)
        yhat = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
        rows.append([k,
                     precision_score(y, yhat), recall_score(y, yhat),
                     fp / (fp + tn + 1e-8), float(thr), tp, fp])
    dfk = pd.DataFrame(rows, columns=['k', 'prec', 'rec', 'fpr', 'thr', 'tp', 'fp'])
    hit = dfk.query('prec >= @target_prec')
    return (hit.iloc[0] if not hit.empty else dfk.iloc[-1]), dfk  # 选精度达标的最小k


def choose_threshold_multi(y_true, p_prob, ks,
                           min_prec=0.55, max_fpr=0.25, min_rec=0.40, beta=2.0):
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    rows = []
    for k in ks:
        thr = np.quantile(p_prob, 1.0 - k)
        yhat = (p_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
        prec = precision_score(y_true, yhat, zero_division=0)
        rec = recall_score(y_true, yhat, zero_division=0)
        fpr = fp / (fp + tn + 1e-8)
        fbeta = (1 + beta * beta) * prec * rec / (beta * beta * prec + rec + 1e-12)
        rows.append((k, thr, prec, rec, fpr, fbeta))
    df = pd.DataFrame(rows, columns=['k', 'thr', 'prec', 'rec', 'fpr', 'fbeta'])
    cand = df.query('prec >= @min_prec and fpr <= @max_fpr and rec >= @min_rec')
    use = cand if not cand.empty else df
    pick = use.sort_values('fbeta', ascending=False).iloc[0]
    return float(pick.thr), float(pick.k), dict(prec=float(pick.prec), rec=float(pick.rec),
                                                fpr=float(pick.fpr), fbeta=float(pick.fbeta)), df



# ── Cell #2 (code) ──

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, show_progress_bar=True)

best_auc = study.best_value
best_params_lgb = map_params(study.best_params)
print(f"\nBest OOF AUC (short-trend): {best_auc:.6f}")
print("Best params:", study.best_params)

# ===================== Refit CV & Collect OOF =====================
oof_pred = np.full(len(y), np.nan, dtype=float)
fold_models = []
for tr_idx, te_idx in splits:
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    pos_ratio = (y_tr == 1).mean()
    scale_pos = (1 - pos_ratio) / max(pos_ratio, 1e-6)
    params_fold = best_params_lgb | {"scale_pos_weight": scale_pos}

    dtr = lgb.Dataset(X_tr, label=y_tr)
    dte = lgb.Dataset(X_te, label=y_te, reference=dtr)

    booster = lgb.train(
        params_fold, dtr,
        num_boost_round=5000,
        valid_sets=[dte], valid_names=["val"],
        callbacks=[
            early_stopping(stopping_rounds=200, first_metric_only=True, verbose=False),
            log_evaluation(period=0)
        ]
    )
    fold_models.append(booster)
    oof_pred[te_idx] = booster.predict(X_te, num_iteration=booster.best_iteration)

mask = ~np.isnan(oof_pred)
y_oof = y[mask];
p_oof = oof_pred[mask]

# ======= 2) CONFIG：新增统一阈值策略与参数 =======
THRESHOLD_MODE = "target"  # "mcc" | "fpr" | "fbeta" | "topk" | topk_auto | "target"
KS_GRID = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
MIN_PREC_TARGET = 0.62  # a bit below your current 0.705 to gain recall
MAX_FPR_TARGET = 0.25  # allow slightly more exploration
MIN_REC_TARGET = 0.40  # new
FBETA_BETA = 2.0  # tilt toward recall
FPR_CAP_FOR_THRESHOLD = 0.231  # 默认 10%，比 5% 更不至于过严
TOPK_RATE = 0.2  # Top-k 比例（5%）
MIN_PRECISION_FOR_TOPK = 0.50  # 目标精度
k_sel = None

# ======= 4) 选择阈值：替换你原来的阈值选择分支 =======
if THRESHOLD_MODE == "mcc":
    best_thr, best_mcc = choose_threshold_mcc(y_oof, p_oof)
    print(f"\nChosen threshold [MCC-max]: {best_thr:.3f} | MCC={best_mcc:.3f}")

elif THRESHOLD_MODE == "fpr":
    best_thr = choose_threshold_fpr(y_oof, p_oof, fpr_cap=FPR_CAP_FOR_THRESHOLD)
    # 为了直观看下最终 FPR
    y_tmp = (p_oof >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_oof, y_tmp).ravel()
    fpr_val = fp / (fp + tn + 1e-8)
    print(f"\nChosen threshold [FPR<={FPR_CAP_FOR_THRESHOLD:.0%}]: {best_thr:.3f} | OOF FPR={fpr_val:.3%}")

elif THRESHOLD_MODE == "fbeta":
    best_thr, best_fbeta = choose_threshold_fbeta(y_oof, p_oof, beta=FBETA_BETA)
    print(f"\nChosen threshold [Fβ, β={FBETA_BETA}]: {best_thr:.3f} | best Fβ={best_fbeta:.3f}")

elif THRESHOLD_MODE == "topk":
    best_thr = choose_topk_threshold(p_oof, k=TOPK_RATE)
    # 看下实际选中比例
    rate = (p_oof >= best_thr).mean()
    print(f"\nChosen threshold [Top-k, k={TOPK_RATE:.1%}]: {best_thr:.3f} | picked={rate:.1%}")

elif THRESHOLD_MODE == "topk_auto":
    # 自动扫描 k，选出能达到 MIN_PRECISION_FOR_TOPK 的最小 k
    hit_row, dfk = pick_k_for_precision(y_oof, p_oof,
                                        target_prec=MIN_PRECISION_FOR_TOPK,
                                        ks=KS_GRID)
    # 取该行的阈值与指标
    best_thr = float(hit_row['thr'])
    k_sel = float(hit_row['k'])
    prec_sel = float(hit_row['prec'])
    rec_sel = float(hit_row['rec'])
    fpr_sel = float(hit_row['fpr'])

    print(f"\nChosen threshold [Top-k-auto, Precision≥{MIN_PRECISION_FOR_TOPK:.0%}]: "
          f"thr={best_thr:.3f} | k={k_sel:.1%} | "
          f"Prec={prec_sel:.3f} | Rec={rec_sel:.3f} | FPR={fpr_sel:.3%}")
elif THRESHOLD_MODE == "target":
    cap = MAX_FPR_TARGET if MAX_FPR_TARGET is not None else 1.0
    best_thr, k_sel, stats, dfk = choose_threshold_multi(
        y_oof, p_oof, KS_GRID,
        min_prec=MIN_PREC_TARGET, max_fpr=MAX_FPR_TARGET, min_rec=MIN_REC_TARGET, beta=FBETA_BETA
    )
    print(f"[Target-driven 2.0] thr={best_thr:.3f} | k={k_sel:.1%} | "
          f"P={stats['prec']:.3f} R={stats['rec']:.3f} FPR={stats['fpr']:.3%} Fβ={stats['fbeta']:.3f}")
else:
    raise ValueError(f"Unknown THRESHOLD_MODE: {THRESHOLD_MODE}")

# ===================== OOF Metrics =====================
y_pred_oof = (p_oof >= best_thr).astype(int)
print(f"OOF AUC={roc_auc_score(y_oof, p_oof):.3f}  AP={average_precision_score(y_oof, p_oof):.3f}  "
      f"Prec={precision_score(y_oof, y_pred_oof):.3f}  Rec={recall_score(y_oof, y_pred_oof):.3f}  "
      f"F1={f1_score(y_oof, y_pred_oof):.3f}")
tn, fp, fn, tp = confusion_matrix(y_oof, y_pred_oof).ravel()
print(f"TP={tp} | FP={fp} | TN={tn} | FN={fn} | FPR={fp / (fp + tn + 1e-8):.3%}")
meta = {
    "threshold_mode": THRESHOLD_MODE,
    "min_precision_for_topk": MIN_PRECISION_FOR_TOPK,
    "k_selected": (float(k_sel) if k_sel is not None else None),
    "best_threshold": float(best_thr),
    "oof_metrics": {
        "auc": float(roc_auc_score(y_oof, p_oof)),
        "ap": float(average_precision_score(y_oof, p_oof)),
        "precision": float(precision_score(y_oof, (p_oof >= best_thr).astype(int))),
        "recall": float(recall_score(y_oof, (p_oof >= best_thr).astype(int)))
    },
    "features": feature_cols,
    "best_params": study.best_params
}
with open(os.path.join(MODEL_DIR, "shorttrend_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("[Info] saved meta to shorttrend_meta.json")
# ===================== SHAP (optional) =====================
if DO_SHAP:
    try:
        import shap

        abs_shaps = []
        for (_, te_idx), booster in zip(splits, fold_models):
            X_te = X[te_idx]
            explainer = shap.TreeExplainer(booster)
            sv = explainer.shap_values(X_te)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            abs_shaps.append(np.abs(sv).mean(axis=0))
        shap_imp = pd.Series(np.mean(np.vstack(abs_shaps), axis=0), index=feature_cols).sort_values(ascending=False)
        print("\n=== SHAP mean(|value|) on OOF ===")
        print(shap_imp.round(4))
    except Exception as e:
        print("[WARN] SHAP disabled:", e)

# ===================== Final train on ALL data =====================
pos_ratio = (y == 1).mean()
scale_pos = (1 - pos_ratio) / max(pos_ratio, 1e-6)
final_params = best_params_lgb | {"scale_pos_weight": scale_pos}

avg_best_iter = int(np.mean([m.best_iteration for m in fold_models])) if fold_models else 1000
dfinal = lgb.Dataset(X, label=y)
final_model = lgb.train(
    final_params, dfinal,
    num_boost_round=int(1.2 * avg_best_iter),
    valid_sets=[dfinal], valid_names=["train"],
    callbacks=[log_evaluation(period=0)]
)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(MODEL_DIR, f"short_trend_rev_{stamp}.txt")
meta_path  = os.path.join(MODEL_DIR, f"short_trend_rev_{stamp}_meta.json")
final_model.save_model(model_path)
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump({
        "features": feature_cols,
        "best_params": study.best_params,
        "threshold": float(best_thr),
        "n_splits": N_SPLITS,
        "side_filter": SIDE_FILTER,
    }, f, ensure_ascii=False, indent=2)
print("Saved:", model_path)
print("Saved:", meta_path)


# ── Cell #3 (markdown) ──
# 单次测试

# ── Cell #4 (code) ──
from sklearn.metrics import brier_score_loss
# ==== Single-run with your fixed params (use first fold as holdout) ====
FIXED_P = {
    'lr': 0.08239713330035937,
    'leaves': 92,
    'depth': 12,
    'min_child': 82,
    'feat_frac': 0.7644111638331136,
    'bag_frac': 0.7149078735734941,
    'l1': 2.3871958510101665,
    'l2': 18.295480169993
}

def ks_statistic(y_true, p_prob):
    fpr, tpr, _ = roc_curve(y_true, p_prob)
    return float(np.max(tpr - fpr))

params_fixed = map_params(FIXED_P)

# 使用第一个折作为“单次训练”的验证集
tr_idx, te_idx = splits[0]
X_tr, y_tr = X[tr_idx], y[tr_idx]
X_te, y_te = X[te_idx], y[te_idx]

# 类不平衡权重
pos_ratio = (y_tr == 1).mean()
scale_pos = (1 - pos_ratio) / max(pos_ratio, 1e-6)
params_fold = params_fixed | {"scale_pos_weight": scale_pos}

dtr = lgb.Dataset(X_tr, label=y_tr)
dte = lgb.Dataset(X_te, label=y_te, reference=dtr)

booster_single = lgb.train(
    params_fold, dtr,
    num_boost_round=5000,
    valid_sets=[dte], valid_names=["val"],
    callbacks=[
        early_stopping(stopping_rounds=200, first_metric_only=True, verbose=False),
        log_evaluation(period=0)
    ]
)

# 预测与概率级指标
p_val = booster_single.predict(X_te, num_iteration=booster_single.best_iteration)
auc  = roc_auc_score(y_te, p_val)
ap   = average_precision_score(y_te, p_val)
brier= brier_score_loss(y_te, p_val)
ks   = ks_statistic(y_te, p_val)
gini = 2*auc - 1

# 用你前面定义的“target”规则挑阈值（直接复用 KS_GRID / MIN_* / MAX_* / FBETA_BETA）
best_thr, k_sel, stats, _ = choose_threshold_multi(
    y_te, p_val, KS_GRID,
    min_prec=MIN_PREC_TARGET, max_fpr=MAX_FPR_TARGET, min_rec=MIN_REC_TARGET, beta=FBETA_BETA
)

yhat = (p_val >= best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()
prec = precision_score(y_te, yhat, zero_division=0)
rec  = recall_score(y_te, yhat, zero_division=0)
f1   = f1_score(y_te, yhat, zero_division=0)
mcc  = matthews_corrcoef(y_te, yhat)
fpr  = fp / (fp + tn + 1e-8)

print("=== Single-run (fixed params) on first fold ===")
print(f"best_iter={booster_single.best_iteration}")
print(f"AUC={auc:.3f} | AP={ap:.3f} | Brier={brier:.4f} | KS={ks:.3f} | Gini={gini:.3f}")
print(f"[Target] thr={best_thr:.3f} | k={k_sel:.1%} | "
      f"P={stats['prec']:.3f} R={stats['rec']:.3f} FPR={stats['fpr']:.3%} Fβ={stats['fbeta']:.3f}")
print(f"Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
print(f"F1={f1:.3f} | MCC={mcc:.3f} | FPR={fpr:.3%}")

# 特征重要性（gain）
imp_gain = pd.Series(
    booster_single.feature_importance(importance_type='gain'),
    index=feature_cols
).sort_values(ascending=False)

print("\nTop-15 feature importance (gain):")
print(imp_gain.head(15).round(4))


# ── Cell #5 (code) ──

