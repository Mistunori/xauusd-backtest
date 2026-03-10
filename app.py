import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="XAUUSD Backtest Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.metric-card {
    background: #111827;
    border: 1px solid #1f2d3d;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}

.metric-label {
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    line-height: 1;
}

.metric-positive { color: #34d399; }
.metric-negative { color: #f87171; }
.metric-neutral  { color: #60a5fa; }
.metric-warning  { color: #fbbf24; }

.regime-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.badge-trend    { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
.badge-reversal { background: #3b1f2b; color: #f472b6; border: 1px solid #db2777; }
.badge-breakout { background: #1a3326; color: #34d399; border: 1px solid #059669; }
.badge-range    { background: #2d2416; color: #fbbf24; border: 1px solid #d97706; }

.target-row {
    display: flex;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1f2d3d;
    font-size: 14px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 24px 0 12px 0;
    border-bottom: 1px solid #1f2d3d;
    padding-bottom: 8px;
}

div[data-testid="stSidebar"] {
    background: #080c18;
    border-right: 1px solid #1f2d3d;
}

div[data-testid="stSidebar"] .stSlider > div {
    color: #e2e8f0;
}

.stFileUploader {
    background: #111827 !important;
    border: 1px dashed #374151 !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 10px 24px;
    width: 100%;
    transition: background 0.2s;
}

.stButton > button:hover {
    background: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# パラメータ定義
# ──────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "adx_trend_min": 25.0,
    "adx_reversal_drop": 5.0,
    "atr_breakout_mult": 1.5,
    "range_lookback": 10,
    "adx_range_max": 20.0,
    "swing_lookback": 5,
    "fvg_min_size_mult": 0.3,
    "zone_lookback": 20,
    "choch_lookback": 5,
    "retest_max_bars": 10,
    "sl_atr_mult": 1.5,
    "be_atr_mult_trend": 1.0,
    "be_atr_mult_reversal": 0.8,
    "be_buffer_atr_mult": 0.15,
    "partial_tp_atr_mult": 2.0,
    "partial_ratio": 0.5,
    "trailing_atr_mult": 1.5,
    "tp_cap_atr_mult": 4.5,
    "max_bars_trend": 48,
    "max_bars_reversal": 24,
    "max_bars_breakout": 20,
    "risk_percent": 1.0,
    "initial_balance": 10000.0,
    "spread_dollar": 0.5,
    "slippage_dollar": 0.1,
    "approve_threshold": 0.20,
}

# ──────────────────────────────────────────────────────────
# インジケーター
# ──────────────────────────────────────────────────────────

def calc_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_s    = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_s + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_s + 1e-10)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx      = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx, plus_di, minus_di

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)

def resample_ohlcv(df, rule):
    r = df.resample(rule, on='timestamp').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna().reset_index()
    return r

def compute_choch_signals(df, n=5):
    """
    CHoCH（チェンジオブキャラクター）を全バーに対して事前計算。
    
    スイング高値/安値の定義：
      前後n本の中で最も高い/低い値を持つバー
    
    CHoCH定義：
      bull_choch[i] = 直前スイング高値をclose確定で上抜けた瞬間
      bear_choch[i] = 直前スイング安値をclose確定で下抜けた瞬間
    
    FVG/Zoneプルバックとの共存を可能にするため
    「現在バーで発生」ではなく「過去N本以内で発生」として
    check_trend_entry等で参照する。
    """
    length   = len(df)
    highs    = df['high'].values
    lows     = df['low'].values
    closes   = df['close'].values

    swing_high_vals = np.full(length, np.nan)
    swing_low_vals  = np.full(length, np.nan)

    # スイング高値/安値を検出（center=True方式、先端はnm本分はNaN）
    for i in range(n, length - n):
        if highs[i] == highs[i-n:i+n+1].max():
            swing_high_vals[i] = highs[i]
        if lows[i] == lows[i-n:i+n+1].min():
            swing_low_vals[i] = lows[i]

    # 直近スイング値を前方伝播し、CHoCH判定
    bull_choch = np.zeros(length, dtype=bool)
    bear_choch = np.zeros(length, dtype=bool)
    last_sh = np.nan
    last_sl = np.nan

    for i in range(1, length):
        # 前バーまでの最新スイング値を更新（look-ahead防止）
        if not np.isnan(swing_high_vals[i-1]):
            last_sh = swing_high_vals[i-1]
        if not np.isnan(swing_low_vals[i-1]):
            last_sl = swing_low_vals[i-1]

        if not np.isnan(last_sh) and closes[i] > last_sh and closes[i-1] <= last_sh:
            bull_choch[i] = True
        if not np.isnan(last_sl) and closes[i] < last_sl and closes[i-1] >= last_sl:
            bear_choch[i] = True

    return bull_choch, bear_choch

def build_indicators(df5m):
    df15m = resample_ohlcv(df5m, '15min')
    df1h  = resample_ohlcv(df5m, '1h')
    for df in [df5m, df15m, df1h]:
        adx, pdi, mdi    = calc_adx(df)
        df['adx']        = adx.values
        df['plus_di']    = pdi.values
        df['minus_di']   = mdi.values
        df['atr']        = calc_atr(df).values
        df['ema21']      = df['close'].ewm(span=21, adjust=False).mean().values
        df['rsi']        = calc_rsi(df['close']).values
        df['atr_ma20']   = df['atr'].rolling(20).mean()
    # 5M足にCHoCH信号を事前計算（FVG/Zoneとのシーケンス判定に使用）
    bull_choch, bear_choch = compute_choch_signals(df5m, n=5)
    df5m['bull_choch'] = bull_choch
    df5m['bear_choch'] = bear_choch
    return df5m, df15m, df1h

# ──────────────────────────────────────────────────────────
# シグナル検出
# ──────────────────────────────────────────────────────────

def detect_fvg(df, i, direction, atr, p):
    lookback = min(30, i)
    min_size = atr * p['fvg_min_size_mult']
    for j in range(i - 1, max(i - lookback, 1), -1):
        if direction == 'buy':
            gl = df['high'].iloc[j-1]
            gh = df['low'].iloc[j+1] if j+1 <= i else df['low'].iloc[j]
            if gh > gl and (gh - gl) >= min_size:
                return (gl, gh)
        else:
            gh = df['low'].iloc[j-1]
            gl = df['high'].iloc[j+1] if j+1 <= i else df['high'].iloc[j]
            if gh > gl and (gh - gl) >= min_size:
                return (gh, gl)
    return None

def detect_zone(df, i, direction, p):
    lookback = p['zone_lookback']
    window = df.iloc[max(0, i-lookback):i]
    if len(window) < 5:
        return None
    if direction == 'buy':
        return float(window['low'].min())
    else:
        return float(window['high'].max())

def detect_choch(df, i, direction, p):
    """
    CHoCH（シーケンス判定）：
    過去choch_window本以内にスイング高値/安値のcloseブレイクが
    発生していたかを確認する。

    FVG/Zoneへのプルバックと共存できる理由：
      CHoCHは数本前に発生 → 現在価格はFVG/Zoneにプルバック済み
      = 同時成立ではなくシーケンスとして判定
    """
    choch_window = 15
    start = max(0, i - choch_window)
    if direction == 'buy':
        return bool(df['bull_choch'].iloc[start:i+1].any())
    else:
        return bool(df['bear_choch'].iloc[start:i+1].any())

def detect_sweep(df, i, direction, p):
    """現在バーで流動性スイープが発生しているか"""
    if i < 10:
        return False
    lb = p['swing_lookback']
    prev = df.iloc[max(0, i-lb*2):i]
    cur  = df.iloc[i]
    atr  = float(df['atr'].iloc[i]) if not pd.isna(df['atr'].iloc[i]) else 1.0
    if direction == 'buy':
        pl = prev['low'].min()
        return float(cur['low']) < pl and float(cur['close']) > pl and (pl - float(cur['low'])) > atr * 0.2
    else:
        ph = prev['high'].max()
        return float(cur['high']) > ph and float(cur['close']) < ph and (float(cur['high']) - ph) > atr * 0.2

def detect_sweep_recent(df, i, direction, p, lookback=8):
    """
    REVERSAL用：過去lookback本以内にスイープが発生していたか。
    スイープ後CHoCH確認 -> FVG/Zoneでエントリーのシーケンスに対応。
    """
    for k in range(max(0, i - lookback), i + 1):
        if detect_sweep(df, k, direction, p):
            return True
    return False

def detect_rsi_div(df, i, direction):
    lb = 20
    if i < lb:
        return False
    wp = df['close'].iloc[i-lb:i+1]
    wr = df['rsi'].iloc[i-lb:i+1]
    if wr.isna().any():
        return False
    if direction == 'sell':
        return float(wp.iloc[-1]) >= float(wp.iloc[:-5].max()) and float(wr.iloc[-1]) < float(wr.iloc[:-5].max())
    else:
        return float(wp.iloc[-1]) <= float(wp.iloc[:-5].min()) and float(wr.iloc[-1]) > float(wr.iloc[:-5].min())

def detect_range(df, i, p):
    lb = p['range_lookback']
    if i < lb:
        return False, 0.0, 0.0
    adx = float(df['adx'].iloc[i])
    w   = df.iloc[i-lb:i+1]
    return adx < p['adx_range_max'], float(w['high'].max()), float(w['low'].min())

# ──────────────────────────────────────────────────────────
# レジーム検出
# ──────────────────────────────────────────────────────────

def get_1h_filter(df1h, ts, p):
    bars = df1h[df1h['timestamp'] <= ts]
    if len(bars) < 5:
        return {'trend_valid': False, 'direction': None, 'adx': 0}
    bar = bars.iloc[-1]
    adx = float(bar['adx']) if not pd.isna(bar['adx']) else 0
    ema = float(bar['ema21']) if not pd.isna(bar['ema21']) else 0
    direction = 'buy' if float(bar['close']) > ema else 'sell'
    return {'trend_valid': adx >= p['adx_trend_min'], 'direction': direction, 'adx': adx}

def get_15m_regime(df15m, ts, h1, p):
    bars = df15m[df15m['timestamp'] <= ts]
    if len(bars) < 20:
        return {'regime': 'RANGE', 'direction': None}
    bar   = bars.iloc[-1]
    adx   = float(bar['adx'])  if not pd.isna(bar['adx'])  else 0
    atr   = float(bar['atr'])  if not pd.isna(bar['atr'])  else 1
    close = float(bar['close'])
    ema21 = float(bar['ema21']) if not pd.isna(bar['ema21']) else close
    atr_ma = float(bar['atr_ma20']) if not pd.isna(bar['atr_ma20']) else atr
    i     = len(bars) - 1

    adx_prev5 = bars['adx'].iloc[-6:-1].mean() if len(bars) >= 6 else adx
    adx_drop  = adx < (adx_prev5 - p['adx_reversal_drop'])

    if adx >= p['adx_trend_min'] and adx_drop:
        h1_dir = h1.get('direction')
        rev_dir = 'sell' if h1_dir == 'buy' else 'buy'
        # RSIダイバージェンスはレジーム判定ゲートから除外
        # → スコアリングに移管（ここで要求するとREVERSAL件数ゼロになる）
        return {'regime': 'REVERSAL', 'direction': rev_dir, 'adx': adx}

    is_range, rh, rl = detect_range(bars, i, p)
    atr_expand = atr > atr_ma * p['atr_breakout_mult']
    if atr_expand and not is_range and len(bars) >= p['range_lookback'] + 5:
        prev_bars = bars.iloc[:-3]
        pi = len(prev_bars) - 1
        was_range, _, _ = detect_range(prev_bars, pi, p)
        if was_range:
            direction = 'buy' if close > rh - (rh - rl) * 0.3 else 'sell'
            return {'regime': 'BREAKOUT', 'direction': direction, 'adx': adx, 'range_high': rh, 'range_low': rl}

    if adx >= p['adx_trend_min'] and h1.get('trend_valid'):
        direction = 'buy' if close > ema21 else 'sell'
        if direction == h1.get('direction'):
            return {'regime': 'TREND', 'direction': direction, 'adx': adx}

    return {'regime': 'RANGE', 'direction': None}

# ──────────────────────────────────────────────────────────
# スコアリング
# ──────────────────────────────────────────────────────────

def session_score(hour, is_breakout=False):
    if 13 <= hour <= 16: return 0.10
    elif hour in [8, 13]: return 0.05
    elif 8 <= hour <= 17: return 0.00
    elif 0 <= hour <= 3:  return -0.10
    else: return -0.20

def score_trend(df5m, i, direction, regime, h1, p):
    score = 0.0
    adx = regime.get('adx', 0)
    if 25 <= adx <= 35: score += 0.10
    elif adx > 35:      score += 0.05
    if h1.get('trend_valid') and h1.get('direction') == direction: score += 0.10
    atr  = float(df5m['atr'].iloc[i])
    fvg  = detect_fvg(df5m, i, direction, atr, p)
    zone = detect_zone(df5m, i, direction, p)
    if fvg and zone: score += 0.15
    rsi = float(df5m['rsi'].iloc[i]) if not pd.isna(df5m['rsi'].iloc[i]) else 50
    if direction == 'buy' and rsi < 60:   score += 0.10
    elif direction == 'sell' and rsi > 40: score += 0.10
    score += session_score(df5m['timestamp'].iloc[i].hour)
    atr_ma = float(df5m['atr_ma20'].iloc[i]) if not pd.isna(df5m['atr_ma20'].iloc[i]) else atr
    ratio  = atr / (atr_ma + 1e-10)
    if 0.8 <= ratio <= 1.5: score += 0.05
    elif ratio > 1.5:        score -= 0.05
    return score

def score_reversal(df5m, i, direction, regime, h1, p):
    score = 0.0
    adx   = regime.get('adx', 0)
    if adx > 35:   score += 0.20
    elif adx >= 25: score += 0.10
    atr   = float(df5m['atr'].iloc[i])
    sweep = detect_sweep(df5m, i, direction, p)
    fvg   = detect_fvg(df5m, i, direction, atr, p)
    zone  = detect_zone(df5m, i, direction, p)
    if sweep and zone: score += 0.15
    if sweep and fvg:  score += 0.15
    bar   = df5m.iloc[i]
    body  = abs(float(bar['close']) - float(bar['open']))
    wt    = float(bar['high']) - max(float(bar['open']), float(bar['close']))
    wb    = min(float(bar['open']), float(bar['close'])) - float(bar['low'])
    if direction == 'buy' and wb > body * 2 and wb > atr * 0.3:   score += 0.10
    elif direction == 'sell' and wt > body * 2 and wt > atr * 0.3: score += 0.10
    atr_ma = float(df5m['atr_ma20'].iloc[i]) if not pd.isna(df5m['atr_ma20'].iloc[i]) else atr
    if atr > atr_ma * 1.5: score += 0.05
    score += session_score(df5m['timestamp'].iloc[i].hour)
    return score

def score_breakout(df5m, i, direction, regime, h1, p):
    score = 0.0
    if h1.get('trend_valid') and h1.get('direction') == direction: score += 0.20
    atr    = float(df5m['atr'].iloc[i])
    atr_ma = float(df5m['atr_ma20'].iloc[i]) if not pd.isna(df5m['atr_ma20'].iloc[i]) else atr
    ratio  = atr / (atr_ma + 1e-10)
    if ratio > 2.0: score += 0.10
    fvg  = detect_fvg(df5m, i, direction, atr, p)
    zone = detect_zone(df5m, i, direction, p)
    if fvg and zone: score += 0.15
    rsi  = float(df5m['rsi'].iloc[i]) if not pd.isna(df5m['rsi'].iloc[i]) else 50
    if direction == 'buy' and rsi > 50:   score += 0.10
    elif direction == 'sell' and rsi < 50: score += 0.10
    rh = regime.get('range_high', 0)
    rl = regime.get('range_low', 0)
    if rh > rl and atr_ma > 0:
        if (rh - rl) / atr_ma < 1.5: score -= 0.15
    score += session_score(df5m['timestamp'].iloc[i].hour, is_breakout=True)
    return score

# ──────────────────────────────────────────────────────────
# エントリーチェック
# ──────────────────────────────────────────────────────────

def check_trend_entry(df5m, i, direction, regime, h1, p):
    if not h1.get('trend_valid'): return None
    if regime.get('adx', 0) < p['adx_trend_min']: return None
    atr   = float(df5m['atr'].iloc[i])
    close = float(df5m['close'].iloc[i])
    fvg   = detect_fvg(df5m, i, direction, atr, p)
    zone  = detect_zone(df5m, i, direction, p)
    in_fvg  = fvg is not None and min(fvg) <= close <= max(fvg)
    in_zone = zone is not None and abs(close - zone) < atr * 0.5
    if not in_fvg and not in_zone: return None
    if not detect_choch(df5m, i, direction, p): return None
    score = score_trend(df5m, i, direction, regime, h1, p)
    if score < p['approve_threshold']: return None
    w = df5m.iloc[max(0,i-p['swing_lookback']*2):i+1]
    sl = (float(w['low'].min()) if direction == 'buy' else float(w['high'].max()))
    return {'sl_price': sl, 'score': score}

def check_reversal_entry(df5m, i, direction, regime, h1, p):
    """
    REVERSAL エントリー条件（シーケンス判定）:
      1. 過去8本以内にスイープ発生
      2. 過去15本以内にCHoCH確認（スイープ後の構造転換）
      3. スコア閾値通過
    スイープとCHoCHを同時要求しないことでREVERSAL件数ゼロを解消。
    """
    if not h1.get('trend_valid'): return None
    # スイープ：過去8本以内（現在バー含む）
    if not detect_sweep_recent(df5m, i, direction, p, lookback=8): return None
    # CHoCH：過去15本以内（スイープ後の転換確認）
    if not detect_choch(df5m, i, direction, p): return None
    # RSIダイバージェンスはスコアリングのみ（ゲートから除外済み）
    score = score_reversal(df5m, i, direction, regime, h1, p)
    if score < p['approve_threshold']: return None
    atr = float(df5m['atr'].iloc[i])
    w   = df5m.iloc[max(0,i-p['swing_lookback']*2):i+1]
    sh  = float(w['high'].max())
    sl  = float(w['low'].min())
    sl_price = (sl - atr * 0.2) if direction == 'buy' else (sh + atr * 0.2)
    return {'sl_price': sl_price, 'score': score}

def check_breakout_entry(df5m, i, direction, regime, h1, breakout_bar, rh, rl, p):
    if i - breakout_bar > p['retest_max_bars']: return None
    atr   = float(df5m['atr'].iloc[i])
    close = float(df5m['close'].iloc[i])
    fvg   = detect_fvg(df5m, i, direction, atr, p)
    zone  = detect_zone(df5m, i, direction, p)
    in_fvg  = fvg is not None and min(fvg) <= close <= max(fvg)
    in_zone = zone is not None and abs(close - zone) < atr * 0.5
    if not in_fvg and not in_zone: return None
    r2 = dict(regime); r2['range_high'] = rh; r2['range_low'] = rl
    score = score_breakout(df5m, i, direction, r2, h1, p)
    if score < p['approve_threshold']: return None
    sl_price = (rh - atr * 1.0) if direction == 'buy' else (rl + atr * 1.0)
    rw = rh - rl
    tp_price = (close + rw) if direction == 'buy' else (close - rw)
    return {'sl_price': sl_price, 'tp_price': tp_price, 'score': score}

# ──────────────────────────────────────────────────────────
# シミュレーション
# ──────────────────────────────────────────────────────────

def simulate_trade(trade, df5m, entry_idx, p):
    atr, direction, regime = trade['atr'], trade['direction'], trade['regime']
    entry, sl, lot = trade['entry_price'], trade['sl_price'], trade['lot_size']
    if regime == 'TREND':
        be_mult, max_bars, use_trail = p['be_atr_mult_trend'], p['max_bars_trend'], True
        fixed_tp = None
    elif regime == 'REVERSAL':
        be_mult, max_bars, use_trail = p['be_atr_mult_reversal'], p['max_bars_reversal'], False
        fixed_tp = trade.get('tp_price')
    else:
        be_mult, max_bars, use_trail = p['be_atr_mult_trend'], p['max_bars_breakout'], False
        fixed_tp = trade.get('tp_price')

    be_applied = partial_closed = trailing_active = False
    partial_pnl = 0.0
    remaining   = lot
    max_price   = entry
    future      = df5m.iloc[entry_idx+1:entry_idx+1+max_bars]

    for idx, bar in future.iterrows():
        high, low = float(bar['high']), float(bar['low'])
        dur = int(idx) - entry_idx
        max_price = max(max_price, high) if direction == 'buy' else min(max_price, low)

        if not be_applied:
            be_t = entry + atr * be_mult if direction == 'buy' else entry - atr * be_mult
            if (direction == 'buy' and high >= be_t) or (direction == 'sell' and low <= be_t):
                buf = atr * p['be_buffer_atr_mult']
                sl  = (entry + buf) if direction == 'buy' else (entry - buf)
                be_applied = True

        if use_trail and not partial_closed:
            pt = entry + atr * p['partial_tp_atr_mult'] if direction == 'buy' else entry - atr * p['partial_tp_atr_mult']
            if (direction == 'buy' and high >= pt) or (direction == 'sell' and low <= pt):
                pu = lot * p['partial_ratio']
                remaining     = lot * (1 - p['partial_ratio'])
                partial_pnl   = (pt - entry) * pu if direction == 'buy' else (entry - pt) * pu
                partial_closed = True
                trailing_active = True

        if trailing_active:
            tsl = (max_price - atr * p['trailing_atr_mult']) if direction == 'buy' else (max_price + atr * p['trailing_atr_mult'])
            sl  = max(sl, tsl) if direction == 'buy' else min(sl, tsl)

        if (direction == 'buy' and low <= sl) or (direction == 'sell' and high >= sl):
            ep  = sl
            pnl = (ep - entry) * remaining if direction == 'buy' else (entry - ep) * remaining
            outcome = 'trailing_sl' if trailing_active else ('be_sl' if be_applied else 'sl_hit')
            return {**trade, 'exit_price': ep, 'pnl': pnl, 'partial_pnl': partial_pnl,
                    'outcome': outcome, 'duration_bars': dur, 'be_applied': be_applied}

        if fixed_tp is not None:
            cap  = atr * p['tp_cap_atr_mult']
            dist = abs(fixed_tp - entry)
            tpc  = (entry + min(dist, cap)) if direction == 'buy' else (entry - min(dist, cap))
            if (direction == 'buy' and high >= tpc) or (direction == 'sell' and low <= tpc):
                pnl = (tpc - entry) * remaining if direction == 'buy' else (entry - tpc) * remaining
                return {**trade, 'exit_price': tpc, 'pnl': pnl, 'partial_pnl': partial_pnl,
                        'outcome': 'tp_hit', 'duration_bars': dur, 'be_applied': be_applied}

    ep  = float(df5m.iloc[min(entry_idx + max_bars, len(df5m)-1)]['close'])
    pnl = (ep - entry) * remaining if direction == 'buy' else (entry - ep) * remaining
    return {**trade, 'exit_price': ep, 'pnl': pnl, 'partial_pnl': partial_pnl,
            'outcome': 'time_exit', 'duration_bars': max_bars, 'be_applied': be_applied}

# ──────────────────────────────────────────────────────────
# メインバックテスト
# ──────────────────────────────────────────────────────────

def run_backtest(df5m_raw, p, progress_bar=None):
    df5m, df15m, df1h = build_indicators(df5m_raw.copy())
    trades, balance, curve = [], p['initial_balance'], [p['initial_balance']]
    pending_bo   = None
    cost = p['spread_dollar'] + p['slippage_dollar']
    total_bars = len(df5m)
    exit_bar = -1  # BUG #3修正: トレード終了バーを追跡

    for i in range(50, total_bars - 1):
        if progress_bar and i % 500 == 0:
            progress_bar.progress(i / total_bars)
        # BUG #3修正: exit_barまでスキップ（ポジション保有中）
        if i <= exit_bar:
            continue

        ts    = df5m['timestamp'].iloc[i]
        close = float(df5m['close'].iloc[i])
        atr   = float(df5m['atr'].iloc[i])
        if pd.isna(atr) or atr <= 0:
            continue

        h1     = get_1h_filter(df1h, ts, p)
        regime = get_15m_regime(df15m, ts, h1, p)
        r      = regime['regime']

        if r == 'RANGE':
            pending_bo = None
            continue

        direction = regime.get('direction')
        if not direction:
            continue

        entry_result = None
        if r == 'TREND':
            entry_result = check_trend_entry(df5m, i, direction, regime, h1, p)
        elif r == 'REVERSAL':
            entry_result = check_reversal_entry(df5m, i, direction, regime, h1, p)
        elif r == 'BREAKOUT':
            rh = regime.get('range_high', 0)
            rl = regime.get('range_low', 0)
            if pending_bo is None or pending_bo.get('direction') != direction:
                pending_bo = {'bar': i, 'direction': direction, 'rh': rh, 'rl': rl}
            if pending_bo:
                entry_result = check_breakout_entry(df5m, i, direction, regime, h1,
                                                    pending_bo['bar'], pending_bo['rh'], pending_bo['rl'], p)
                if entry_result:
                    pending_bo = None

        if not entry_result:
            continue

        sl_price = entry_result['sl_price']
        sl_dist  = abs((close + cost if direction == 'buy' else close - cost) - sl_price)
        if sl_dist <= 0:
            continue

        entry_price = close + cost if direction == 'buy' else close - cost
        risk_dollar = balance * (p['risk_percent'] / 100)
        lot_size    = risk_dollar / sl_dist

        if r == 'REVERSAL':
            w  = df5m.iloc[max(0, i-p['swing_lookback']*3):i+1]
            sh = float(w['high'].max()); sl_sw = float(w['low'].min())
            tp_raw = sl_sw if direction == 'buy' else sh
            cap    = atr * p['tp_cap_atr_mult']
            tp_price = (entry_price + min(abs(tp_raw - entry_price), cap)) if direction == 'buy' \
                       else (entry_price - min(abs(tp_raw - entry_price), cap))
        elif r == 'BREAKOUT':
            tp_price = entry_result.get('tp_price', entry_price + atr * 3.0)
        else:
            tp_price = entry_price + atr * p['tp_cap_atr_mult'] if direction == 'buy' \
                       else entry_price - atr * p['tp_cap_atr_mult']

        trade = {
            'entry_time': str(ts), 'exit_time': '', 'regime': r,
            'direction': direction, 'entry_price': entry_price,
            'sl_price': sl_price, 'tp_price': tp_price,
            'lot_size': lot_size, 'atr': atr, 'score': entry_result['score'],
            'pnl': 0.0, 'partial_pnl': 0.0, 'outcome': '', 'duration_bars': 0, 'be_applied': False
        }
        trade = simulate_trade(trade, df5m, i, p)
        # BUG #3修正: このトレードが終了するバーを記録
        exit_bar = i + trade['duration_bars']
        net   = trade['pnl'] + trade['partial_pnl']
        balance += net
        curve.append(balance)
        trades.append(trade)
        if balance <= 0:
            break

    if progress_bar:
        progress_bar.progress(1.0)
    return trades, curve, df5m

# ──────────────────────────────────────────────────────────
# 結果分析
# ──────────────────────────────────────────────────────────

def analyze(trades, curve, df5m, p):
    if not trades:
        return None
    initial  = p['initial_balance']
    final    = curve[-1]
    net_pnls = [t['pnl'] + t['partial_pnl'] for t in trades]
    wins     = [v for v in net_pnls if v > 0]
    losses   = [v for v in net_pnls if v <= 0]
    total    = len(trades)
    win_rate = len(wins) / total * 100
    gp       = sum(wins)
    gl       = abs(sum(losses)) if losses else 0
    pf       = gp / gl if gl > 0 else float('inf')
    peak     = initial; max_dd = 0.0
    for b in curve:
        if b > peak: peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd: max_dd = dd
    total_ret = (final - initial) / initial * 100
    days      = (pd.to_datetime(df5m['timestamp'].iloc[-1]) - pd.to_datetime(df5m['timestamp'].iloc[0])).days
    months    = max(days / 30, 1)
    monthly   = total_ret / months
    avg_win   = np.mean(wins)   if wins   else 0
    avg_loss  = abs(np.mean(losses)) if losses else 0
    avg_rr    = avg_win / avg_loss if avg_loss > 0 else 0
    tpd       = total / max(days, 1)

    regime_stats = {}
    for t in trades:
        r = t['regime']
        if r not in regime_stats:
            regime_stats[r] = {'total': 0, 'wins': 0, 'pnl': 0.0}
        regime_stats[r]['total'] += 1
        regime_stats[r]['pnl']   += t['pnl'] + t['partial_pnl']
        if t['pnl'] + t['partial_pnl'] > 0:
            regime_stats[r]['wins'] += 1

    outcomes = {}
    for t in trades:
        outcomes[t['outcome']] = outcomes.get(t['outcome'], 0) + 1

    return {
        'total': total, 'win_rate': win_rate, 'pf': pf, 'max_dd': max_dd,
        'total_ret': total_ret, 'monthly': monthly, 'avg_rr': avg_rr,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'tpd': tpd,
        'final': final, 'initial': initial, 'regime_stats': regime_stats,
        'outcomes': outcomes, 'net_pnls': net_pnls, 'days': days
    }

# ──────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────

def metric_card(label, value, color_class="metric-neutral"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """

def main():
    # ── ヘッダー ──
    st.markdown("""
    <h1 style="color:#e2e8f0;font-size:22px;margin-bottom:4px;letter-spacing:-0.5px;">
    📊 XAUUSD BACKTEST ENGINE
    </h1>
    <p style="color:#4b5563;font-size:13px;font-family:'Space Mono',monospace;margin-bottom:24px;">
    Multi-Regime Strategy · TREND / REVERSAL / BREAKOUT
    </p>
    """, unsafe_allow_html=True)

    # ── サイドバー ──
    with st.sidebar:
        st.markdown('<p class="section-header">⚙ PARAMETERS</p>', unsafe_allow_html=True)

        p = dict(DEFAULT_PARAMS)

        st.markdown("**リスク管理**")
        p['risk_percent']    = st.slider("1トレードリスク (%)",  0.5, 3.0, 1.0, 0.1)
        p['initial_balance'] = st.number_input("初期残高 ($)", 1000, 100000, 10000, 1000)
        p['spread_dollar']   = st.slider("スプレッド ($)", 0.1, 2.0, 0.5, 0.1)

        st.markdown("**レジーム判定**")
        p['adx_trend_min']  = st.slider("ADXトレンド閾値",  15.0, 35.0, 25.0, 1.0)
        p['adx_range_max']  = st.slider("ADXレンジ閾値",    10.0, 25.0, 20.0, 1.0)
        p['atr_breakout_mult'] = st.slider("ATRブレイク倍率", 1.0, 3.0, 1.5, 0.1)

        st.markdown("**エグジット**")
        p['sl_atr_mult']        = st.slider("SL ATR倍率",     0.5, 3.0, 1.5, 0.1)
        p['partial_tp_atr_mult'] = st.slider("部分利確 ATR倍率", 1.0, 4.0, 2.0, 0.1)
        p['trailing_atr_mult']  = st.slider("トレーリング ATR倍率", 0.5, 3.0, 1.5, 0.1)
        p['tp_cap_atr_mult']    = st.slider("TP上限 ATR倍率",  2.0, 8.0, 4.5, 0.5)

        st.markdown("**スコアリング**")
        p['approve_threshold'] = st.slider("approveスコア閾値", 0.0, 0.5, 0.20, 0.05)

    # ── ファイルアップロード ──
    uploaded = st.file_uploader(
        "CSVファイルをアップロード（timestamp, open, high, low, close, volume）",
        type=['csv']
    )

    if uploaded is None:
        st.markdown("""
        <div style="background:#111827;border:1px dashed #374151;border-radius:8px;
                    padding:40px;text-align:center;margin-top:16px;">
            <p style="color:#4b5563;font-family:'Space Mono',monospace;font-size:13px;">
                ↑ CSVをアップロードしてバックテストを開始
            </p>
            <p style="color:#374151;font-size:12px;margin-top:8px;">
                5M OHLCV · timestamp / open / high / low / close / volume
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── データ読み込み ──
    try:
        df = pd.read_csv(uploaded)
        df.columns = [c.lower().strip() for c in df.columns]
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        st.error(f"CSVの読み込みに失敗しました: {e}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(metric_card("データ期間", f"{df['timestamp'].iloc[0].strftime('%Y/%m/%d')} 〜 {df['timestamp'].iloc[-1].strftime('%Y/%m/%d')}"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("総バー数", f"{len(df):,}本"), unsafe_allow_html=True)
    with col3:
        days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
        st.markdown(metric_card("期間", f"{days}日間"), unsafe_allow_html=True)

    st.markdown("")

    if st.button("▶ バックテスト実行"):
        prog = st.progress(0)
        status = st.empty()
        status.markdown('<p style="color:#60a5fa;font-family:Space Mono,monospace;font-size:12px;">実行中...</p>', unsafe_allow_html=True)

        with st.spinner(""):
            trades, curve, df5m = run_backtest(df, p, prog)

        status.empty()
        prog.empty()

        if not trades:
            st.warning("トレードが1件もありませんでした。パラメータを調整してください。")
            return

        stats = analyze(trades, curve, df5m, p)

        # ── サマリーメトリクス ──
        st.markdown('<p class="section-header">📈 PERFORMANCE SUMMARY</p>', unsafe_allow_html=True)

        cols = st.columns(4)
        metrics = [
            ("月次リターン", f"{stats['monthly']:+.1f}%",
             "metric-positive" if stats['monthly'] > 0 else "metric-negative"),
            ("勝率", f"{stats['win_rate']:.1f}%",
             "metric-positive" if stats['win_rate'] >= 48 else "metric-warning"),
            ("プロフィットF", f"{stats['pf']:.2f}",
             "metric-positive" if stats['pf'] >= 1.4 else "metric-warning"),
            ("最大DD", f"{stats['max_dd']:.1f}%",
             "metric-positive" if stats['max_dd'] <= 20 else "metric-negative"),
        ]
        for col, (label, val, cls) in zip(cols, metrics):
            with col:
                st.markdown(metric_card(label, val, cls), unsafe_allow_html=True)

        st.markdown("")
        cols2 = st.columns(4)
        metrics2 = [
            ("総トレード数", f"{stats['total']}件", "metric-neutral"),
            ("1日平均", f"{stats['tpd']:.1f}件", "metric-positive" if stats['tpd'] >= 10 else "metric-warning"),
            ("平均RR", f"{stats['avg_rr']:.2f}", "metric-positive" if stats['avg_rr'] >= 1.5 else "metric-warning"),
            ("純損益", f"${stats['final'] - stats['initial']:+,.0f}", "metric-positive" if stats['final'] > stats['initial'] else "metric-negative"),
        ]
        for col, (label, val, cls) in zip(cols2, metrics2):
            with col:
                st.markdown(metric_card(label, val, cls), unsafe_allow_html=True)

        # ── 資産曲線 ──
        st.markdown('<p class="section-header">💹 EQUITY CURVE</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=curve, mode='lines', name='残高',
            line=dict(color='#34d399', width=2),
            fill='tozeroy', fillcolor='rgba(52,211,153,0.05)'
        ))
        fig.add_hline(y=p['initial_balance'], line=dict(color='#374151', dash='dash', width=1))
        fig.update_layout(
            paper_bgcolor='#111827', plot_bgcolor='#111827',
            font=dict(color='#9ca3af', family='Space Mono', size=11),
            margin=dict(l=10, r=10, t=10, b=10), height=280,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='#1f2d3d', zeroline=False),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── レジーム別 + エグジット別 ──
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<p class="section-header">🎯 レジーム別成績</p>', unsafe_allow_html=True)
            regime_colors = {'TREND': '#60a5fa', 'REVERSAL': '#f472b6', 'BREAKOUT': '#34d399'}
            rs = stats['regime_stats']
            rows = []
            for reg, st_d in sorted(rs.items()):
                wr = st_d['wins'] / st_d['total'] * 100 if st_d['total'] > 0 else 0
                rows.append({'レジーム': reg, '件数': st_d['total'],
                             '勝率': f"{wr:.1f}%", 'PnL': f"${st_d['pnl']:+.0f}"})
            if rows:
                rdf = pd.DataFrame(rows)
                st.dataframe(rdf, use_container_width=True, hide_index=True)

            # レジーム別件数バー
            names = list(rs.keys())
            vals  = [rs[n]['total'] for n in names]
            colors = [regime_colors.get(n, '#6b7280') for n in names]
            fig2 = go.Figure(go.Bar(
                x=names, y=vals, marker_color=colors,
                text=vals, textposition='outside', textfont=dict(color='#9ca3af', size=11)
            ))
            fig2.update_layout(
                paper_bgcolor='#111827', plot_bgcolor='#111827',
                font=dict(color='#9ca3af', family='Space Mono', size=11),
                margin=dict(l=10, r=10, t=10, b=10), height=200,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#1f2d3d'),
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.markdown('<p class="section-header">🚪 エグジット種別</p>', unsafe_allow_html=True)
            oc = stats['outcomes']
            oc_colors = {
                'tp_hit': '#34d399', 'trailing_sl': '#60a5fa',
                'be_sl': '#fbbf24', 'sl_hit': '#f87171', 'time_exit': '#9ca3af'
            }
            fig3 = go.Figure(go.Pie(
                labels=list(oc.keys()), values=list(oc.values()),
                marker_colors=[oc_colors.get(k, '#6b7280') for k in oc.keys()],
                hole=0.5, textfont=dict(family='Space Mono', size=10)
            ))
            fig3.update_layout(
                paper_bgcolor='#111827', plot_bgcolor='#111827',
                font=dict(color='#9ca3af', family='Space Mono', size=10),
                margin=dict(l=10, r=10, t=10, b=10), height=260,
                legend=dict(bgcolor='#111827', font=dict(size=10))
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ── 損益分布 ──
        st.markdown('<p class="section-header">📊 損益分布</p>', unsafe_allow_html=True)
        pnls = stats['net_pnls']
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=[v for v in pnls if v > 0], name='利益',
            marker_color='rgba(52,211,153,0.7)', nbinsx=30
        ))
        fig4.add_trace(go.Histogram(
            x=[v for v in pnls if v <= 0], name='損失',
            marker_color='rgba(248,113,113,0.7)', nbinsx=30
        ))
        fig4.update_layout(
            paper_bgcolor='#111827', plot_bgcolor='#111827',
            font=dict(color='#9ca3af', family='Space Mono', size=11),
            margin=dict(l=10, r=10, t=10, b=10), height=220, barmode='overlay',
            xaxis=dict(showgrid=True, gridcolor='#1f2d3d', title='PnL ($)'),
            yaxis=dict(showgrid=True, gridcolor='#1f2d3d'),
            legend=dict(bgcolor='#111827')
        )
        st.plotly_chart(fig4, use_container_width=True)

        # ── 目標達成チェック ──
        st.markdown('<p class="section-header">✅ 目標達成チェック</p>', unsafe_allow_html=True)
        targets = [
            ("月次リターン ≥ 5%",   stats['monthly'] >= 5.0,   f"{stats['monthly']:.1f}%"),
            ("月次リターン ≥ 10%",  stats['monthly'] >= 10.0,  f"{stats['monthly']:.1f}%"),
            ("最大DD ≤ 20%",       stats['max_dd'] <= 20.0,   f"{stats['max_dd']:.1f}%"),
            ("勝率 ≥ 48%",         stats['win_rate'] >= 48.0, f"{stats['win_rate']:.1f}%"),
            ("PF ≥ 1.4",          stats['pf'] >= 1.4,        f"{stats['pf']:.2f}"),
            ("1日平均 ≥ 10件",     stats['tpd'] >= 10.0,     f"{stats['tpd']:.1f}件"),
        ]
        cols3 = st.columns(2)
        for idx, (label, ok, val) in enumerate(targets):
            with cols3[idx % 2]:
                icon = "✅" if ok else "❌"
                color = "#34d399" if ok else "#f87171"
                st.markdown(
                    f'<div style="background:#111827;border:1px solid #1f2d3d;border-radius:6px;'
                    f'padding:10px 14px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-size:13px;color:#9ca3af;">{icon} {label}</span>'
                    f'<span style="font-family:Space Mono,monospace;font-size:14px;font-weight:700;color:{color};">{val}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ── トレードログ ──
        st.markdown('<p class="section-header">📋 トレードログ（直近50件）</p>', unsafe_allow_html=True)
        tdf = pd.DataFrame(trades[-50:])[[
            'entry_time', 'regime', 'direction', 'score',
            'entry_price', 'exit_price', 'outcome', 'duration_bars'
        ]]
        tdf['pnl'] = [t['pnl'] + t['partial_pnl'] for t in trades[-50:]]
        tdf['pnl'] = tdf['pnl'].map(lambda x: f"${x:+.2f}")
        tdf['score'] = tdf['score'].map(lambda x: f"{x:.2f}")
        tdf['entry_price'] = tdf['entry_price'].map(lambda x: f"{x:.2f}")
        tdf['exit_price']  = tdf['exit_price'].map(lambda x: f"{x:.2f}" if x else "-")
        st.dataframe(tdf, use_container_width=True, hide_index=True)

if __name__ == '__main__':
    main()
