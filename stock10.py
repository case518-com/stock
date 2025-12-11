import requests
import pandas as pd
import pandas_ta as ta
import streamlit as st

# =========================
# 基本設定
# =========================
from datetime import datetime
today = datetime.today().strftime("%Y-%m-%d")
#print(today)

take_profit_mode = "統計分布"
START_DATE = "2025-01-01"
END_DATE = today

# =========================
# 資料抓取與指標
# =========================
def get_stock_data(stock_id, start_date=START_DATE, end_date=END_DATE):
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date
    }
    res = requests.get(url, params=params).json()
    if "data" not in res or len(res["data"]) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(res["data"])
    rename_map = {
        "Trading_Volume": "volume",
        "open": "open",
        "max": "high",
        "min": "low",
        "close": "close",
        "date": "date"
    }
    df = df.rename(columns=rename_map)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df

def add_indicators(df):
    if df.empty or len(df) < 30:
        return df, None, None
    df["RSI_14"] = ta.rsi(df["close"], length=14)
    kdj = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    df = pd.concat([df, kdj], axis=1)
    bb = ta.bbands(df["close"], length=20)
    df = pd.concat([df, bb], axis=1)
    upper_col = next((c for c in df.columns if "BBU_" in c), None)
    lower_col = next((c for c in df.columns if "BBL_" in c), None)
    mid_col   = next((c for c in df.columns if "BBM_" in c), None)
    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["macd_line"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd_line"].ewm(span=9).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["__BBM_COL__"] = mid_col
    return df, upper_col, lower_col

# =========================
# 訊號判斷
# =========================
def check_signals(df, upper_col, lower_col):
    signals = []
    if df.empty or upper_col is None or lower_col is None:
        return signals
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    if last["close"] > last[upper_col]:
        signals.append(f"布林突破偏多 (收盤 {last['close']:.2f} > 上軌 {last[upper_col]:.2f})")
    elif last["close"] < last[lower_col]:
        signals.append(f"布林跌破偏空 (收盤 {last['close']:.2f} < 下軌 {last[lower_col]:.2f})")
    if "STOCHk_14_3_3" in df.columns and "STOCHd_14_3_3" in df.columns:
        if last["STOCHk_14_3_3"] > last["STOCHd_14_3_3"]:
            signals.append(f"KD黃金交叉偏多 (K={last['STOCHk_14_3_3']:.2f}, D={last['STOCHd_14_3_3']:.2f})")
        elif last["STOCHk_14_3_3"] < last["STOCHd_14_3_3"]:
            signals.append(f"KD死亡交叉偏空 (K={last['STOCHk_14_3_3']:.2f}, D={last['STOCHd_14_3_3']:.2f})")
    if "RSI_14" in df.columns:
        if last["RSI_14"] >= 70:
            signals.append(f"RSI超買警示 (RSI={last['RSI_14']:.2f})")
        elif last["RSI_14"] <= 30:
            signals.append(f"RSI超賣警示 (RSI={last['RSI_14']:.2f})")
    if "macd_line" in df.columns and "macd_signal" in df.columns:
        if last["macd_line"] > last["macd_signal"]:
            signals.append(f"MACD黃金交叉偏多 (MACD={last['macd_line']:.2f}, Signal={last['macd_signal']:.2f})")
        elif last["macd_line"] < last["macd_signal"]:
            signals.append(f"MACD死亡交叉偏空 (MACD={last['macd_line']:.2f}, Signal={last['macd_signal']:.2f})")
    if last["volume"] > prev["volume"] * 1.2:
        signals.append(f"放量偏多 (今日量={last['volume']:.0f} > 昨日量{prev['volume']:.0f}×1.2)")
    elif last["volume"] < prev["volume"] * 0.8:
        signals.append(f"量縮偏空 (今日量={last['volume']:.0f} < 昨日量{prev['volume']:.0f}×0.8)")
    return signals

# =========================
# 停利策略（11 種）
# =========================
def calc_take_profit(df, last, prev, mode, stop_loss=None):
    if mode == "前高":
        return round(df["high"].tail(20).max(), 2)
    elif mode == "平均漲幅":
        swings = []
        for i in range(1, len(df)):
            chg = df["close"].iloc[i] - df["close"].iloc[i-1]
            if chg > 0:
                swings.append(chg)
        avg_gain = (sum(swings) / len(swings)) if swings else 0.0
        return round(last["close"] + avg_gain, 2)
    elif mode == "亞當理論":
        lookback = min(10, len(df) - 1)
        if lookback <= 1:
            return round(last["close"], 2)
        recent_drop = df["high"].iloc[-lookback] - df["low"].iloc[-1]
        return round(last["close"] + abs(recent_drop), 2)
    elif mode == "動態停利":
        hi = df["high"].tail(5).max()
        return round(hi * 0.97, 2)
    elif mode == "風險報酬比":
        if stop_loss is None:
            return round(df["high"].tail(20).max(), 2)
        risk = last["close"] - stop_loss
        return round(last["close"] + max(risk, 0) * 2.0, 2)
    elif mode == "ATR波動":
        atr = df["ATR_14"].iloc[-1] if "ATR_14" in df.columns else None
        if atr is None or pd.isna(atr):
            atr = ta.atr(df["high"], df["low"], df["close"], length=14).iloc[-1]
        return round(last["close"] + float(atr) * 2.0, 2)
    elif mode == "回撤":
        max_close = df["close"].max()
        drawdown = max_close - last["close"]
        return round(last["close"] + max(drawdown, 0), 2)
    elif mode == "機器學習":
        avg_pct = df["close"].pct_change().mean()
        est_gain = avg_pct * last["close"] if pd.notna(avg_pct) else 0.0
        return round(last["close"] + est_gain, 2)
    elif mode == "分批":
        return round(last["close"] * 1.05, 2)
    elif mode == "統計分布":
        pct75 = df["close"].pct_change().quantile(0.75)
        pct75 = pct75 if pd.notna(pct75) else 0.05
        return round(last["close"] * (1 + pct75), 2)
    elif mode == "事件驅動":
        return round(last["close"] * 1.08, 2)
    else:
        return None

# =========================
# 交易建議
# =========================
def trading_decision(df, signals, upper_col, mode):
    if df.empty or upper_col is None:
        return "建議：觀望 (資料不足)", None, None, None
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    buy_signals = [s for s in signals if ("偏多" in s or "放量" in s)]
    sell_signals = [s for s in signals if ("偏空" in s or "超買" in s)]
    decision = "建議：觀望 (訊號不足)"
    buy_price = None
    stop_loss = None
    take_profit = None
    mid_col = df["__BBM_COL__"].iloc[-1] if "__BBM_COL__" in df.columns else None
    mid_val = (last[mid_col] if mid_col else None)
    if len(buy_signals) >= 3:
        decision = "建議：買入 (多指標共振)"
        buy_price = round(last[upper_col] * 1.01, 2) if last["close"] > last[upper_col] else round(last["close"], 2)
        stop_loss = round(min(mid_val, prev["low"]), 2) if mid_val is not None else round(prev["low"], 2)
        take_profit = calc_take_profit(df, last, prev, mode, stop_loss=stop_loss)
    elif len(sell_signals) >= 2:
        decision = "建議：停利或賣出 (偏空訊號)"
    return decision, buy_price, stop_loss, take_profit

# =========================
# 回測
# =========================
def backtest_modes(df, upper_col, lower_col, modes):
    results = []
    if df.empty or upper_col is None or lower_col is None or len(df) < 30:
        st.write("資料不足，無法回測")
        return
    for mode in modes:
        trades = []
        position = None
        for i in range(21, len(df)):
            window = df.iloc[:i+1]
            last = window.iloc[-1]
            signals = check_signals(window, upper_col, lower_col)
            decision, buy_price, stop_loss, take_profit = trading_decision(window, signals, upper_col, mode)
            if position is None and "買入" in decision and buy_price:
                position = {
                    "buy_date": last["date"],
                    "buy_price": float(buy_price),
                    "stop_loss": float(stop_loss) if stop_loss is not None else None,
                    "take_profit": float(take_profit) if take_profit is not None else None,
                    "mode": mode
                }
            elif position is not None:
                if position["stop_loss"] is not None and last["low"] <= position["stop_loss"]:
                    trades.append({
                        "buy_date": position["buy_date"],
                        "sell_date": last["date"],
                        "buy_price": position["buy_price"],
                        "sell_price": position["stop_loss"],
                        "result": position["stop_loss"] - position["buy_price"],
                        "mode": position["mode"]
                    })
                    position = None
                elif position["take_profit"] is not None and last["high"] >= position["take_profit"]:
                    trades.append({
                        "buy_date": position["buy_date"],
                        "sell_date": last["date"],
                        "buy_price": position["buy_price"],
                        "sell_price": position["take_profit"],
                        "result": position["take_profit"] - position["buy_price"],
                        "mode": position["mode"]
                    })
                    position = None
        wins = [t for t in trades if t["result"] > 0]
        win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
        max_win = max((t["result"] for t in trades), default=0.0)
        results.append({
            "mode": mode,
            "trades": len(trades),
            "win_rate": win_rate,
            "max_win": max_win
        })
    qualified = [r for r in results if r["win_rate"] >= 80.0]
    if qualified:
        st.write("### 勝率 ≥ 80% 的策略")
        for r in qualified:
            st.write(f"- 模式: {r['mode']}, 交易次數: {r['trades']}, 勝率: {r['win_rate']:.2f}%, 最大獲利: {r['max_win']:.2f}")
    else:
        st.write("勝率 ≥ 80% 的策略：無")
    best = max(results, key=lambda x: x["max_win"]) if results else None
    if best:
        st.write("### 最大獲利排名第 1 的策略")
        st.write(f"模式: {best['mode']}, 交易次數: {best['trades']}, 勝率: {best['win_rate']:.2f}%, 最大獲利: {best['max_win']:.2f}")
    else:
        st.write("無可比較的回測結果")

# =========================
# Streamlit 介面
# =========================
st.title("📈 Stock8 技術分析系統!")

# 使用者輸入股票代號
stock_id = st.text_input("輸入股票代號", "2330")  # 預設值台積電

if st.button("開始分析"):
    df = get_stock_data(stock_id)
    if df.empty:
        st.error("資料抓取失敗或為空")
    else:
        df, upper_col, lower_col = add_indicators(df)
        signals = check_signals(df, upper_col, lower_col)
        decision, buy_price, stop_loss, take_profit = trading_decision(df, signals, upper_col, take_profit_mode)

        st.subheader(f"股票 {stock_id} 最新分析")
        st.write("最新日期:", df["date"].iloc[-1])
        st.write("收盤價:", round(df["close"].iloc[-1], 2))

        st.write("技術指標訊號:")
        for s in signals:
            st.write("-", s)

        st.write("交易建議:", decision)
        if buy_price:
            st.write("建議買入價位:", buy_price)
        if stop_loss:
            st.write("停損價位:", stop_loss)
        if take_profit:
            st.write(f"停利價位 ({take_profit_mode}):", take_profit)

        st.subheader("回測結果")
        all_modes = [
            "前高", "平均漲幅", "亞當理論", "動態停利",
            "風險報酬比", "ATR波動", "回撤", "機器學習",
            "分批", "統計分布", "事件驅動"
        ]
        backtest_modes(df, upper_col, lower_col, all_modes)


