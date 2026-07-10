import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from google import genai
from datetime import datetime, timedelta
import json

# ==========================================
# 頁面基本設定
# ==========================================
st.set_page_config(
    page_title="AI 台股趨勢分析系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 核心功能函數
# ==========================================

def get_stock_data(symbol, api_key, start_date, end_date):
    """F-002: 從 FinMind API 獲取台股歷史數據"""
    fetch_start_date = (pd.to_datetime(start_date) - timedelta(days=100)).strftime('%Y-%m-%d')
    fetch_end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": symbol,
        "start_date": fetch_start_date,
        "end_date": fetch_end_date
    }
    
    if api_key: 
        params["token"] = api_key
        
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        data = response.json()
        
        if data.get("status") != 200 or not data.get("data"):
            st.error(f"API 回應錯誤或查無此股票代碼 ({symbol})。請確認代碼是否正確（台股如: 2330, 0050）。")
            return None
            
        df = pd.DataFrame(data["data"])
        
        df = df.rename(columns={
            "max": "high",
            "min": "low",
            "Trading_Volume": "volume"
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"FinMind API 連線失敗，請檢查網路狀態。詳細錯誤: {str(e)}")
        return None
    except Exception as e:
        st.error(f"資料處理發生未知錯誤: {str(e)}")
        return None

def filter_by_date_range(df, start_date, end_date):
    """根據日期範圍過濾數據"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (df['date'] >= start) & (df['date'] <= end)
    return df.loc[mask].reset_index(drop=True)

def get_moving_averages(df):
    """F-003: 計算移動平均線 (MA5, MA10, MA20, MA60)"""
    df = df.sort_values('date')
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()
    return df

def generate_ai_insights(api_key, symbol, start_date, end_date, price_change, start_price, end_price, df_filtered):
    """F-006: 使用 Google 新版 genai 套件進行專業技術分析 (完全自動偵測模型)"""
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return f"API 用戶端建立失敗，請檢查您的 Gemini API Key: {str(e)}"

    # ---------------------------------------------------------
    # 終極解決方案：自動向 Google 獲取可用模型清單
    # ---------------------------------------------------------
    target_model = "gemini-pro" # 絕對保底預設值
    try:
        available_models = []
        for m in client.models.list():
            available_models.append(m.name)
            
        gemini_models = [m for m in available_models if "gemini" in m.lower()]
        
        if gemini_models:
            # 優先尋找名稱中帶有 flash 的高效能模型
            flash_models = [m for m in gemini_models if "flash" in m]
            if flash_models:
                target_model = flash_models[0]
            else:
                # 找不到 flash 就抓清單裡的第一個可用 gemini 模型
                target_model = gemini_models[0]
                
    except Exception:
        pass # 如果獲取清單失敗，就沿用保底的 gemini-pro

    # 新版 SDK 呼叫時通常不需加 'models/' 前綴
    target_model = target_model.replace('models/', '')

    # 準備給 AI 的 JSON 數據
    df_ai = df_filtered[['date', 'open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MA60']].copy()
    df_ai['date'] = df_ai['date'].dt.strftime('%Y-%m-%d')
    df_ai = df_ai.fillna("N/A")
    data_json = df_ai.to_json(orient='records', force_ascii=False)
    
    system_prompt = """
## 系統角色 (System Message)
你是一位專業的技術分析師，專精於股票技術分析和歷史數據解讀。你的職責包括：

1. 客觀描述股票價格的歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 識別技術面的歷史支撐阻力位
4. 提供純教育性的技術分析知識

重要原則：
- 僅提供歷史數據分析和技術指標解讀，絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度
- 使用專業術語但保持易懂
- 所有分析僅供教育和研究目的
- 強調技術分析的局限性和不確定性
- 使用繁體中文回答

免責聲明：所提供的分析內容純粹基於歷史數據的技術解讀，僅供教育和研究參考，不構成任何投資建議。
"""
    
    user_prompt = f"""
請基於以下股票歷史數據進行深度技術分析：

### 基本資訊
- 股票代號：{symbol}
- 分析期間：{start_date} 至 {end_date}
- 期間價格變化：{price_change:.2f}% (從 NT${start_price:.2f} 變化到 NT${end_price:.2f})

### 完整交易數據
以下是該期間的完整交易數據：
{data_json}

### 分析架構：技術面完整分析
1. 趨勢分析
2. 技術指標分析
3. 價格行為分析
4. 風險評估
5. 市場觀察

分析目標：{symbol}
"""
    
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        response = client.models.generate_content(
            model=target_model,
            contents=full_prompt
        )
        return f"*(成功自動連接並使用模型: **{target_model}**)*\n\n" + response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {str(e)}\n\n(系統嘗試使用的模型為: {target_model}，請確認您的 API Key 權限是否足夠)"

# ==========================================
# 介面與主程式邏輯
# ==========================================

def main():
    st.title("AI 台股趨勢分析系統")
    st.markdown("<hr style='border: 2px solid transparent; border-image: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet); border-image-slice: 1;'>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ 分析設定")
        st.markdown("<hr style='border: 2px solid transparent; border-image: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet); border-image-slice: 1;'>", unsafe_allow_html=True)
        
        symbol = st.text_input("股票代碼 (例如: 2330, 0050)", value="2330").upper()
        
        finmind_api_key = st.text_input("FinMind API Key (選填)", type="password", help="免費用戶可留空")
        gemini_api_key = st.text_input("Gemini API Key (必填)", type="password", help="請輸入 Google AI Studio 的 API 金鑰")
        
        default_end = datetime.today()
        default_start = default_end - timedelta(days=90)
        
        start_date = st.date_input("起始日期", value=default_start)
        end_date = st.date_input("結束日期", value=default_end)
        
        analyze_btn = st.button("📊 進行分析", use_container_width=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        ---
        ### 📢 免責聲明
        本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
        """)

    if analyze_btn:
        if not symbol:
            st.sidebar.error("請輸入股票代碼！")
            return
        if not gemini_api_key:
            st.sidebar.error("請輸入 Gemini API Key 以進行 AI 分析！")
            return
        if start_date > end_date:
            st.sidebar.error("起始日期不能晚於結束日期！")
            return
            
        with st.spinner("正在從 FinMind 獲取台股歷史數據..."):
            df_raw = get_stock_data(symbol, finmind_api_key, start_date, end_date)
            
        if df_raw is not None and not df_raw.empty:
            df_with_ma = get_moving_averages(df_raw)
            df_filtered = filter_by_date_range(df_with_ma, start_date, end_date)
            
            if df_filtered.empty:
                st.warning("所選日期範圍內無交易數據，請擴大日期範圍重試。")
                return
                
            st.info(f"✅ 成功獲取並處理 {symbol} 數據，共 {len(df_filtered)} 筆交易記錄。")

            start_price = df_filtered.iloc[0]['close']
            end_price = df_filtered.iloc[-1]['close']
            price_change_abs = end_price - start_price
            price_change_pct = (price_change_abs / start_price) * 100
            
            st.subheader("📈 基本統計資訊")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("起始價格", f"NT$ {start_price:.2f}")
            with col2:
                st.metric("結束價格", f"NT$ {end_price:.2f}")
            with col3:
                st.metric("期間價格變化", 
                          f"NT$ {end_price:.2f}", 
                          delta=f"{price_change_abs:.2f} ({price_change_pct:.2f}%)")

            st.subheader("📊 股價 K 線圖與技術指標")
            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df_filtered['date'],
                open=df_filtered['open'],
                high=df_filtered['high'],
                low=df_filtered['low'],
                close=df_filtered['close'],
                name='K線'
            ))

            colors = {'MA5': 'blue', 'MA10': 'orange', 'MA20': 'purple', 'MA60': 'red'}
            for ma_name, color in colors.items():
                fig.add_trace(go.Scatter(
                    x=df_filtered['date'], 
                    y=df_filtered[ma_name],
                    mode='lines',
                    name=ma_name,
                    line=dict(color=color, width=1.5)
                ))

            fig.update_layout(
                title=f"台股 {symbol} 歷史股價走勢 ({start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')})",
                xaxis_title="日期",
                yaxis_title="價格 (TWD)",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=600,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🤖 Gemini AI 技術面深度分析")
            with st.spinner("Gemini 正在分析技術指標與歷史走勢，這可能需要幾十秒，請稍候..."):
                ai_report = generate_ai_insights(
                    api_key=gemini_api_key,
                    symbol=symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    price_change=price_change_pct,
                    start_price=start_price,
                    end_price=end_price,
                    df_filtered=df_filtered
                )
            
            st.success("分析完成！")
            st.markdown(f"> **💡 分析報告 ({symbol})**")
            st.write(ai_report)

            st.subheader("📋 最近 10 筆交易日詳細數據")
            df_recent10 = df_filtered.tail(10).sort_values('date', ascending=False).reset_index(drop=True)
            
            df_display = df_recent10[['date', 'open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20', 'MA60']].copy()
            df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
            for col in ['open', 'high', 'low', 'close', 'MA5', 'MA10', 'MA20', 'MA60']:
                df_display[col] = df_display[col].apply(lambda x: f"NT$ {x:.2f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(df_display, use_container_width=True)

if __name__ == "__main__":
    main()