import streamlit as st
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image

#
s = TSForecastingExperiment()
#
opm = ["モデルを指定する", "ベストモデル自動選択[時間がかかります！]"]
#
mlist = ['Naive Forecaster',
        'Grand Means Forecaster',
        'Seasonal Naive Forecaster',
        'Polynomial Trend Forecaster',
        'ARIMA',
        'Auto ARIMA',
        'Exponential Smoothing',
        'ETS',
        'Theta Forecaster',
        'STLF',
        'Croston',
        'BATS',
        'TBATS',
        'Prophet',
        'Linear w/ Cond. Deseasonalize & Detrending',
        'Elastic Net w/ Cond. Deseasonalize & Detrending',
        'Ridge w/ Cond. Deseasonalize & Detrending',
        'Lasso w/ Cond. Deseasonalize & Detrending',
        'Lasso Least Angular Regressor w/ Cond. Deseasonalize & Detrending',
        'Bayesian Ridge w/ Cond. Deseasonalize & Detrending',
        'Huber w/ Cond. Deseasonalize & Detrending',
        'Orthogonal Matching Pursuit w/ Cond. Deseasonalize & Detrending',
        'K Neighbors w/ Cond. Deseasonalize & Detrending',
        'Decision Tree w/ Cond. Deseasonalize & Detrending',
        'Random Forest w/ Cond. Deseasonalize & Detrending',
        'Extra Trees w/ Cond. Deseasonalize & Detrending',
        'Gradient Boosting w/ Cond. Deseasonalize & Detrending',
        'AdaBoost w/ Cond. Deseasonalize & Detrending',
        'Extreme Gradient Boosting w/ Cond. Deseasonalize & Detrending',
        'Light Gradient Boosting w/ Cond. Deseasonalize & Detrending']




##########################################
def plt_chart(df, pred):
    #実データプロット
    plt.plot(df.index, df.values, label="実データ")
    #予測データプロット
    plt.plot(pred.index, pred["y_pred"].values, label="予測データ")
    plt.title('時系列予測チャート')
    plt.xlabel('時系列')
    plt.ylabel(df.columns[0])
    plt.legend()
    return plt    


###########################################
# セッションステートにリストが存在しない場合は初期化
if 'model' not in st.session_state:
    st.session_state['model'] = None

###########################################



st.title("時系列データの予測")

st.text("時系列データを分析します")
image = Image.open("./images/headertimeseries.png")
st.image(image)
st.caption("データ(CSV)をアップロードしてください！")

st.subheader("Step1. 時系列のデータをアップロード")
upfile = st.file_uploader("データをアップロードしてください", type="csv")

#データを表示するフレーム
loaddata = pd.DataFrame()
dfview = st.dataframe(loaddata, width=1000)

col1, col2 = st.columns(2)
with col1:
    selectmtd = st.selectbox("ベストモデル自動選択［時間かかります！］", options=opm, index=0)
    fhspan = st.text_input("予測ホライズン:", value=3)
    span = st.selectbox("時系列選択：", options=["Y","M","D"], index=1) 
with col2:
    modellst = st.selectbox("モデル指定", options=mlist, index=9)
    predspan = st.text_input("予測期間:", value=10)
    chktune = st.checkbox("モデルをチューニングする")

st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #0000ff;  /* 背景色 */
        color: white;  /* 文字色 */
        padding: 15px;  /* パディング */
        text-align: center;  /* テキストを中央揃え */
        text-decoration: none;  /* テキストの下線をなし */
        font-size: 25px;  /* フォントサイズ */
        border-radius: 4px;  /* 角を丸くする */
        cursor: pointer;  /* カーソルをポインタに */
    }
    </style>
    """,
    unsafe_allow_html=True
)    
btmln = st.button("学習 & 予測")

#予測データを表示するフレーム
st.subheader("予測:")
dfpred = pd.DataFrame()
predview = st.dataframe(dfpred, width=1000)


#ファイルがアップロードされたら
if upfile:
    #データの一列目を日付型に変換しリサンプルする（Seriesでなくても良いが一列目は日付型）
    loaddata = pd.read_csv(upfile, index_col=0)
    loaddata.index = pd.to_datetime(loaddata.index)
    loaddata = loaddata.resample(span).sum()
    #セットアップ
    s.setup(data=loaddata, fh=3, session_id=123)
    st.write("セットアップ結果:")
    st.dataframe(s.pull(), width=1000, height=150)
    #データフレームにアップロードデータ表示
    dfview.dataframe(loaddata, width=1000 ,height=100)
    
if btmln:
    #Get ModelID
    modelID = s.models()[s.models().Name==modellst].index.values[0]
    if selectmtd=="モデルを指定する":
        model = s.create_model(modelID)
        #評価指数表示
        st.write("評価指標:")
        st.dataframe(s.pull(), width=1000)
        pred_holdout = s.predict_model(model)
        predview.dataframe(pred_holdout, width=1000, height=150)
        st.session_state['model'] = model
        st.write(model)
        s.plot_model(model, display_format='streamlit')
        #指定期間予想させチャートに表示
        predection = s.predict_model(model, fh=int(predspan))
        st.subheader(f"指定期間{predspan}期間の予測チャート")
        plt = plt_chart(loaddata, predection)
        st.pyplot(plt)
    else:
        best = s.compare_models()
        st.write("モデルの比較結果")
        st.dataframe(s.pull(), width=1000)
        pred_holdout = s.predict_model(best)
        predview.dataframe(pred_holdout, height=150)
        st.write("ベストモデル評価指標")
        st.write(s.pull())
        s.plot_model(best, display_format='streamlit')
        predection = s.predict_model(best, fh=int(predspan))
        plt = plt_chart(loaddata, predection)
        st.pyplot(plt)
        
    if chktune:
        tunedmodel = s.tune_model(st.session_state['model'])
        st.dataframe(s.pull(), width=1000)
        predection = s.predict_model(tunedmodel, fh=int(predspan))
        st.subheader(f"チューニング後{predspan}期間の予測チャート")
        plt = plt_chart(loaddata, predection)
        st.pyplot(plt)        
        
    

