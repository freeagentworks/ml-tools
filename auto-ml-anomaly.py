import streamlit as st
import pandas as pd
from pycaret.anomaly import AnomalyExperiment
# import matplotlib.pyplot as plt
# import japanize_matplotlib
from PIL import Image

s = AnomalyExperiment()

mlist = ['Angle-base Outlier Detection',
        'Clustering-Based Local Outlier',
        'Connectivity-Based Local Outlier',
        'Isolation Forest',
        'Histogram-based Outlier Detection',
        'K-Nearest Neighbors Detector',
        'Local Outlier Factor',
        'One-class SVM detector',
        'Principal Component Analysis',
        'Minimum Covariance Determinant',
        'Subspace Outlier Detection',
        'Stochastic Outlier Selection']



#モデルの評価プロット
def plt_evalute(model):
    s.plot_model(model, plot='tsne', display_format='streamlit') 
    s.plot_model(model, plot='umap', display_format='streamlit')




st.title("異常値の検出")

st.caption("異常データを検出します")
image = Image.open("./images/headeranomaly.png")
st.image(image)
st.caption("データ(CSV)をアップロードしてください！")

st.subheader("Step1. 調べたいデータをアップロード")
upfile = st.file_uploader("データをアップロードしてください", type="csv")

#データを表示するフレーム
loaddata = pd.DataFrame()
dfview = st.dataframe(loaddata, width=1000)
#モデルのリストをリストボックスに
modellst = st.selectbox("モデルを選択", options=mlist, index=3)

#書式付き実行ボタン
st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        background-color: #ffd700;  /* 背景色 */
        color: black;  /* 文字色 */
        padding: 15px;  /* パディング */
        text-align: center;  /* テキストを中央揃え */
        text-decoration: none;  /* テキストの下線をなし */
        font-size: 50px;  /* フォントサイズ */
        border-radius: 4px;  /* 角を丸くする */
        cursor: pointer;  /* カーソルをポインタに */
    }
    </style>
    """,
    unsafe_allow_html=True
)
btndetection = st.button("異常値検出")

# #予測データを表示するフレーム
# st.subheader("検出結果：")
# dfpred = pd.DataFrame()
# predview = st.dataframe(dfpred, width=1000)


#ファイルがアップロードされたら
if upfile:
    #セットアップ
    loaddata = pd.read_csv(upfile, index_col=0)
    s.setup(data=loaddata, session_id=123)
    #st.write("セットアップ結果:")
    #st.dataframe(s.pull(), width=1000, height=100)
    
    #データフレームにアップロードデータ表示
    dfview.dataframe(loaddata, width=1000 ,height=200)

#コマンドボタンが押された場合の処理
if btndetection:
    #状態確認コントロール
    state = st.status("進捗状態確認")
    #Get ModelID
    modelID = s.models()[s.models().Name == modellst].index.values[0]
    model = s.create_model(modelID)
    state.success("モデル作成完了")
    result = s.assign_model(model)
    st.subheader("検出結果：")
    st.dataframe(result, width=1000, height=300)
    state.success("異常値予測完了")
    state.text("評価チャート描画中...")
    plt_evalute(model=model)
    state.success("完了")
    
    
    