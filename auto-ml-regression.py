import streamlit as st
import pandas as pd
from pycaret.regression import RegressionExperiment
import matplotlib.pyplot as plt
from PIL import Image

s = RegressionExperiment()

mlist = ['Linear Regression',
        'Lasso Regression',
        'Ridge Regression',
        'Elastic Net',
        'Least Angle Regression',
        'Lasso Least Angle Regression',
        'Orthogonal Matching Pursuit',
        'Bayesian Ridge',
        'Automatic Relevance Determination',
        'Passive Aggressive Regressor',
        'Random Sample Consensus',
        'TheilSen Regressor',
        'Huber Regressor',
        'Kernel Ridge',
        'Support Vector Regression',
        'K Neighbors Regressor',
        'Decision Tree Regressor',
        'Random Forest Regressor',
        'Extra Trees Regressor',
        'AdaBoost Regressor',
        'Gradient Boosting Regressor',
        'MLP Regressor',
        'Light Gradient Boosting Machine',
        'Dummy Regressor']






##########################################################################
# セッションステートにリストが存在しない場合は初期化

if 'cols' not in st.session_state:
    st.session_state['cols'] = []

#使用可能モデルのデータフレーム
if 'dfmdl' not in st.session_state:
    st.session_state['dfmdl'] = None

if 'data' not in st.session_state:
    st.session_state['data'] = None
#
if 'tunedbestmodel' not in st.session_state:
    st.session_state['tunedbestmodel'] = None
    
if 'bestmodel' not in st.session_state:
    st.session_state['bestmodel'] = None
    
if 'tunemodel' not in st.session_state:
    st.session_state['tunemodel'] = None

if 'model' not in st.session_state:
    st.session_state['model'] = None
    
###########################################################################

st.title("教師ありデータで回帰予測！")

st.text("回帰アルゴリズムで数値を予測。")
image = Image.open("./images/headerregression.png")
st.image(image)
st.caption("データ(CSV)をアップロードしてください！")

st.subheader("Step1. 教師データ分類をアップロード！")
#
trainfile = st.file_uploader("教師付きデータをアップロードしてください(*必須)", type="csv")
dftrain = pd.DataFrame()
dftrainview = st.dataframe(dftrain, width=1000)

valfile = st.file_uploader("予測させたいデータをアップロード", type="csv")
dfval = pd.DataFrame()
dfvalview = st.dataframe(dfval, width=1000)

st.subheader("Step2. 正解列を指定して回帰モデル作成")


#
if trainfile:
    dftrain = pd.read_csv(trainfile)
    #最後の列をとりあえずラベル列として設定
    #データフレームと列名をセッションステートに保存
    st.session_state['data'] = dftrain
    st.session_state['cols'] = dftrain.columns.to_list()
    dftrainview.dataframe(dftrain, height=150)
    
if valfile:
    dfval = pd.read_csv(valfile)
    #データフレームと列名をセッションステートに保存
    dfvalview.dataframe(dfval, height=150)

#２列構成ターゲット列選択、学習ボタン
col1, col2 = st.columns(2)
with col1:
    lstcol = st.session_state['cols']
    labelcol = st.selectbox("正解列:", options= st.session_state['cols'], index=len(lstcol)-1)
    chkbest = st.checkbox("最適モデルを自動選択：")
with col2:
    lstmodel = st.selectbox("アルゴリズム選択：", options=mlist, index=22)
    chktune = st.checkbox("モデルのチューニング：")
    

col3, col4 = st.columns(2)
with col3:
    st.markdown(
        """
        <style>
        .stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
            background-color: #006400;  /* 背景色 */
            color: white;  /* 文字色 */
            padding: 15px;  /* パディング */
            text-align: center;  /* テキストを中央揃え */
            text-decoration: none;  /* テキストの下線をなし */
            font-size: 16px;  /* フォントサイズ */
            border-radius: 4px;  /* 角を丸くする */
            cursor: pointer;  /* カーソルをポインタに */
        }
        </style>
        """,
        unsafe_allow_html=True
    )     
    btnTrain = st.button("学習")
with col4:
    st.markdown(
        """
        <style>
        .stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
            background-color: #006400;  /* 背景色 */
            color: white;  /* 文字色 */
            padding: 15px;  /* パディング */
            text-align: center;  /* テキストを中央揃え */
            text-decoration: none;  /* テキストの下線をなし */
            font-size: 16px;  /* フォントサイズ */
            border-radius: 4px;  /* 角を丸くする */
            cursor: pointer;  /* カーソルをポインタに */
        }
        </style>
        """,
        unsafe_allow_html=True
    )       
    btnPred = st.button("予測")