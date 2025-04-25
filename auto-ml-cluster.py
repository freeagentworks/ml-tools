import streamlit as st
import pandas as pd
from pycaret.clustering import *
import matplotlib.pyplot as plt
from PIL import Image


def plot_cluster_charet(model):
    plot_model(model, plot='elbow', display_format='streamlit')
    plot_model(model, plot='cluster', display_format='streamlit')
    plot_model(model, plot='tsne', display_format='streamlit')
    plot_model(model, plot='silhouette', display_format='streamlit')
    plot_model(model, plot='distribution', display_format='streamlit')
    #evaluate_model(model, display_format='streamlit')
    

##########################################################################
# セッションステートにリストが存在しない場合は初期化
if 'data' not in st.session_state:
    st.session_state['data'] = None

if 'model' not in st.session_state:
    st.session_state['model'] = None


###########################################################################

st.title("データをクラスター分類します！")

st.text("顧客のクラス分けなどに使えます")
image = Image.open("./images/headercluster.png")
st.image(image)
st.caption("データセットをアップロードしてください！")

st.subheader("Step1. 分類したいデータ、アップロード！")
upfile = st.file_uploader("データをアップロードしてください", type="csv")

dfview = st.dataframe()

st.subheader("Step2. 分類方法、除外列、分類数を指定し分類")

#２列構成ターゲット列選択、学習ボタン
col1, col2 = st.columns(2)
with col1:
    jogaicol = st.text_input("除外したい列")
    
with col2:
    lstmodel = st.selectbox("分類方法：", options= ['kmeans', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics','birch'],)
    clusternum = st.text_input("分類数を入力：", value=4)
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
btnCluster = st.button("分類")


#
if upfile:
    df = pd.read_csv(upfile)
    st.session_state['data'] = df
    dfview.dataframe(df, height=150)
    setdata = setup(df, session_id = 123, ignore_features=[jogaicol])

#
if btnCluster:
    model = create_model(lstmodel, num_clusters=int(clusternum))
    st.session_state['model'] = model
    st.subheader("分類結果：")
    plot_cluster_charet(model)
    st.subheader("分類済データセット：")
    st.dataframe(assign_model(model))


    

