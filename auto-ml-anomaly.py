import streamlit as st
import pandas as pd
from pycaret.anomaly import AnomalyExperiment
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image






st.title("異常値の検出")

st.text("異常データを検出します")
image = Image.open("./images/headeranomaly.png")
st.image(image)
st.caption("データ(CSV)をアップロードしてください！")

st.subheader("Step1. 調べたいデータをアップロード")
upfile = st.file_uploader("データをアップロードしてください", type="csv")

#データを表示するフレーム
loaddata = pd.DataFrame()
dfview = st.dataframe(loaddata, width=1000)



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
        font-size: 25px;  /* フォントサイズ */
        border-radius: 4px;  /* 角を丸くする */
        cursor: pointer;  /* カーソルをポインタに */
    }
    </style>
    """,
    unsafe_allow_html=True
)    
btmln = st.button("検出")