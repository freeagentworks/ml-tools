import streamlit as st
import pandas as pd
from pycaret.regression import RegressionExperiment
import matplotlib.pyplot as plt
from PIL import Image

#Instance create
s = RegressionExperiment()

#Models List
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
def plt_evalute(model):
    s.plot_model(model, plot='residuals', display_format='streamlit') 
    s.plot_model(model, plot='feature_all', display_format='streamlit')
    s.plot_model(model, plot='feature', display_format='streamlit')
    s.plot_model(model, plot='learning', display_format='streamlit')
    s.plot_model(model, plot='rfe', display_format='streamlit')
    s.plot_model(model, plot='error', display_format='streamlit')
    s.plot_model(model, plot='manifold', display_format='streamlit')    

    


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

#Train data upload
if trainfile:
    dftrain = pd.read_csv(trainfile)
    #最後の列をとりあえずラベル列として設定
    #データフレームと列名をセッションステートに保存
    st.session_state['data'] = dftrain
    st.session_state['cols'] = dftrain.columns.to_list()
    dftrainview.dataframe(dftrain, height=150)
    
#Valid data upload
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
    
if btnTrain:
    state = st.status("進捗状況確認")
    #setup
    s.setup(data=dftrain, target=labelcol, session_id=123)
    state.success("セットアップ完了")
    #モデルリストを取得
    model_list = s.models()

    #最適モデルを選択
    if chkbest:
        state.text("最適モデル探索中")
        best_model = s.compare_models()
        state.success("ベストモデル決定")
        st.subheader("ベストモデル：")
        st.write(s.pull())
        pred_holdout = s.predict_model(best_model)
        st.subheader("予測結果：")
        st.write(pred_holdout)
        st.write(s.pull())
        print("◾️格納時のモデル：",best_model)
        st.session_state['bestmodel'] = best_model
        s.save_model(best_model, "currentmodel_pipeline")
        state.success("モデルの検証")
        st.subheader("評価指標：")
        plt_evalute(best_model)
        state.success("グラフ表示")
        state.caption("完了しました！")

    #リストからモデル選択
    else:
        modelID = model_list[model_list.Name == lstmodel].index.values[0]
        st.subheader(lstmodel)
        state.text("モデル作成中･･･")
        model = s.create_model(modelID)
        state.success("モデル作成完了")
        st.write(s.pull())
        pred_holdout = s.predict_model(model)
        st.subheader("予測結果：")
        st.write(pred_holdout)
        st.write(s.pull())
        print("◾️格納時のモデル：",model)
        st.session_state['model'] = model
        s.save_model(model, "currentmodel_pipeline")
        state.success("モデルの検証")
        st.subheader("評価指標：")
        plt_evalute(model)
        state.success("グラフ表示")
        state.caption("完了しました！")            
    
    if chktune:
        st.subheader("チューニング後のモデル：")
        #未だチューニングしていないモデル
        usemodel =  None
        if st.session_state['bestmodel'] is not None:
            usemodel = st.session_state['bestmodel']
        else:
            usemodel = st.session_state['model'] 

        state.text("チューニング中･･･")
        tunedmodel = s.tune_model(usemodel)
        st.write(s.pull())
        state.success("チューニング終了")
        pred_holdout = s.predict_model(tunedmodel)
        st.subheader("予測結果：")
        st.write(pred_holdout)
        st.write(s.pull())
        state.success("モデルの検証")
        print("◾️格納時のモデル：",tunedmodel)
        #モデルのシリアライズ
        s.save_model(tunedmodel, "currentmodel_pipeline")
        # #セッション変数にモデルを格納
        # if chkbest:
        #     st.session_state['tunedbestmodel']=tunedmodel
        # else:
        #     st.session_state['tunemodel']=tunedmodel            
        st.subheader("評価指標：")
        plt_evalute(tunedmodel)
        state.success("チューニング後グラフ")
        state.caption("チューニング完了！")
        
if btnPred:
    isTrain = False
    if st.session_state['tunedbestmodel'] is not None:
        isTrain = True
    elif st.session_state['bestmodel'] is not None:
        isTrain = True
    elif st.session_state['tunemodel'] is not None:
        isTrain = True
    elif st.session_state['model'] is not None:
        isTrain = True
    else:
        st.subheader("最初に教師付きデータを学習させて下さい")
        
    if isTrain:
        #保存したモデルを呼び出す
        usemodel = s.load_model("currentmodel_pipeline")
        #予測
        pred = s.predict_model(usemodel, data=dfval)
        #表示
        st.subheader("予測結果付きデータセット：")
        st.dataframe(pred, width=1000)
        st.write(usemodel)
