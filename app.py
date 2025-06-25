import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="入札参加予測システム",
    page_icon="🏗️"
)

# 認証設定
names = ['顧客A', '顧客B', 'admin']
usernames = ['customer_a', 'customer_b', 'admin']

# パスワードをハッシュ化（実際の運用では事前に生成）
passwords = ['password123', 'customer456', 'admin789']
hashed_passwords = stauth.Hasher(passwords).generate()

credentials = {
    'usernames': {
        usernames[0]: {
            'name': names[0],
            'password': hashed_passwords[0]
        },
        usernames[1]: {
            'name': names[1], 
            'password': hashed_passwords[1]
        },
        usernames[2]: {
            'name': names[2],
            'password': hashed_passwords[2]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    'some_cookie_name',
    'some_signature_key',
    cookie_expiry_days=30
)

# ログイン処理
name, authentication_status, username = authenticator.login('ログイン', 'main')

if authentication_status == False:
    st.error('ユーザー名またはパスワードが間違っています')
elif authentication_status == None:
    st.warning('ユーザー名とパスワードを入力してください')
elif authentication_status:
    # ログイン成功時の処理
    
    # ログアウトボタン
    authenticator.logout('ログアウト', 'sidebar')
    st.sidebar.write(f'ようこそ *{name}* さん')
    
    # タイトル
    st.title("🏗️ 入札参加予測システム")
    
    # モデルフォルダのパス（本番環境では相対パスに変更）
    MODEL_FOLDER = "models_2"  # 相対パスに変更
    DATA_PATH = "入札データ_処理済み.csv"  # 相対パスに変更
    
    # 対象企業リスト
    target_companies = [
        '金本建設（株）', '北沢建設（株）', '（株）カリス', '南信土木建築（有）', 
        '神稲建設（株）', '（株）パテック', '（株）三六組', '（株）トライネット',
        '（有）北原土木', '木下工業（株）', 'クラウニング（株）', '木下建設（株）',
        '細澤建設（株）', '小木曽建設（株）', '（株）清信建設興業', '（株）シノダ',
        '小池建設（株）', '宮嘉建設（有）', '吉川建設（株）', '（有）福士組',
        '長豊建設（株）', '道栄建設（株）', '伊賀良建設（株）', '（有）畑中工務所',
        '（株）尾畑組', '（有）今村工務所', '阿智工務店（株）', '勝間田建設（株）',
        '古田工業（株）', '大協建設（株）', '（株）下平組', '三建建設（有）',
        '高本建設（株）', '（有）新野工務店', '（有）竹村工務所', '（株）サンテクト',
        '（株）吉野組', '太田土建（有）', '池端工業（株）', '（株）近藤工務店',
        '矢木コーポレーション（株）', '（有）新井工務店', '共歩（株）', '藤井興業（株）',
        '南部建設（有）', '宮下建設（有）', '（株）金田組', '大平建設（株）',
        '（有）雄長組', '（株）南建設', '長野機材（株）', '山崎建設（株）',
        '（株）片桐工務所', '（有）ヒラサワ', '（有）イチバ土木', '（有）尾畑組',
        '飯伊森林組合'
    ]

    def get_probability_level(probability):
        """確率を予測と確率レベルに変換する"""
        if probability < 0.5:
            return "不参加", "低", 0
        elif probability < 0.75:
            return "参加", "中", 1
        else:
            return "参加", "高", 2

    @st.cache_data
    def load_encoders():
        """実際のデータからエンコーダーを作成"""
        try:
            df = pd.read_csv(DATA_PATH)
            df = df.drop(columns=['No'], errors='ignore')
            df.fillna('', inplace=True)
            
            def clean_numeric_column(series):
                def convert_to_numeric(x):
                    if pd.isna(x) or x == '' or x == '非公開' or x == '事後公開':
                        return 0
                    try:
                        if isinstance(x, str):
                            x = x.replace(',', '').replace('円', '')
                        return float(x)
                    except (ValueError, TypeError):
                        return 0
                return series.apply(convert_to_numeric)
            
            df['資格点数'] = clean_numeric_column(df['資格点数'])
            df['予定価格（税抜）'] = clean_numeric_column(df['予定価格（税抜）'])
            df['工種リスト'] = df['工種（業種）'].apply(lambda x: str(x).split('・'))
            df['工事概要リスト'] = df['工事概要（業務概要）'].apply(lambda x: str(x).split())
            
            mlb_type = MultiLabelBinarizer()
            mlb_type.fit(df['工種リスト'])
            
            mlb_summary = MultiLabelBinarizer()
            mlb_summary.fit(df['工事概要リスト'])
            
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(df[['入札方式', '発注機関', '工事場所']])
            
            return mlb_type, mlb_summary, ohe
        except Exception as e:
            st.error(f"データ読み込みエラー: {e}")
            return None, None, None

    def load_company_model(company_name):
        """企業別モデルを読み込む"""
        model_filename = f"{company_name.replace('/', '_').replace('（', '(').replace('）', ')')}_smote_xgboost.pkl"
        model_path = os.path.join(MODEL_FOLDER, model_filename)
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except:
            return None

    def create_feature_vector(bidding_method, agency, location, qualification_score, 
                             planned_price, work_type, work_summary, mlb_type, mlb_summary, ohe):
        """特徴量ベクトルを作成"""
        try:
            # 数値特徴量
            X_num = np.array([qualification_score, planned_price])
            
            # 工種・工事概要
            work_type_list = [str(work_type).split('・')]
            summary_list = [str(work_summary).split()]
            
            X_type = mlb_type.transform(work_type_list)
            X_summary = mlb_summary.transform(summary_list)
            
            # カテゴリ
            cat_data = pd.DataFrame({
                '入札方式': [bidding_method],
                '発注機関': [agency],
                '工事場所': [location]
            })
            X_cat = ohe.transform(cat_data)
            
            # 特徴量結合
            X = np.hstack([X_num, X_type.flatten(), X_summary.flatten(), X_cat.flatten()])
            return X
        except Exception as e:
            st.error(f"特徴量作成エラー: {e}")
            return None

    # エンコーダー読み込み
    mlb_type, mlb_summary, ohe = load_encoders()

    if mlb_type is None:
        st.error("データの読み込みに失敗しました")
        st.stop()

    # 入力フォーム
    st.header("案件情報入力")

    col1, col2 = st.columns(2)

    with col1:
        bidding_method = st.text_input("入札方式", value="")
        agency = st.text_input("発注機関", value="")
        location = st.text_input("工事場所", value="")

    with col2:
        qualification_score = st.number_input("資格点数", min_value=0, max_value=1500, value=0)
        planned_price = st.number_input("予定価格（円）", min_value=0, value=0, step=1000000)
        work_type = st.text_input("工種", value="")

    work_summary = st.text_area("工事概要", value="", height=100)

    # 予測実行
    if st.button("予測実行"):
        
        # 特徴量作成
        features = create_feature_vector(
            bidding_method, agency, location, qualification_score,
            planned_price, work_type, work_summary, mlb_type, mlb_summary, ohe
        )
        
        if features is None:
            st.error("特徴量の作成に失敗しました")
            st.stop()
        
        # 予測実行
        predictions = []
        
        for company in target_companies:
            model = load_company_model(company)
            
            if model is not None:
                try:
                    features_2d = features.reshape(1, -1)
                    prediction = model.predict(features_2d)[0]
                    probability = model.predict_proba(features_2d)[0][1]
                    
                    prediction_result, prob_level, sort_key = get_probability_level(probability)
                    
                    predictions.append({
                        '企業名': company,
                        '予測': prediction_result,
                        '確率': prob_level,
                        'ソートキー': sort_key,
                        '確率数値': probability
                    })
                except:
                    predictions.append({
                        '企業名': company,
                        '予測': 'エラー',
                        '確率': '-',
                        'ソートキー': -1,
                        '確率数値': 0
                    })
            else:
                predictions.append({
                    '企業名': company,
                    '予測': 'モデルなし',
                    '確率': '-',
                    'ソートキー': -1,
                    '確率数値': 0
                })
        
        # 結果表示
        results_df = pd.DataFrame(predictions)
        results_df = results_df.sort_values('ソートキー', ascending=False)
        
        st.header("予測結果")
        
        # サマリー
        participating = results_df[results_df['予測'] == '参加']
        high_prob = results_df[results_df['確率'] == '高']
        medium_prob = results_df[results_df['確率'] == '中']
        low_prob = results_df[results_df['確率'] == '低']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("参加予測企業数", len(participating))
        with col2:
            st.metric("確率高", len(high_prob))
        with col3:
            st.metric("確率中", len(medium_prob))
        with col4:
            st.metric("確率低", len(low_prob))
        
        # 結果テーブル
        display_df = results_df[['企業名', '予測', '確率']].copy()
        st.dataframe(display_df, use_container_width=True, height=400)