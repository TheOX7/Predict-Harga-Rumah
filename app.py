import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from streamlit_extras.colored_header import colored_header
from streamlit_echarts import st_echarts
import joblib
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

st.set_page_config(
    page_title='Predict Harga Rumah',
    layout='wide'
)

def enter():
    st.markdown("<br>", unsafe_allow_html=True)

def horizontal_line():
    st.markdown("<hr>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; font-size:24px'>
            <b>
            Prediksi <br> Harga Rumah <br>
            </b>
        </div>
    """, unsafe_allow_html=True)
    
    horizontal_line()
    
    selected_option_menu = option_menu(menu_title=None, 
                        options=["Prediksi Harga", 'Rata-Rata Harga Rumah'], 
                        icons=['house'], 
                        menu_icon="cast", default_index=0
                    )
    
    horizontal_line()
    
    st.markdown("""
        <div style='text-align: center; font-size:20px'>
            <b>Related Links</b> <br>
            <a href="https://www.rumah123.com" style="text-decoration: none;">Data Source</a> <br>
            <a href="https://github.com/TheOX7/KPK-2" style="text-decoration: none;">Github Repository</a>
        </div>
    """, unsafe_allow_html=True)
    
    horizontal_line()
    
enter()

if selected_option_menu == "Prediksi Harga":
    colored_header(
        label="Prediksi Harga Rumah",
        description="",
        color_name="orange-70",
    )     
    
    enter()
        
    # Load dataset
    df = pd.read_csv('data/cleaned_data.csv')

    # Dictionary 'kecamatan' & 'kab/kota'
    kecamatan_mapping = {value: i for i, value in enumerate(df['kecamatan'].sort_values().unique())}
    kab_kota_mapping = {
        'Jakarta Barat': 0, 'Jakarta Pusat': 1, 'Jakarta Selatan': 2,
        'Jakarta Timur': 3, 'Jakarta Utara': 4
    }

    # Input Features
    carport_col, kmr_tidur_col, kmr_mandi_col = st.columns(3)
    with kmr_tidur_col:
        jmlh_kamar_tidur = st.number_input("Jumlah Kamar Tidur", min_value=0, value=0)
    with kmr_mandi_col:
        jmlh_kamar_mandi = st.number_input("Jumlah Kamar Mandi", min_value=0, value=0)
    with carport_col:
        carport = st.number_input("Jumlah Carport", min_value=0, value=0)

    luas_tanah_col, luas_bangunan_col, pasokan_listrik_col = st.columns(3)
    with luas_tanah_col:
        luas_tanah = st.number_input("Luas Tanah", min_value=0, value=0)
    with luas_bangunan_col:
        luas_bangunan = st.number_input("Luas Bangunan", min_value=0, value=0)
    with pasokan_listrik_col:
        pasokan_listrik = st.number_input("Pasokan Listrik (Watt)", min_value=0, value=0)

    keamanan_col, taman_col, kab_kota_col, kecamatan_col = st.columns([1,1,2,2])
    with keamanan_col:
        keamanan = st.radio("Apakah ada keamanan?", ["Tidak ada", "Ada"])
        keamanan = 1 if keamanan == "Ada" else 0
    with taman_col:
        taman = st.radio("Apakah ada taman?", ["Tidak ada", "Ada"])
        taman = 1 if taman == "Ada" else 0
    with kab_kota_col:
        kab_kota_input = st.selectbox("Kab./Kota", df['kab/kota'].sort_values().unique().tolist())
        kab_kota = kab_kota_mapping.get(kab_kota_input, 0)  # Apply mapping 'kab_kota_mapping'
    with kecamatan_col:
        df_filtered = df[df['kab/kota'] == kab_kota_input]  # Filter based on selected kab/kota
        kecamatan_input = st.selectbox("Kecamatan", df_filtered['kecamatan'].sort_values().unique().tolist())
        kecamatan = kecamatan_mapping.get(kecamatan_input, 0)  # Apply mapping 'kecamatan_mapping'

    jarak_rs_col, jarak_sekolah_col, jarak_tol_col = st.columns(3)
    with jarak_rs_col:
        jarak_rs = st.number_input("Jarak Rumah Sakit Terdekat (Km)", min_value=0.0, value=0.0)
    with jarak_sekolah_col:
        jarak_sekolah = st.number_input("Jarak Sekolah Terdekat (Km)", min_value=0.0, value=0.0)
    with jarak_tol_col:
        jarak_tol = st.number_input("Jarak Tol Terdekat (Km)", min_value=0.0, value=0.0)

    # Prepare DataFrame for scaling
    input_data = {
        'jumlah_kamar_tidur': [jmlh_kamar_tidur], 
        'jumlah_kamar_mandi': [jmlh_kamar_mandi], 
        'luas_tanah': [luas_tanah], 
        'luas_bangunan': [luas_bangunan], 
        'carport': [carport], 
        'pasokan_listrik': [pasokan_listrik], 
        'kab/kota': [kab_kota], 
        'kecamatan': [kecamatan], 
        'keamanan': [keamanan], 
        'taman': [taman], 
        'jarak_rs_terdekat': [jarak_rs], 
        'jarak_sekolah_terdekat': [jarak_sekolah], 
        'jarak_tol_terdekat': [jarak_tol]
    }

    input_df = pd.DataFrame(input_data)

    # Min-max scaling on the relevant columns based on the original dataset
    scaler = MinMaxScaler()
    scaled_cols = ['jumlah_kamar_tidur', 'jumlah_kamar_mandi', 'luas_tanah', 'luas_bangunan', 'carport', 'pasokan_listrik', 'jarak_rs_terdekat', 'jarak_sekolah_terdekat', 'jarak_tol_terdekat']

    scaler.fit(df[scaled_cols])
    input_df[scaled_cols] = scaler.transform(input_df[scaled_cols])

    if st.button("Predict"):        
        def predict_harga(prediction_scaled):
            min_harga = df['harga'].min()
            max_harga = df['harga'].max()
            prediction = (prediction_scaled * (max_harga - min_harga)) + min_harga
            
            prediction = round(prediction)
        
            if prediction >= 1000000000:
                prediction_str = f"Rp {prediction/1000000000:.2f} Miliar"
            elif prediction >= 1000000:
                prediction_str = f"Rp {prediction/1000000:.2f} Juta"
            else:
                prediction_str = f"Rp {prediction:.2f}"
                
            return prediction_str
        
        # Load linear regression model 
        model = joblib.load('xgb_model.joblib')
        prediction_scaled = model.predict(input_df)[0]
        
        harga_prediction = predict_harga(prediction_scaled)
        st.success(f"Prediksi Harga Rumah (XGBoost) = {harga_prediction}")
            
    
if selected_option_menu == "Rata-Rata Harga Rumah":
    colored_header(
        label="Rata-rata Harga Rumah per Kota/Kab.",
        description="",
        color_name="orange-70",
    )     
      
    df = pd.read_csv('data/cleaned_data.csv')
    
    # Menghitung rata-rata harga berdasarkan kota/kab (dalam satuan miliar rupiah)
    mean_by_kota = df.groupby('kab/kota')['harga'].mean() / 1000000000  # Ubah ke satuan miliar rupiah
    mean_by_kota = mean_by_kota.reset_index()\
                        .sort_values(by='harga', ascending=True)  
    mean_by_kota = round(mean_by_kota, 2)         
                     
    options = {
        "yAxis": {
            "type": "category",
            "data": mean_by_kota['kab/kota'].tolist(),
            "axisTick": {"alignWithLabel": True},  
        },
        "xAxis": {
            "type": "value",
            "axisLabel": {"formatter": "{value} Miliar"}
        },
        "tooltip": {
            "trigger": "axis",
            "formatter": "{b}: {c} Miliar",
            "axisPointer": {"type": "shadow"}
        },
        "series": [
            {
            "data": mean_by_kota['harga'].tolist(), 
            "type": "bar",
            "itemStyle": {"color": "green"}
            }],    }

    st_echarts(options=options, height="500px")

    enter(); enter()

    # Rata-rata Harga Rumah per Kecamatan
    colored_header(
        label="Rata-rata Harga Rumah per Kecamatan",
        description="",
        color_name="orange-70",
    )     
    
    kab_kota_filter = st.selectbox("Select Kota", df['kab/kota'].sort_values().unique().tolist())
    df = df[df['kab/kota'] == kab_kota_filter]
    
    # Menghitung rata-rata harga berdasarkan kecamatan (dalam satuan miliar rupiah)
    mean_by_kecamatan = df.groupby('kecamatan')['harga'].mean() / 1000000000  # Ubah ke satuan miliar rupiah
    mean_by_kecamatan = mean_by_kecamatan.reset_index()\
                        .sort_values(by='harga', ascending=True)  
    mean_by_kecamatan = round(mean_by_kecamatan, 2)         
                     
    options = {
        "yAxis": {
            "type": "category",
            "data": mean_by_kecamatan['kecamatan'].tolist(),
            "axisTick": {"alignWithLabel": True},
            "axisLabel": {
            "interval": 2,  # Menampilkan semua label sumbu Y
        },
        },
        "xAxis": {
            "type": "value",
            "axisLabel": {"formatter": "{value} Miliar"}
        },
        "tooltip": {
            "trigger": "axis",
            "formatter": "{b}: {c} Miliar",
            "axisPointer": {"type": "shadow"}
        },
        "series": [
            {
            "data": mean_by_kecamatan['harga'].tolist(), 
            "type": "bar",
            "itemStyle": {"color": "green"}
            }],
    }

    st_echarts(options=options, height="600px")