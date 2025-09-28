from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import os, io, time

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal
from function import ModusImputer,ModusTwoGroups

lokasi_file = Path(__file__).resolve()
# print("Lokasi file:", lokasi_file)
lokasi_folder_utama = lokasi_file.parents[1]
# print("Direktori folder utama:", lokasi_folder_utama)
path_model = lokasi_folder_utama / 'model' / 'logreg_for_marketing_2.sav'
# print("Path lengkap ke model:", path_model)
path_data_clean = lokasi_folder_utama / 'data' / 'bank_marketing_clean.csv'
# print("Path lengkap ke model:", path_data_clean)

with open(path_model, "rb") as f:
    model = pickle.load(f)


st.set_page_config(
    page_title="Bank Marketing Campaign Predict",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈",
)

st.title("Bank Marketing Campaign Customer Predictor App")
st.markdown("""
---
Aplikasi ini memanfaatkan machine learning untuk memprediksi kemungkinan konversi nasabah terhadap produk deposito berjangka, 
sehingga tim pemasaran dapat menargetkan nasabah yang paling potensial, meningkatkan efisiensi kampanye, serta mengoptimalkan penggunaan sumber daya.
""", unsafe_allow_html=True)

st.markdown("""
Model: <br>
**Logistic Regression** <br>
*Conversion Rate (Precision)* : :green[80%],&nbsp; *ROAS* : :green[55x]

""", unsafe_allow_html=True)

tab_single, tab_batch = st.tabs(["Single-Predict", "Multi-Predict"])


#_____________________Single-Predict________________________________

def input_feature_sidebar():
    age = st.sidebar.slider("Age", 17, 98, 35)
    campaign = st.sidebar.slider("Campaign (Contact Count)", 1, 56, 2)
    previous = st.sidebar.slider("Previous Contacts", 0, 7, 0)
    cons_price_idx = st.sidebar.slider("Consumer Price Index", 92.201, 94.767, 93.5)
    cons_conf_idx = st.sidebar.slider("Consumer Confidence Index", -50.8, -26.9, -40.0)
    nr_employed = st.sidebar.slider("Number of Employed", 4963.6, 5228.1, 5170.0)
    previous_contacted = st.sidebar.selectbox("Previously Contacted?", [0, 1])
    
    job = st.sidebar.selectbox("Job", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 
                                        'retired', 'management', 'unemployed', 'self-employed', 
                                        'entrepreneur', 'student'])
    
    marital = st.sidebar.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.sidebar.selectbox("Education", ['basic.4y', 'high.school', 'basic.6y', 
                                                'basic.9y', 'professional.course', 'university.degree'])
    housing = st.sidebar.selectbox("Has Housing Loan?", ['yes', 'no'])
    loan = st.sidebar.selectbox("Has Personal Loan?", ['yes', 'no'])
    contact = st.sidebar.selectbox("Contact Communication Type", ['telephone', 'cellular'])
    month = st.sidebar.selectbox("Last Contact Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 
                                                        'dec', 'mar', 'apr', 'sep'])
    day_of_week = st.sidebar.selectbox("Day of Week Contacted", ['mon', 'tue', 'wed', 'thu', 'fri'])
    poutcome = st.sidebar.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'])
    
    # Susun ke dalam dataframe
    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'campaign': campaign,
        'previous': previous,
        'poutcome': poutcome,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'nr.employed': nr_employed,
        'previous_contacted': previous_contacted
            }

    return pd.DataFrame([data])


st.sidebar.markdown(
    """Made By : <br>
        1. [Muhammad Khisanul Fakhrudin Akbar](https://www.linkedin.com/in/muhammad-khisanul-fakhrudin-akbar/) <br>
        2. [Cindy Handoko Tantowibowo](https://www.linkedin.com/in/cindy-handoko-tantowibowo-55a2751a7/)
    """,unsafe_allow_html=True
)
st.sidebar.header("Single Input Feature:")
st.sidebar.write("")

input_df = input_feature_sidebar()
tab_single.dataframe(input_df)
bt_single_predict = tab_single.button("Predict!")

if bt_single_predict:
    y_pred = model.predict(input_df)
    if y_pred[0] == 1:
        tab_single.subheader("**:green[Subscribed]**")
        tab_single.markdown("""
        Berdasarkan data yang dimasukkan, model memprediksi bahwa nasabah ini **berpotensi berlangganan produk deposito**. 
        Hal ini menunjukkan adanya peluang konversi, sehingga tim pemasaran dapat menindaklanjuti dengan pendekatan yang tepat untuk memastikan keberhasilan.
        """)
    else:
        tab_single.subheader("**Not Subscribed**")
        tab_single.markdown("""
        Berdasarkan data yang dimasukkan, model memprediksi bahwa nasabah ini **kemungkinan tidak berlangganan produk deposito**. 
        Peluang konversi relatif rendah, sehingga upaya pemasaran dapat diprioritaskan pada segmen lain yang lebih potensial.
        """)


# csv_bytes = io.BytesIO(example_csv.encode('utf-8'))

# st.sidebar.download_button(
#     label="Download Example CSV",
#     data=csv_bytes,
#     file_name="example_data.csv",
#     mime="text/csv",
# )
# upload_csv = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
# st.sidebar.write("")
# if upload_csv is not None:
#     input_df = pd.read_csv(upload_csv)

# else:
#     st.sidebar.header("Input Feature:")
#     st.sidebar.write("")
#     input_df = input_feature_sidebar()

#_____________________Multi-Predict________________________________

example_csv = """age,job,marital,education,housing,loan,contact,month,day_of_week,campaign,previous,poutcome,cons.price.idx,cons.conf.idx,nr.employed,previous_contacted
35,technician,single,high.school,no,no,cellular,may,mon,2,0,success,93.5,-36.5,5020,1
47,services,married,basic.9y,yes,no,cellular,jun,tue,4,1,nonexistent,93.8,-35.2,5040,1
29,housemaid,single,basic.6y,no,no,cellular,jul,wed,1,0,success,93.2,-38.0,5005,1
52,retired,married,university.degree,yes,no,cellular,aug,thu,3,2,success,94.0,-34.5,5010,1
41,blue-collar,single,basic.9y,no,no,cellular,sep,fri,5,0,nonexistent,93.6,-36.8,5030,1
38,admin.,divorced,professional.course,yes,no,cellular,oct,mon,6,0,success,93.9,-35.0,5025,1
33,self-employed,single,basic.6y,no,no,cellular,nov,tue,2,0,nonexistent,93.4,-37.2,5000,1
49,services,married,high.school,yes,yes,cellular,dec,wed,7,1,success,94.1,-33.8,5045,1
36,technician,single,professional.course,no,no,cellular,may,thu,3,0,nonexistent,93.7,-36.0,5035,1
42,housemaid,divorced,basic.4y,yes,no,cellular,jun,fri,4,0,success,93.3,-35.5,5015,1
30,student,single,basic.4y,no,no,telephone,jul,mon,1,0,nonexistent,93.0,-37.5,5150,0
55,retired,married,university.degree,yes,yes,telephone,aug,tue,6,1,failure,94.2,-29.8,5180,0
37,blue-collar,single,basic.9y,yes,no,telephone,sep,wed,5,0,failure,93.6,-33.9,5125,0
44,admin.,divorced,professional.course,no,no,telephone,oct,thu,3,0,nonexistent,94.0,-31.5,5140,0
31,self-employed,single,basic.6y,yes,no,telephone,nov,fri,2,0,failure,93.2,-36.7,5160,0
48,services,married,high.school,no,yes,telephone,dec,mon,4,0,nonexistent,93.9,-34.2,5135,0
39,technician,single,professional.course,yes,no,telephone,may,tue,3,0,failure,94.1,-32.0,5175,0
46,housemaid,divorced,basic.4y,no,no,telephone,jun,wed,5,0,nonexistent,93.5,-36.0,5150,0
34,student,single,basic.4y,yes,no,telephone,jul,thu,1,0,failure,93.0,-38.5,5120,0
51,retired,married,university.degree,yes,yes,telephone,aug,fri,6,1,nonexistent,94.0,-30.5,5185,0
32,blue-collar,single,basic.9y,no,no,telephone,sep,mon,2,0,failure,93.6,-37.0,5145,0
40,admin.,divorced,professional.course,yes,no,telephone,oct,tue,4,0,nonexistent,94.2,-33.0,5165,0
36,self-employed,single,basic.6y,no,no,telephone,nov,wed,3,0,failure,93.7,-36.5,5130,0
45,services,married,high.school,yes,yes,telephone,dec,thu,5,1,nonexistent,94.1,-32.5,5155,0
38,technician,single,professional.course,no,no,telephone,may,fri,2,0,failure,93.4,-35.0,5170,0
49,housemaid,divorced,basic.4y,yes,no,telephone,jun,mon,4,0,nonexistent,93.8,-33.5,5125,0
27,student,single,basic.4y,no,no,cellular,jul,tue,1,0,success,93.1,-37.8,5010,1
54,retired,married,university.degree,yes,yes,cellular,aug,wed,6,2,success,94.2,-34.0,5020,1
35,blue-collar,single,basic.9y,no,no,cellular,sep,thu,3,0,nonexistent,93.5,-36.2,5035,1
43,admin.,divorced,professional.course,yes,no,cellular,oct,fri,5,0,success,94.0,-35.5,5025,1
31,self-employed,single,basic.6y,no,no,cellular,nov,mon,2,0,nonexistent,93.6,-37.0,5005,1
50,services,married,high.school,yes,yes,cellular,dec,tue,4,1,success,94.1,-34.5,5015,1
39,technician,single,professional.course,no,no,cellular,may,wed,3,0,nonexistent,93.7,-36.0,5020,1
41,housemaid,divorced,basic.4y,yes,no,cellular,jun,thu,5,0,success,93.8,-35.2,5005,1
28,student,single,basic.4y,no,no,cellular,jul,fri,1,0,nonexistent,93.2,-38.2,5010,1
53,retired,married,university.degree,yes,yes,cellular,aug,mon,6,1,success,94.0,-33.0,5025,1
37,blue-collar,single,basic.9y,no,no,cellular,sep,tue,3,0,nonexistent,93.5,-36.5,5030,1
44,admin.,divorced,professional.course,yes,no,cellular,oct,wed,5,0,success,94.2,-34.5,5005,1
33,self-employed,single,basic.6y,no,no,cellular,nov,thu,2,0,nonexistent,93.4,-37.5,5015,1
48,services,married,high.school,yes,yes,cellular,dec,fri,4,1,success,94.1,-35.0,5020,1
36,technician,single,professional.course,no,no,cellular,may,mon,3,0,nonexistent,93.6,-36.0,5000,1
"""

csv_bytes = io.BytesIO(example_csv.encode('utf-8'))

with tab_batch:
    csv_bytes = io.BytesIO(example_csv.encode('utf-8'))

    st.download_button(
        label="Download CSV Example",
        data=csv_bytes,
        file_name="example_input.csv",
        mime="text/csv"
    )

    csv_upload = st.file_uploader("Upload a CSV file :", type="csv")
    if csv_upload:

        upload_df = pd.read_csv(csv_upload)
        y_preds = model.predict(upload_df)
        y_preds_series = pd.Series(y_preds).map({1: "Subscribed ✅", 0: "Not Subscribed"})

        teks = "{}% Complete"
        bar = st.progress(0)
        for i in range(100):
            bar.progress(i +1 , text=teks.format(i+1))
            time.sleep(0.01)
        bar.empty()
        
        # Buat dua kolom
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Features")
            st.dataframe(upload_df)

        with col2:
            st.subheader("Prediction Result")
            st.dataframe(y_preds_series.rename("Prediction"))

