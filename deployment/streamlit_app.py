import streamlit as st
st.set_page_config(
    page_title="Marine Biodiversity Classification",
    layout="wide"
)
import eda
import prediction

page = st.sidebar.selectbox('Pilih Halaman: ', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.main()
else:
    prediction.main()