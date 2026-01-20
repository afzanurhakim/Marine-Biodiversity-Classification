import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

  
def main():
    # --- Judul dan Deskripsi ---
    current_dir = Path(__file__).parent
    st.title("EDA and Model Overview of Marine Biodiversity Classification")
    st.markdown(
        """
        Halaman ini menampilkan EDA dan overview model 
        klasifikasi Transfer Learning **InceptionV3**
        """
    )
    st.markdown("---")

    # --- Bagian 1: Eksplorasi Data (Menggunakan Data Aktual) ---
    st.header("Overview Dataset")
    
    st.markdown(
        """
        Dataset yang digunakan adalah kumpulan gambar biota laut dengan tujuan untuk 
        mengklasifikasikan 5 spesies utama: **Dolphin, Fish, Lobster, Octopus, dan Sea Horse**.
        """
    )

    # Data Distribusi Aktual dari Ringkasan Notebook
    data_dist = {
        'Dolphin': [198, 50, 20, 198+50+20],
        'Fish': [290, 50, 20, 290+50+20],
        'Lobster': [252, 50, 20, 252+50+20],
        'Octopus': [253, 50, 20, 253+50+20],
        'Sea Horse': [252, 50, 20, 252+50+20],
        'Total': [1245, 250, 100, 1595]
    }
    df_distribusi_raw = pd.DataFrame(data_dist, index=['Training Set', 'Validation Set', 'Test Set', 'Total'])
    
    st.subheader("Distribusi Kelas Dataset")
    st.dataframe(df_distribusi_raw)
    
    st.markdown("---")
    
    # --- Grafik Distribusi Data ---

    # 1. Grafik Training Set (Vertical)
    df_plot_train = df_distribusi_raw.drop(columns=['Total']).loc['Training Set'].reset_index()
    df_plot_train.columns = ['Kelas', 'Jumlah Data']

    st.subheader("Distribusi Kelas (Training Set)")
    fig_train, ax_train = plt.subplots(figsize=(10, 5)) 
    sns.barplot(x='Kelas', y='Jumlah Data', data=df_plot_train, ax=ax_train, palette='mako')
    ax_train.set_title('Distribusi Training Set', fontsize=14)
    ax_train.set_xlabel('Kelas')
    ax_train.set_ylabel('Jumlah Data')
    plt.xticks(rotation=15)
    st.pyplot(fig_train)

    st.markdown("---")
    
    # 2. Grafik Validation & Test Set (Side-by-side)
    st.subheader("Distribusi Kelas (Validation & Test Set)")
    
    # Ambil data untuk Validation dan Test
    df_plot_val = df_distribusi_raw.drop(columns=['Total']).loc['Validation Set'].reset_index()
    df_plot_val.columns = ['Kelas', 'Jumlah Data']

    df_plot_test = df_distribusi_raw.drop(columns=['Total']).loc['Test Set'].reset_index()
    df_plot_test.columns = ['Kelas', 'Jumlah Data']

    col_val, col_test = st.columns(2)

    with col_val:
        # Plot Validation Set
        fig_val, ax_val = plt.subplots(figsize=(7, 4))
        sns.barplot(x='Kelas', y='Jumlah Data', data=df_plot_val, ax=ax_val, palette='Pastel1')
        ax_val.set_title('Distribusi Validation Set', fontsize=14)
        ax_val.set_xlabel('Kelas')
        ax_val.set_ylabel('Jumlah Data')
        plt.xticks(rotation=15)
        st.pyplot(fig_val)
        
    with col_test:
        # Plot Test Set
        fig_test, ax_test = plt.subplots(figsize=(7, 4))
        sns.barplot(x='Kelas', y='Jumlah Data', data=df_plot_test, ax=ax_test, palette='Pastel2')
        ax_test.set_title('Distribusi Test Set', fontsize=14)
        ax_test.set_xlabel('Kelas')
        ax_test.set_ylabel('Jumlah Data')
        plt.xticks(rotation=15)
        st.pyplot(fig_test)
    
    st.markdown("---")

    # --- Ringkasan Properti Gambar Baru ---
    st.subheader("Ringkasan Properties Gambar Sample")
    
    st.markdown("**1. Ukuran dan Format Gambar Sample**")
    st.code("""
Kelas 'Dolphin': Ukuran Gambar: (612, 408) (Format: JPEG)
Kelas 'Fish': Ukuran Gambar: (500, 334) (Format: JPEG)
Kelas 'Lobster': Ukuran Gambar: (300, 225) (Format: JPEG)
Kelas 'Octopus': Ukuran Gambar: (300, 200) (Format: JPEG)
Kelas 'Sea Horse': Ukuran Gambar: (225, 300) (Format: JPEG)
    """)
    
    st.markdown("**2. Color Channel Gambar Sample**")
    st.code("""
Class 'Dolphin': Color Channel: RGB
Class 'Fish': Color Channel: RGB
Class 'Lobster': Color Channel: RGB
Class 'Octopus': Color Channel: RGB
Class 'Sea Horse': Color Channel: RGB
    """)

    st.markdown("---")

    # 2.1 Dolphin
    st.subheader("Kelas Dolphin (Lumba-lumba)")
    dolphin_path = current_dir / "5dolphin.png"
    st.image(str(dolphin_path), caption="Contoh Gambar Kelas Dolphin")
    st.markdown(
        """
        Kelas pertama adalah lumba-lumba yang memiliki ciri-ciri sebagai berikut:
        * Semua sampel diambil di bawah air dengan latar belakang biru cerah atau gelap menunjukkan laut dalam
        * Memiliki sirip, ekor, dan moncong yang cukup panjang dengan warna dominan abu-abu atau biru kehitaman
        * Lumba-lumba bisa ditemukan secara individu maupun berkelompok
        """
    )
    st.markdown("---")

    # 2.2 Fish
    st.subheader("Kelas Fish (Ikan)")
    fish_path = current_dir / "5fish.png"
    st.image(str(fish_path), caption="Contoh Gambar Kelas Fish")
    st.markdown(
        """
        Ikan merupakan kelas dengan variasi terbesar, meliputi ciri-ciri:
        * Berbagai bentuk, ukuran, dan pola warna, mencerminkan keragaman spesies.
        * Memiliki sisik yang jelas dan sirip yang menonjol (sirip dada, punggung, dan ekor).
        * Sampel bisa menunjukkan ikan individu atau sekumpulan ikan (schooling) di latar belakang terumbu karang atau air terbuka.
        """
    )
    st.markdown("---")
    
    # 2.3 Lobster
    st.subheader("Kelas Lobster")
    lobster_path = current_dir / "5lobster.png"
    st.image(str(lobster_path), caption="Contoh Gambar Kelas Lobster")
    st.markdown(
        """
        Ciri-ciri utama Lobster dalam dataset:
        * Memiliki cangkang keras (karapas) yang tebal dan sepasang capit besar yang digunakan untuk pertahanan dan makan.
        * Warna dominan adalah cokelat kemerahan, oranye, atau hijau tua, yang membantunya bersembunyi di dasar laut.
        * Sering difoto di habitat dasarnya, seperti bersembunyi di celah-celah batu atau berjalan di dasar laut berpasir.
        """
    )
    st.markdown("---")

    # 2.4 Octopus
    st.subheader("Kelas Octopus (Gurita)")
    octopus_path = current_dir / "5octopus.png"
    st.image(str(octopus_path), caption="Contoh Gambar Kelas Octopus")
    st.markdown(
        """
        Gurita dikenal karena kemampuannya beradaptasi dengan lingkungan:
        * Tubuh lunak dan tidak bertulang, dengan kepala bulat dan delapan lengan (tentakel) yang dilengkapi pengisap.
        * Sampel sering menunjukkan kemampuan kamuflase; gurita dapat mengubah warna kulitnya agar sesuai dengan latar belakang di sekitarnya (batu, pasir, karang).
        * Umumnya terlihat berdiam diri atau berenang perlahan di dekat dasar laut.
        """
    )
    st.markdown("---")

    # 2.5 Sea Horse
    st.subheader("Kelas Sea Horse (Kuda Laut)")
    seahorse_path = current_dir / "5seahorse.png"
    st.image(str(seahorse_path), caption="Contoh Gambar Kelas Sea Horse")
    st.markdown(
        """
        Ciri-ciri khas Kuda Laut:
        * Bentuk tubuh yang unik, tegak seperti kuda, dengan moncong panjang dan ekor yang dapat digunakan untuk menggenggam.
        * Ukuran relatif kecil dan sering ditemukan menempel pada rumput laut, terumbu karang, atau objek lain di dalam air.
        * Pergerakan lambat dan seringkali gambar diambil secara close-up.
        """
    )
    st.markdown("---")

    # --- Bagian 2: Ringkasan Hasil Klasifikasi ---
    st.header("Ringkasan Hasil Model (InceptionV3)")
    
    st.write(
        """
        Model Klasifikasi InceptionV3 yang dilatih telah menghasilkan kinerja sebagai berikut:
        """
    )
    
    # Metrik Kinerja Utama
    col_acc, col_f1, col_rec = st.columns(3)
    
    with col_acc:
        st.metric(label="Akurasi Model", value="87%", delta="Target Tercapai (Need Improvement)") 
        
    with col_f1:
        st.metric(label="F1-Score (Macro Average)", value="0.87", delta="Kinerja Cukup Baik") 
    
    with col_rec:
        st.markdown("<h3 style='text-align: center; color: #5B5B5B;'>Kelas Paling Kuat</h3>", unsafe_allow_html=True)
        st.info("Dolphin (F1: 0.97)") 

    st.subheader("Laporan Klasifikasi (Classification Report)")
    
    # Simulasi Classification Report
    data_report = {
        'Kelas': ['Dolphin', 'Fish', 'Lobster', 'Octopus', 'Sea Horse', 'Macro Avg'],
        'Precision': [1.00, 0.87, 1.00, 0.80, 0.76, 0.89],
        'Recall':    [0.95, 0.65, 0.80, 1.00, 0.95, 0.87],
        'F1-Score':  [0.97, 0.74, 0.89, 0.89, 0.84, 0.87]
    }
    df_report = pd.DataFrame(data_report)
    df_report = df_report.set_index('Kelas')

    st.dataframe(df_report)
    
    st.markdown(
        """
        * **Rekomendasi Utama**: Berdasarkan data aktual, kelas **Fish** memiliki jumlah data train terbanyak (290), 
          namun memiliki F1-Score terendah, mengindikasikan bahwa variasi gambar pada kelas ini mungkin kurang representatif sehingga 
          perlu penambahan variasi data untuk meningkatkan performa model
        """
    )

# Jalankan aplikasi Streamlit
if __name__ == '__main__':
    main()