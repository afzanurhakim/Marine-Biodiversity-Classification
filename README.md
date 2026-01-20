<div align="center">
  <img src="trueorigin.jpeg" alt="TrueOrigin Logo" width="200" style="margin-bottom: 0px;">
  <h1 style="margin-top: 0px;">TrueOrigin</h1>
  <p><i>AI vs Real Artwork Detection</i></p>
</div>

---

## Program Description

**True Origin** dikembangkan sebagai solusi teknologi bagi organisasi/perusahaan untuk melakukan filtrasi awal terhadap karya seni, untuk mendeteksi apakah karya tersebut merupakan hasil buatan AI atau karya manusia asli. Tool ini sangat berguna pada kompetisi seni untuk menyaring karya yang masuk sebelum tahap kurasi lebih lanjut oleh juri. Dengan menggunakan teknologi Computer Vision berbasis Convolutional Neural Network (CNN) dengan arsitektur MobileNetV2 yang telah dioptimasi, TrueOrigin mampu menjalankan tugas spesifik ini secara akurat. Tujuan utamanya adalah membantu meminimalkan perselisihan terkait karya buatan AI yang memenangkan kompetisi seni non-AI

## Problem Background

Meningkatnya penggunaan teknologi AI generatif dalam pembuatan karya visual menimbulkan tantangan serius dalam proses identifikasi keaslian karya, terutama pada skala besar seperti lomba atau seleksi terbuka. Karya non-AI memiliki keragaman visual yang sangat tinggi, mencakup berbagai gaya seni, medium, dan teknik, sehingga membentuk kelas yang tidak homogen dan sulit didefinisikan secara konsisten. Kondisi ini semakin diperumit oleh munculnya karya digital modern yang secara visual semakin menyerupai hasil AI-generated, menyebabkan tumpang tindih karakteristik visual antara karya AI dan non-AI. 

Proses identifikasi manual maupun otomatis berisiko mengalami kesalahan, khususnya dalam bentuk lolosnya karya AI sebagai karya non-AI (false negative), yang dapat mengancam integritas seleksi. Di sisi lain, keterbatasan waktu dan sumber daya membuat proses pemeriksaan mendalam terhadap seluruh karya menjadi tidak efisien. Oleh karena itu, dibutuhkan sebuah mekanisme penyaringan awal yang mampu memprioritaskan deteksi karya berpotensi AI-generated secara waspada, sehingga proses verifikasi lanjutan dapat difokuskan secara lebih efektif tanpa mengorbankan keadilan dan akurasi keputusan akhir.

## Project Objective

Berdasarkan permasalahan identifikasi keaslian karya visual dalam konteks penggunaan AI generatif, tujuan yang ingin dicapai dalam project ini antara lain:

* **Penyaringan Awal Karya Visual:** Menyediakan mekanisme penyaringan awal untuk mengidentifikasi karya visual yang berpotensi AI-generated dalam jumlah besar secara cepat dan efisien.
* **Minimisasi Risiko False Negative:** Mengutamakan pendeteksian karya AI-generated guna meminimalkan risiko lolosnya karya AI yang diklasifikasikan sebagai non-AI, terutama dalam konteks seleksi atau lomba seni.
* **Dukungan Proses Verifikasi Bertahap:** Membantu proses seleksi dengan mengarahkan karya yang terindikasi AI-generated ke tahap pemeriksaan lanjutan, sehingga verifikasi mendalam dapat dilakukan secara lebih terfokus dan efisien.
* **Efisiensi Waktu dan Sumber Daya:** Mengurangi beban kerja manual reviewer dengan menyediakan alat bantu awal yang mampu menangani volume data besar tanpa mengorbankan integritas dan keadilan proses seleksi.

## Dataset Information

Dataset ini didapatkan dari kaggle (https://www.kaggle.com/datasets/kausthubkannan/ai-and-human-art-classification/discussion/479343) yang gambarnya didapat dari existing dataset HuggingFace dan tambahan dari Pinterest.

## Tech Stack

Proyek ini dibangun menggunakan **Python** sebagai bahasa utama, dengan dukungan ekosistem library berikut:

| No | Library | Fungsi |
| :--- | :--- | :--- |
| 1 | **OS** | Akses direktori, path dataset, dan manajemen file |
| 2 | **Random** | PShuffle / sampling data secara acak |
| 3 | **Warnings** | Menonaktifkan warning agar output notebook lebih bersih |
| 4 | **Pandas** | Pengolahan data tabular/metadata (jika diperlukan untuk analisis) |
| 5 | **NumPy** | Operasi numerik dan manipulasi array |
| 6 | **Matplotlib** | Visualisasi (grafik training, evaluasi, dsb.) |
| 7 | **TensorFlow** | Framework utama deep learning |
| 8 | **Keras** | Training & inferensi model (Sequential/Model, layers, callbacks, optimizer) |
| 9 | **ImageDataGenerator** | Pipeline input gambar + augmentasi & preprocessing |
| 10 | **MobileNetV2** | Transfer learning backbone untuk ekstraksi fitur |
| 11 | **PIL** | Load untuk manipulasi gambar dasar |
| 12 | **Scikit-learn** | Evaluasi model (classification report, confusion matrix, display) |
| 13 | **Streamlit** | Pembuatan antarmuka aplikasi (Deployment) |

**Tools Pendukung:**
* **VSCode**: Digunakan sebagai *Integrated Development Environment* (IDE) utama untuk penulisan, pengujian, dan pengelolaan kode program.
* **Hugging Face**: Digunakan sebagai platform untuk penyimpanan, publikasi, dan *deployment* model machine learning.


## Data Pipeline & Methodology

1. **Data Acquisition**: Menggunakan dataset gambar karya visual AI-generated dan non-AI dari sumber publik (Kaggle).

2. **Data Preparation & Validation**:  
   * Pengorganisasian struktur direktori dataset berdasarkan kelas (AI vs non-AI).  
   * Penyesuaian ukuran dan format gambar agar konsisten dengan kebutuhan model.  
   * Pembagian data ke dalam set training, validation, dan testing.  
   * Penerapan preprocessing dan augmentasi gambar untuk meningkatkan generalisasi model.

3. **Exploratory Data Analysis (EDA)**: Analisis distribusi jumlah gambar pada masing-masing kelas serta visualisasi sampel untuk memahami karakteristik visual AI-generated dan non-AI.

4. **Modeling**:  
   * Pemanfaatan pendekatan transfer learning menggunakan **MobileNetV2** sebagai feature extractor.  
   * Penyesuaian layer klasifikasi untuk tugas klasifikasi biner (AI vs non-AI).

5. **Evaluation & Inference**: Evaluasi performa model menggunakan confusion matrix dan classification report, serta pemanfaatan model sebagai mekanisme penyaringan awal (early filtering) karya visual.


## Project Output 
[TrueOrigin App](https://huggingface.co/spaces/goetzel11/TrueOrigin)

## Project Structure

```text
├── deployment/               
├── EDA.ipynb   
├── Model_Inference.ipynb         
├── Model_Training.ipynb            
└── README.md                 
```

## Team Members

<table align="center">
  <tr>
    <td align="center" width="220">
      <img src="https://github.com/UngiiiKiat.png" width="110" height="110" style="border-radius: 50%;" />
      <h3>Rangga Irwanto Putra Kiat</h3>     
      <p><b>Role:</b> Data Analyst</p>
      <a href="https://github.com/UngiiiKiat">GitHub</a> • 
      <a href="https://www.linkedin.com/in/ranggakiat/">LinkedIn</a>
    </td>

   <td align="center" width="220">
      <img src="https://github.com/afzanurhakim.png" width="110" height="110" style="border-radius: 50%;" />
      <h3>Muhammad Afza Nur Hakim</h3>
      <p><b>Role:</b> Data Scientist</p>
      <a href="https://github.com/afzanurhakim">GitHub</a> • 
      <a href="https://www.linkedin.com/in/afzanurhakim/">LinkedIn</a>
    </td>

   <td align="center" width="220">
      <img src="https://github.com/Ghulaaaam.png" width="110" height="110" style="border-radius: 50%;" />
      <h3>Achmad Ghulam Habib Al-widani</h3>
      <p><b>Role:</b> Data Scientist</p>
      <a href="https://github.com/Ghulaaaam">GitHub</a> • 
      <a href="https://www.linkedin.com/in/achmad-ghulam-479b56255/">LinkedIn</a>
    </td>

   <td align="center" width="220">
      <img src="https://github.com/goetzel11.png" width="110" height="110" style="border-radius: 50%;" />
      <h3>Muhammad Aldzahabi Mawarid</h3>
      <p><b>Role:</b> Data Engineer</p>
      <a href="https://github.com/goetzel11">GitHub</a> • 
      <a href="https://www.linkedin.com/in/Aldzahabi/">LinkedIn</a>
    </td>
  </tr>
</table>

<div align="center"> <p>Made with ❤️ by TrueOrigin Team</p> </div>
