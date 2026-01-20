## Problem Statement

Proses pemantauan dan pencatatan keanekaragaman hayati laut (marine biodiversity) masih menghadapi tantangan besar karena volume data gambar atau video bawah air yang sangat besar dan kondisi lingkungan yang kompleks.

Proses identifikasi dan klasifikasi spesies biota laut (seperti membedakan biota seperti lumba-lumba, ikan, lobster, gurita, dan kuda laut) saat ini masih sering bergantung pada:

- Analisis manual oleh ahli biologi laut dari rekaman footage bawah air.

- Metode survei tradisional yang memakan waktu dan mahal.

- Proses pengolahan data yang lambat yang bisa saja menghambat respons cepat terhadap perubahan ekologis.

Kesalahan atau keterlambatan dalam klasifikasi jenis biota laut dapat menyebabkan:

- Ketidakakuratan dalam estimasi populasi dan tren ekologis.

- Pengambilan keputusan konservasi yang kurang tepat atau terlambat.

- Pemborosan waktu dan sumber daya peneliti dalam menyaring data mentah.

Oleh karena itu, diperlukan sebuah pendekatan otomatis berbasis Computer Vision yang mampu mengklasifikasikan lima jenis biota laut tersebut secara akurat menggunakan gambar yang diambil dari kamera bawah air baik dari Remotely Operated Vehicle (ROV) ataupun stasiun pemantauan

## Project Objective

Project ini dibuat untuk mengembangkan model Computer Vision (CNN) yang mampu mengklasifikan lima jenis biota laut (lumba-lumba, ikan, lobster, gurita, dan kuda laut) secara akurat menggunakan gambar dari kamera bawah air. Sehingga project ini dapat meningkatkan keandalan marine biodiversity monitoring, memberikan insight ekologis yang cepat, serta mendukung upaya konservasi laut global.

## Dataset Information

Dataset ini didapatkan dari kaggle (https://www.kaggle.com/datasets/ananya12verma/marine-image-dataset-for-classification)

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
| 10 | **InceptionV3** | Transfer learning backbone untuk ekstraksi fitur |
| 11 | **PIL** | Load untuk manipulasi gambar dasar |
| 12 | **Scikit-learn** | Evaluasi model (classification report, confusion matrix, display) |
| 13 | **Streamlit** | Pembuatan antarmuka aplikasi (Deployment) |

**Tools Pendukung:**
* **VSCode**: Digunakan sebagai *Integrated Development Environment* (IDE) utama untuk penulisan, pengujian, dan pengelolaan kode program.
* **Hugging Face**: Digunakan sebagai platform untuk penyimpanan, publikasi, dan *deployment* model machine learning.


## Project Output 
[Marine Biodiversity Classification App](https://huggingface.co/spaces/afzanurhakim/Marine-Biodiversity-Classification)

