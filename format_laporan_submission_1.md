# Laporan Proyek Machine Learning - Aida Kusuma Wardah

## Domain Proyek

Penyakit jantung merupakan salah satu penyebab utama kematian di seluruh dunia. Deteksi dini penyakit ini sangat penting untuk mencegah komplikasi serius yang dapat berujung pada kematian. Dengan bantuan machine learning, kita dapat membangun model prediktif yang dapat memprediksi apakah seseorang berisiko terkena penyakit jantung berdasarkan parameter medis seperti tekanan darah, kadar kolesterol, dan lainnya.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Penyakit jantung adalah masalah kesehatan yang serius, dan menggunakan machine learning untuk prediksi membantu dalam pengambilan keputusan klinis. Dengan memprediksi risiko penyakit jantung, dokter dapat membuat keputusan lebih cepat dan tepat dalam memberikan intervensi medis kepada pasien.
  
  Format Referensi: [Judul Referensi](https://scholar.google.com/) 

## Business Understanding
### Problem Statements
Menjelaskan pernyataan masalah latar belakang:
- Bagaimana kita dapat memprediksi kemungkinan seseorang menderita penyakit jantung yang akurat menggunakan data medis?
- Algoritma dan metode machine learning apa yang efektif untuk memprediksi penyakit jantung?

### Goals
Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan model prediksi yang mampu mengidentifikasi pasien yang berisiko terkena penyakit jantung.
- meraih tingkat akurasi yang baik agar bisa memprediksi lebih bagus untuk pengambilan keputusan klinis
    ### Solution statements
    - Menggunakan beberapa algoritma klasifikasi (Random Forest, SVM).
    - improvement dataset menggunakan metode oversampling SMOTE
    - Menggunakan metrik evaluasi seperti accuracy, precision, recall, dan F1-score.

## Data Understanding
Dataset: Dataset ini terdiri dari 1.319 entri (data), dan 9 fitur
Sumber dataset: https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset  

### Variabel-variabel pada dataset tersebut adalah sebagai berikut:
- age	int64	Usia individu dalam satuan tahun.
- gender	int64	Kode kategori gender (0 untuk perempuan, 1 untuk laki-laki).
- impluse	int64	Denyut jantung individu dalam satuan bpm (beats per minute).
- pressurehight	int64	Tekanan darah sistolik (mmHg), yaitu tekanan saat jantung memompa darah.
- pressurelow	int64	Tekanan darah diastolik (mmHg), yaitu tekanan saat jantung beristirahat di antara denyutan.
- glucose	float64	Kadar gula darah individu dalam satuan mg/dL.
- kcm	float64	Level Creatine Kinase-MB dalam ng/mL, enzim yang menjadi indikator kerusakan otot jantung.
- troponin	float64	Konsentrasi Troponin dalam ng/mL, protein yang meningkat saat terjadi kerusakan pada otot jantung.
- class	object	Kategori hasil diagnosis: Negative (kelas 0, tidak ada serangan jantung) dan Positive (kelas 1, ada serangan jantung).



## Data Preparation
Data Cleaning: Menghapus atau mengganti nilai yang hilang. Menangani outlier jika ada.
Encoding: Mengubah kolom target ke dalam format numerik menggunakan teknik one-hot encoding atau label encoding.
Normalization: Menerapkan RobustScaler untuk menormalkan data numerik dengan mengurangi median dan membaginya dengan IQR
Data Split: Memisahkan data menjadi set latih dan uji dengan perbandingan 80% untuk latih dan 20% untuk uji.
Improvement Data menggunakan SMOTE: untuk peningkatan akurasi model dan menyeimbangkan data
## Modeling
Membangun beberapa model klasifikasi:
    - Random Forest: Algoritma ensemble yang kuat untuk mengatasi overfitting.
    - SVM (Support Vector Machine): Algoritma yang efektif untuk data yang tidak linear.

## Evaluation
Metrik Evaluasi:
    - Accuracy: Persentase prediksi yang benar dari keseluruhan prediksi.
    - Precision: Proporsi prediksi positif yang benar.
    - Recall: Kemampuan model mendeteksi kelas positif.
    - F1-score: Harmonic mean dari precision dan recall, ideal untuk data yang tidak seimbang.
Hasil Evaluasi: 
Random Forest - Train Set Accuracy: 1.0000
Random Forest - Test Set Accuracy: 0.9684
Random Forest Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.98      0.98       102
           1       0.96      0.95      0.95        56

    accuracy                           0.97       158
   macro avg       0.97      0.96      0.97       158
weighted avg       0.97      0.97      0.97       158

SVM - Train Set Accuracy: 0.7310
SVM - Test Set Accuracy: 0.7152

SVM Classification Report:
               precision    recall  f1-score   support

           0       0.81      0.74      0.77       102
           1       0.58      0.68      0.63        56

    accuracy                           0.72       158
   macro avg       0.70      0.71      0.70       158
weighted avg       0.73      0.72      0.72       158

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

