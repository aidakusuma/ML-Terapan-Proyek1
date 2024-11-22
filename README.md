# Laporan Proyek Machine Learning - Aida Kusuma Wardah

## Domain Proyek

Penyakit jantung merupakan kondisi di mana bagian-bagian jantung, termasuk pembuluh darah, selaput, katup, dan otot jantung, mengalami gangguan. Berbagai faktor dapat menyebabkan penyakit ini, seperti penyumbatan pada pembuluh darah jantung, peradangan, infeksi, atau kelainan yang sudah ada sejak lahir.
Jantung adalah organ penting yang terdiri dari otot dan terbagi menjadi empat ruang. Ruang-ruang ini dipisahkan oleh katup jantung yang berfungsi untuk mengatur aliran darah agar mengalir dalam satu arah.
Pada dinding jantung terdapat pembuluh darah yang dikenal sebagai arteri koroner, yang bertugas mengalirkan darah kaya oksigen ke seluruh bagian jantung. Pembuluh ini terbagi menjadi dua cabang, yaitu arteri koroner kanan dan kiri.
Jantung juga dilindungi oleh dua selaput yang disebut perikardium. Selaput ini berfungsi untuk melindungi jantung, menjaga posisinya tetap stabil, serta mencegah cedera akibat gesekan saat jantung berdenyut.
Penyakit jantung adalah masalah kesehatan yang serius, dan menggunakan machine learning untuk prediksi membantu dalam pengambilan keputusan klinis. Dengan memprediksi risiko penyakit jantung, dokter dapat membuat keputusan lebih cepat dan tepat dalam memberikan intervensi medis kepada pasien.

Sumber: https://www.alodokter.com/penyakit-jantung

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
Dataset: Dataset ini terdiri dari 1.319 entri (data), dan 9 fitur. ini menunjukkan bahwa dataset sangat cukup untuk melakukan analisis dan pelatihan model machine learning
Sumber dataset: https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset  

### Variabel-variabel pada dataset tersebut adalah sebagai berikut:
- age	=	Usia individu dalam satuan tahun.
- gender	=	Kode kategori gender (0 untuk perempuan, 1 untuk laki-laki).
- impluse	=	Denyut jantung individu dalam satuan bpm (beats per minute).
- pressurehight	=	Tekanan darah sistolik (mmHg), yaitu tekanan saat jantung memompa darah.
- pressurelow	=	Tekanan darah diastolik (mmHg), yaitu tekanan saat jantung beristirahat di antara denyutan.
- glucose	=	Kadar gula darah individu dalam satuan mg/dL.
- kcm	=	Level Creatine Kinase-MB dalam ng/mL, enzim yang menjadi indikator kerusakan otot jantung.
- troponin	=	Konsentrasi Troponin dalam ng/mL, protein yang meningkat saat terjadi kerusakan pada otot jantung.
- class	object	Kategori hasil diagnosis: Negative (kelas 0, tidak ada serangan jantung) dan Positive (kelas 1, ada serangan jantung).

![data distribusi kelas serangan jantung](image.png)
terlihat dalam grafik tersebut dataset tidak seimbang, data negatif berjumlah sekitar 400 dan data positif berjumlah sekitar 300, maka perlunya metode untuk menyeimbangkan data

![matriks korelasi](image-1.png)
1. Korelasi antara fitur-fitur:

a. Korelasi ditunjukkan oleh nilai dalam matriks (berkisaran antara -1 hingga 1). Warna merah gelap mewakili korelasi positif tinggi, sementara warna biru muda menunjukkan korelasi negatif atau mendekati nol.

b. Fitur pressureheight dan pressurelow memiliki korelasi tinggi (0.6), menunjukkan bahwa nilai kedua fitur ini cenderung berubah seiring.

c. Fitur troponin memiliki korelasi yang cukup tinggi terhadap class (0.53), menunjukkan hubungan yang cukup kuat.

2. Korelasi dengan target (class):

a. Fitur age (0.29) dan kcm (0.29) memiliki hubungan positif sedang dengan class.

b. Fitur lain, seperti impulse, pressureheight, pressurelow, glucose, memiliki korelasi rendah atau mendekati nol dengan class.

## Data Preparation
pada tahap ini dataset diperiksa apakah ada data yang hilang menggunakan fungsi isnull().sum()
lalu selanjutnya identifikasi outlier menggunakan boxplot pada fitur numerik tertentu
setelah itu dilakukan penghapusan outlier yang menggunakan metode interquartile range (IQR)
Setelah outlier dihapus, jumlah data yang tersisa dihitung, diikuti oleh persentase data yang dihapus untuk evaluasi dampak pembersihan. 
Statistik deskriptif sebelum dan sesudah penghapusan juga ditampilkan untuk membandingkan distribusi data. 
Sebagai tambahan, visualisasi distribusi fitur dilakukan untuk menggambarkan perubahan setelah penghapusan outlier, memberikan wawasan tentang keefektifan proses pembersihan.
lalu melakukan scaling pada fitur data menggunakan RobustScaler untuk mengurangi pengaruh outlier dengan menghitung median dan IQR, lalu mengonversinya kembali menjadi DataFrame dengan nama kolom yang sama.

## Modeling
Membangun beberapa model klasifikasi:
    - Random Forest: Algoritma ensemble yang kuat untuk mengatasi overfitting.
    parameter parameter yang digunakan dalam model ini yaitu:
    n_estimator = 100, Menentukan jumlah pohon keputusan yang digunakan dalam model.
    random_state = 42, Mengontrol konsistensi hasil dengan memastikan proses acak tetap sama.
    - SVM (Support Vector Machine): Algoritma yang efektif untuk data yang tidak linear.
    parameter parameter yang digunakan pada model ini:
    kernel = 'linear', Memilih fungsi pemisah data berupa garis lurus atau hyperplane.
    C=1, Mengatur keseimbangan antara akurasi pelatihan dan margin keputusan untuk meminimalkan overfitting.
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

terlihat bahwa algoritma model random forest memiliki akurasi yang lebih baik, bisa dilihat pada data diatas
hal ini berarti akan mampu mengidentifikasi penyakit jantung lebih akurat, sesuai goals yang akan diraih pada proyek ini
