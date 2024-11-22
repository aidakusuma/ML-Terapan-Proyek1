# Laporan Proyek Machine Learning - Aida Kusuma Wardah

## Domain Proyek

Penyakit jantung merupakan kondisi di mana bagian-bagian jantung, termasuk pembuluh darah, selaput, katup, dan otot jantung, mengalami gangguan. Berbagai faktor dapat menyebabkan penyakit ini, seperti penyumbatan pada pembuluh darah jantung, peradangan, infeksi, atau kelainan yang sudah ada sejak lahir. Jantung adalah organ penting yang terdiri dari otot dan terbagi menjadi empat ruang. Ruang-ruang ini dipisahkan oleh katup jantung yang berfungsi untuk mengatur aliran darah agar mengalir dalam satu arah.
Pada dinding jantung terdapat pembuluh darah yang dikenal sebagai arteri koroner, yang bertugas mengalirkan darah kaya oksigen ke seluruh bagian jantung. Pembuluh ini terbagi menjadi dua cabang, yaitu arteri koroner kanan dan kiri. Jantung juga dilindungi oleh dua selaput yang disebut perikardium. Selaput ini berfungsi untuk melindungi jantung, menjaga posisinya tetap stabil, serta mencegah cedera akibat gesekan saat jantung berdenyut. Penyakit jantung adalah masalah kesehatan yang serius, dan menggunakan machine learning untuk prediksi membantu dalam pengambilan keputusan klinis. Dengan memprediksi risiko penyakit jantung, dokter dapat membuat keputusan lebih cepat dan tepat dalam memberikan intervensi medis kepada pasien.

Berbagai studi tentang prediksi penyakit jantung dengan menggunakan machine learning antara lain
- Lestari, W., & Sumarlinda, S. (2023). Studi Komparatif Model Klasifikasi Kerentanan Penyakit Jantung Menggunakan Algoritma Machine Learning. SATIN - Sains Dan Teknologi Informasi, 9(1), 107-115. https://doi.org/10.33372/stn.v9i1.918
- Carolina Wibowo, A. ., Ardi Lestari, S. ., & Nurchim, N. (2024). ANALISIS PENGGUNAAN MACHINE LEARNING DALAM KLASIFIKASI PENENTUAN PENYAKIT JANTUNG. Simtek : Jurnal Sistem Informasi Dan Teknik Komputer, 9(2), 97–101. https://doi.org/10.51876/simtek.v9i2.395

### Mengapa masalah ini perlu diselesaikan?
- Tinggi Angka Kematian: Penyakit jantung merupakan salah satu penyebab utama kematian di dunia, menyebabkan jutaan kematian setiap tahunnya. Penanganan yang efektif dapat mengurangi angka kematian tersebut.
- Dampak Kesehatan dan Kualitas Hidup: Penyakit jantung dapat mengurangi kualitas hidup penderita, menghambat mobilitas, dan mempengaruhi kesehatan secara keseluruhan. Deteksi dini dan pengelolaan yang tepat dapat meningkatkan harapan hidup dan kualitas hidup pasien.
- Beban Ekonomi: Penyakit jantung menambah beban ekonomi melalui biaya perawatan medis yang tinggi dan kehilangan produktivitas. Mengatasi masalah ini dapat mengurangi pengeluaran sistem kesehatan dan memperbaiki stabilitas ekonomi.
### Bagaimana masalah ini dapat diselesaikan?
- Deteksi Dini Melalui Prediksi: Model machine learning dapat digunakan untuk menganalisis data medis, seperti riwayat kesehatan, gejala, dan faktor risiko lainnya, untuk memprediksi kemungkinan seseorang mengidap penyakit jantung. Dengan menggunakan algoritma seperti regresi logistik, decision trees, atau neural networks, model ini dapat memberikan prediksi dini yang membantu dokter dalam mengambil langkah pencegahan lebih cepat.
- Penyaringan Otomatis (Screening): Model machine learning dapat diterapkan untuk menyaring pasien yang berisiko tinggi mengidap penyakit jantung melalui analisis data besar, seperti hasil tes darah, elektrokardiogram (EKG), dan data medis lainnya. Algoritma seperti support vector machines (SVM) dan random forests dapat meningkatkan akurasi dan efisiensi dalam proses penyaringan, mengidentifikasi pasien yang membutuhkan perhatian lebih lanjut.

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

kondisi data yang akan digunakan sebagai berikut:


## Data Preparation
Pada tahap ini, dilakukan beberapa tahapan untuk mempersiapkan dataset sebelum digunakan dalam model machine learning.
- Handling Missing Values
  Dataset ini tidak mengandung missing values, yang telah diperiksa menggunakan fungsi isnull().sum(). Hasil pemeriksaan menunjukkan bahwa tidak ada nilai yang hilang pada dataset, sehingga tidak perlu dilakukan penanganan lebih lanjut untuk missing values.
- Outlier Detection and Removal
Outlier dideteksi menggunakan boxplot pada fitur numerik tertentu seperti age, impluse, pressurehight, pressurelow, glucose, kcm, dan troponin. Data yang terdeteksi sebagai outlier, berdasarkan metode Interquartile Range (IQR), dihapus menggunakan fungsi remove_outliers(). Fungsi ini menghitung batas bawah dan atas berdasarkan kuartil pertama (Q1) dan kuartil ketiga (Q3) untuk menentukan apakah data termasuk outlier.
Jumlah data sebelum dan setelah penghapusan outlier:
Sebelum penghapusan, jumlah data adalah 1319.
Setelah penghapusan, jumlah data menjadi 789.
Persentase data yang dihapus: Data yang dihapus akibat outlier mencapai 40.18%.
- Feature Engineering
Setelah penghapusan outlier, dilakukan scaling pada data menggunakan RobustScaler untuk mengurangi pengaruh outlier yang mungkin masih ada. Hasil dari proses scaling ini kemudian dikembalikan ke dalam bentuk DataFrame dengan nama kolom yang sama, untuk mempersiapkan data sebelum digunakan dalam model machine learning.
- Split Data
Pada tahap Split Data, dataset dibagi menjadi dua bagian utama: training data dan testing data. 80% untuk training data dan 20% untuk testing data.
- Balancing Data
dikearenakan adanya ketidakseimbangan data, maka dilakukan balancing data menggunakan teknik seperti SMOTE (Synthetic Minority Over-sampling Technique) atau undersampling pada kelas mayoritas. Hal ini untuk memastikan model machine learning tidak bias terhadap kelas yang lebih dominan.
  
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
