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

##kondisi data yang akan digunakan sebagai berikut:
### Data Columns (Total: 9 columns)

| #   | Column        | Non-Null Count | Dtype   |
|-----|---------------|----------------|---------|
| 0   | age           | 1319 non-null  | int64   |
| 1   | gender        | 1319 non-null  | int64   |
| 2   | impluse       | 1319 non-null  | int64   |
| 3   | pressurehight | 1319 non-null  | int64   |
| 4   | pressurelow   | 1319 non-null  | int64   |
| 5   | glucose       | 1319 non-null  | float64 |
| 6   | kcm           | 1319 non-null  | float64 |
| 7   | troponin      | 1319 non-null  | float64 |
| 8   | class         | 1319 non-null  | object  |

### Data Summary

dilihat pada data dibawah ini tidak terdapat missing values sehingga data siap digunakan
| #   | Column        | Count |
|-----|---------------|-------|
| 0   | age           |   0   |
| 1   | gender        |   0   | 
| 2   | impluse       |   0   |
| 3   | pressurehight |   0   |
| 4   | pressurelow   |   0   |
| 5   | glucose       |   0   |
| 6   | kcm           |   0   |
| 7   | troponin      |   0   |
| 8   | class         |   0   |

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
- Random Forest:

Random Forest adalah algoritma ensemble yang menggunakan banyak pohon keputusan (decision trees) untuk membuat keputusan yang lebih akurat dan stabil. Algoritma ini bekerja dengan membangun beberapa pohon keputusan secara acak dan menggabungkan hasilnya untuk mendapatkan keputusan akhir. Dalam kasus klasifikasi, keputusan akhir dibuat dengan cara voting mayoritas (class dengan suara terbanyak). Model ini memiliki beberapa keuntungan seperti menghindari overfitting dan memberikan hasil yang lebih stabil.

parameter parameter yang digunakan dalam model ini yaitu:
1. n_estimator = 100, Menunjukkan jumlah pohon keputusan yang digunakan dalam hutan acak. Semakin banyak pohon, semakin baik hasil yang dihasilkan (pada titik tertentu)
2. random_state = 42, Menjamin bahwa pembagian data dan pembuatan model dapat direplikasi, sehingga hasilnya konsisten setiap kali dijalankan.

- SVM (Support Vector Machine):

Support Vector Machine (SVM) adalah algoritma klasifikasi yang berfokus pada pencarian hyperplane terbaik yang memisahkan kelas-kelas dalam dataset. SVM mencoba untuk menemukan batas yang memaksimalkan margin antara kelas yang berbeda. SVM sangat efektif untuk dataset dengan dimensi tinggi dan memiliki kemampuan untuk menangani data non-linear jika kernel yang tepat digunakan.

parameter parameter yang digunakan pada model ini:
1. kernel = 'linear',  Kernel linear digunakan untuk mencari hyperplane yang memisahkan dua kelas dalam ruang fitur. Kernel ini cocok jika data relatif linier dan tidak memerlukan pemetaan ke ruang dimensi yang lebih tinggi.

2. C=1, Parameter ini mengontrol margin kesalahan (penalti untuk kesalahan klasifikasi). Nilai C yang lebih tinggi dapat membuat model lebih ketat (lebih sedikit margin kesalahan) tetapi juga berisiko overfitting.

## Evaluation
Metrik Evaluasi yang digunakan yaitu:
1. Accuracy

Accuracy adalah persentase prediksi yang benar dari keseluruhan prediksi yang dibuat oleh model. Metrik ini memberikan gambaran umum seberapa sering model memprediksi dengan benar.
Rumus: Accuracy = (True Positives + True Negatives) / Total Predictions

2. Precision

Precision adalah proporsi dari prediksi positif yang benar, atau seberapa akurat model saat memprediksi kelas positif. Dalam konteks penyakit jantung, ini berarti berapa banyak dari prediksi bahwa seseorang menderita penyakit jantung (kelas 1) yang benar-benar menderita penyakit tersebut.
Rumus: Precision = True Positives / (True Positives + False Positives)

3. Recall

Recall adalah kemampuan model untuk mendeteksi semua kelas positif yang sebenarnya. Dalam kasus penyakit jantung, recall mengukur seberapa baik model dalam mendeteksi pasien yang benar-benar mengidap penyakit jantung.
Rumus: Recall = True Positives / (True Positives + False Negatives)

4. F1-Score

F1-Score adalah rata-rata harmonis antara precision dan recall. Metrik ini digunakan untuk memberikan gambaran lebih seimbang antara precision dan recall, terutama ketika ada ketidakseimbangan antara kedua metrik tersebut.
Rumus: F1-Score = 2 × (Precision × Recall) / (Precision + Recall)


Hasil Evaluasi: 
1. Random Forest
   - Train Set Accuracy: 1.0000
   - Test Set Accuracy: 0.9747
   - Random Forest Classification Report:

                       precision    recall  f1-score   support
                    0       0.97      0.99      0.98       102
                    1       0.98      0.95      0.96        56
             accuracy                           0.97       158
            macro avg       0.98      0.97      0.97       158
         weighted avg       0.97      0.97      0.97       158
2. SVM
   - Train Set Accuracy: 0.9225
   - Test Set Accuracy: 0.9304
   - SVM Classification Report:

                       precision    recall  f1-score   support
                    0       0.95      0.94      0.95       102
                    1       0.89      0.91      0.90        56
             accuracy                           0.93       158
            macro avg       0.92      0.93      0.92       158
         weighted avg       0.93      0.93      0.93       158

kesimpulan: 
1. Model yang dikembangkan dengan menggunakan algoritma Random Forest dan SVM berhasil memprediksi kemungkinan penyakit jantung berdasarkan data medis yang telah dipersiapkan. Hasil evaluasi menunjukkan bahwa model Random Forest sangat akurat, dengan akurasi mencapai 97% pada data uji. Ini menunjukkan bahwa model telah berhasil menjawab problem statement dengan memberikan prediksi yang akurat terkait penyakit jantung.
2. Berdasarkan hasil evaluasi, model Random Forest mencapai akurasi 97% pada data uji, yang lebih tinggi dibandingkan dengan model SVM yang hanya mencapai 71% pada data uji. Selain itu, precision, recall, dan F1-score pada Random Forest juga menunjukkan kinerja yang sangat baik, mengindikasikan bahwa model ini dapat mengidentifikasi pasien dengan risiko penyakit jantung secara akurat. Hal ini menunjukkan bahwa model tersebut berhasil mencapai tujuan yang diharapkan, yaitu meningkatkan akurasi prediksi untuk pengambilan keputusan klinis.
3. Penggunaan Random Forest sebagai solusi utama terbukti memberikan dampak signifikan terhadap akurasi dan efektivitas prediksi penyakit jantung. Dengan menggunakan teknik SMOTE untuk menangani ketidakseimbangan data, model ini tidak hanya dapat mengidentifikasi pasien dengan risiko tinggi, tetapi juga memberikan hasil yang lebih stabil dan dapat diandalkan dibandingkan dengan model lain seperti SVM. Ini menunjukkan bahwa solusi yang diterapkan berdampak positif dalam meningkatkan performa model dan mendukung pengambilan keputusan medis yang lebih cepat dan akurat.
