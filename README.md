# UTS_Pembelajaran-Mesin_1217050020_ArifRahmanS
Nama : Arif Rahman Sopian 
NIM:1217050020 
Dosen Pengampu: H.,Aldy Rialdy Atmadja,M.T
=============================================================================================================================================================================
1. Pemahaman Masalah
Saya akan mencoba membangun model klasifikasi yang mampu membedakan jenis buah sitrus berdasarkan karakteristik fisiknya

Tujuan utama dari klasifikasi adalah untuk mengidentifikasi apakah buah termasuk dalam kelas grapefruit atau orange berdasarkan fiturfitur yang tersedia. Mengingat kedua jenis buah ini seringkali memilikii kemiripan secara bentuk atau visual, terutama pada tipe tertentu.

Variabel input fitur pada dataset ini meliputi nama buah, yang akan saya jadikan sebagai pembeda awal, serta berbagai parameter fisik seperti diameter buah dalam cm, weight berat buah dalam gram, intensitas warna merah, intensitas warna hijau, dan intensitas warna biru. Sementara itu, variabel target yang akan diprediksi adalah kolom class, yang berisi dua kategori utama yaitu ‘grapefruit’ dan ‘orange’

2. Pengumpulan dan Load Data
Langkah pertama yang saya lakukan dalam eksperimen ini adalah mengumpulkan data yang akan digunakan. Dataset yang saya gunakan bernama `citrus.csv`.

Untuk memuat data ke dalam praktik menggunakan Googlw Colab, saya menguunakan library `pandas`. Berikut adalah potongan kode yang saya jalankan:

python
import pandas as pd
df=pd.read_csv('citrus.csv')
print(df.head())

Setelah berhasil memuat data, saya segera melihat isi dataset. Terlihat bahwa dataset ini memiliki 7 kolom. Kolom name berisi pengenal unik untk setiap sampel, `diameter`, `weight`, `red`, `green`, `blue` adalah fitur numerik, dan kolom `class` adalah target yang berisi string 'grapefruit' atau 'orange'. tidak ada yang aneh pada data atau rusak terlihat baik baik saja. lalu saya yakin untuk melanjutkan ke tahap eksplorasi.

3. Eksplorasi Data
saya lakukan dengan teliti karena hasil analisis akan sangat mempengaruhi keputusan preprocessing nantinya.

- Struktur Data
Saya memeriksa struktur data menggunakan `df.info()`. Hasilnya menunjukkan bahwa dataset memiliki 1000 baris dan 7 kolom. Semua kolom fitur numerik (`diameter`, `weight`, ` red`, `green`, `blue`) bertipe data `float64`, sementara kolom `name` dan `class` adalah `object` . 

Selanjutnya, saya meringkasan statistik menggunakan `df.describe()` dari output, saya bisa melihat bahwa diameter buah berkisar antara 4.78 cm hingga 8.96 cm dengan rata-rata sekitar 6.85 cm. Berat buah dari 74.51 gram sampai 206.87 gram, dengan ratarata 140.34 gram. Menariknya, nilai rata-ata untuk warna merah (red) sekitar 164, hijau (green) sekitar 93, dan biru (blue) sekitar 71. Ini mengindikasikan bahwa secara umum, buah dalam dataset ini cenderung memiliki warna kemerahan atau oranye (karena nilai merah tinggi), yang masuk akal karena kita berhadapan dengan buah sitrus matang.

Saya menghitung jumlah sampel per kelas. Hasilnya distribusi kelas terbilang cukup seimbang. Kelas grapefruit berjumlah 500 sampel, dan kelas ‘orange’ juga 500 sampel Dengan distribusi 50:50, model bukanlah bias terhadap satu kelas tertentu, sehingga evaluasi akurasi lebih akurat Saya memeriksa apakah ada nilai yang hilang menggunakan `df.isnull().sum()`. semua kolom yaitu bernilai 0 untuk missing values...

Selanjutnya, saya membuat heatmap korelasi antar fitur. Saya terkejut melihat bahwa korelasi antara `diameter` dan `weight` sangat tinggi, mencapai di atas 0.9. Artinya, semakin besar diameter buah, hampir pasti semakin berat buah tersebut. Ini masuk akal secara fisika. Saya juga melihat korelasi antara fitur warna, misalnya `red` dan `green` memiliki korelasi negatif yang cukup kuat (sekitar -0.7). Semakin merah suatu buah, maka semakin kurang hijaunya.

5. Split Data
data dibagi menjadi dua bagian: data latih (training) dan data uji (testing) yaitu membagi dengan proporsi 80% untuk training dan 20% untuk testing. Mengingat total data adalah 1000 baris, maka 800 baris digunakan  melatih model, dan 200 baris akan digunakan untuk menguji performa model.

6. Decision Tree
Model pertama adalah Decision Tree. Decision Tree akan membagi ruang fitur (misalnya “apakah diameter > 7?”) menjadi area-area kecil. Kelebihannya adalah model ini sangat mudah diterapkan karena kita bisa melihat jalur keputusannya. Saya melatih model ini dengan menggunakan `DecisionTreeClassifier`.

7. Naive Bayes
Model kedua adalah Naive Bayes dengan asumsi distribusi Gaussian saya memilih `GaussianNB` Konsep di balik Naive Bayes adalah Teorema Bayes.
Support Vector Machine (SVM)
Model ketiga adalah SVM. SVM bekerja dengan mencari *hyperplane*. saya menggunakan kernel linear (default) untuk menghindari komputasi yang terlalu kompleks.

Saya melatih ketiga model dengan memanggil fungsi .fit(X_train, y_train) pada masing-masing objek classifier.

8. Prediksi
langkah berikutnya adalah melakukan prediksi menggunakan data testing (`X_test`) dengan memanggil fungsi `.predict(X_test)` untuk setiap model. Hasil prediksi ini berupa array berisi 200 angka (0 untuk grapefruit, 1 untuk orange). Array prediksi ini kemudian akan saya bandingkan dengan nilai sebenarnya dari data testing (`y_test`).

9. Evaluasi Model
Saya menggunakan metrik yang lebih akurat: Accuracy, Precision, Recall, dan F1-score. Saya juga menyertakan Confusion Matrix untuk melihat secara detail di mana letak kesalahan klasifikasi.

Berikut adalah hasil evaluasi yang saya peroleh setelah menjalankan kode:

Decision Tree:
- Accuracy: 0.1180
- Precision: 0.1216
- Recall: 0.1180
- F1-score: 0.1194

Naive Bayes:
- Accuracy: 0.2585
- Precision: 0.0939
- Recall: 0.2585
- F1-score: 0.1368

SVM:
- Accuracy: 0.2780
- Precision: 0.0773
- Recall: 0.2780
- F1-score: 0.1209

Hal ini mengindikasikan ada sesuatu yang sangat salah. Accuracy Decision Tree hanya 11.8%, artinya dari 100 prediksi, hanya 12 yang benar. Naive Bayes dan SVM sedikit lebih baik (sekitar 25-27%) tapi masih di bawah rata-rata 50%.

10. Hasil
Hasil yang sangat rendah menegaskan bahwa tidak ada model yang mampu menemukan pola.
