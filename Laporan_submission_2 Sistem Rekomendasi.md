# Laporan Proyek Machine Learning - Adelia Octora Pristisahida

## Domain Proyek
Latar belakang proyek ini berfokus pada analisis preferensi pengguna dan pembuatan sistem rekomendasi film menggunakan dataset MovieLens 100k. Dataset ini, yang mencakup 100.000 ulasan pengguna dari berbagai film, telah menjadi salah satu sumber utama dalam penelitian dan pengembangan sistem rekomendasi karena menyediakan beragam metadata yang memungkinkan analisis mendalam tentang pola preferensi penonton. Dengan hadirnya platform streaming yang semakin populer seperti Netflix, Disney+, dan Amazon Prime, kebutuhan akan sistem rekomendasi yang efektif semakin meningkat. Sistem ini membantu pengguna menemukan film sesuai minat mereka di tengah koleksi besar pilihan yang tersedia, sehingga meningkatkan kepuasan dan keterlibatan pengguna terhadap platform.

Sistem rekomendasi telah terbukti menjadi alat penting dalam meningkatkan pengalaman pengguna melalui rekomendasi yang lebih relevan dan personal. Berdasarkan literatur, sistem rekomendasi umumnya menggunakan pendekatan berbasis konten (content-based filtering), kolaboratif (collaborative filtering), atau hibrida (hybrid) untuk menyarankan item sesuai preferensi penggunaan algoritma pembelajaran mesin, sistem ini dapat mempelajari pola preferensi dari data historis, seperti film yang sebelumnya ditonton dan diulas oleh pengguna lain dengan selera serupa, sehingga menghasilkan rekomendasi yang dipersonalisasi .

Proyek ini menggunakan pendekatan berbasis konten (content-based filtering) untuk menyelesaika masalah rekomendasi. Dengan mengimplementasikan algororitma machine learning pada dataset ini, proyek ini diharapkan dapat menghasilkan sistem rekomendasi film yang efisien dan responsif. Hal ini tidak hanya bermanfaat bagi pengguna dengan menyediakan pilihan film yang sesuai dengan selera mereka, tetapi juga bagi pengembang platform streaming, yang dapat memanfaatkan informasi rekomendasi untuk meningkatkan kepuasan pengguna. Secara keseluruhan, proyek ini bertujuan untuk mengembangkan rekomendasi yang relevan dan dapat disesuaikan dengan berbagai profil pengguna, sehingga memberikan pengalaman menonton yang lebih personal dan mendalam.

Referensi:
[Introduction to Recommender Systems Handbook](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1)
[Recommender Systems: The Textbook](https://link.springer.com/book/10.1007/978-3-319-29659-3)
[Sistem Rekomendasi Film Menggunakan Content Based Filtering](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163)

## Business Understanding
Masalah utama yang dihadapi adalah kesulitan pengguna dalam menemukan film yang sesuai dengan preferensi mereka di tengah ribuan pilihan. Hal ini disebabkan oleh jumlah film yang sangat banyak, yang dapat membingungkan pengguna. juga perbedaan preferensi individu yang sulit diidentifikasi secara manual tanpa bantuan algoritma. Serta keterbatasan dalam merekomendasikan film baru bagi pengguna baru atau film yang belum banyak ditonton oleh pengguna lain.

### Problem Statements
#### 1. Keterbatasan dalam menemukan film yang sesuai dengan preferensi pengguna di tengah banyaknya pilihan
Pengguna sering kali menghadapi “information overload” saat memilih film di platform streaming karena jumlah pilihan yang sangat banyak. Tanpa bantuan sistem rekomendasi, pengguna mungkin sulit menemukan film yang sesuai dengan minat atau preferensi mereka, sehingga pengalaman menonton menjadi kurang memuaskan. Oleh karena itu, perlu adanya sistem rekomendasi yang dapat mempermudah pengguna dalam menemukan film yang relevan berdasarkan preferensi mereka.
#### 2. Kurangnya personalisasi rekomendasi yang mempertimbangkan preferensi unik setiap pengguna
Setiap pengguna memiliki preferensi yang berbeda terhadap genre, gaya, dan jenis film. Tanpa analisis yang akurat terhadap pola preferensi pengguna, sistem rekomendasi yang disediakan cenderung memberikan rekomendasi yang bersifat umum dan kurang personal. Hal ini menurunkan efektivitas rekomendasi dan tidak memaksimalkan kepuasan pengguna. Dibutuhkan pendekatan yang dapat menangkap karakteristik dan kebiasaan menonton masing-masing pengguna secara akurat.

### Goals
#### 1. Meningkatkan kemampuan sistem rekomendasi dalam menyaring pilihan film sesuai minat pengguna
Untuk mengatasi "information overload" yang dirasakan pengguna, tujuan utamanya adalah menyediakan rekomendasi yang sesuai dengan selera pengguna, sehingga mereka tidak perlu mencari film secara manual. Dengan content-based filtering, sistem dapat merekomendasikan film berdasarkan fitur dari film yang disukai pengguna, seperti genre, sutradara, atau aktor. Sebagai contoh, jika seorang pengguna cenderung menyukai film ber-genre “Drama” dan “Romance,” sistem akan merekomendasikan film dengan genre yang serupa.
#### 2. Menyediakan rekomendasi yang lebih personal dengan menyesuaikan fitur konten film berdasarkan preferensi pengguna
Content-based filtering dapat membantu dengan cara menyesuaikan rekomendasi berdasarkan kesamaan konten antara film yang sudah ditonton dan yang akan direkomendasikan. Pendekatan ini memungkinkan sistem untuk membuat profil pengguna berdasarkan film yang disukai, lalu mencocokkannya dengan film-film yang memiliki karakteristik serupa. Dengan begitu, rekomendasi yang diberikan lebih spesifik dan relevan.
#### 3. Mengoptimalkan efisiensi sistem rekomendasi dengan pendekatan berbasis konten yang dapat dikembangkan secara modular
Pendekatan content-based filtering lebih efisien dalam hal pemrosesan data karena sistem hanya perlu mengandalkan fitur konten dari film dan profil pengguna. 

### Solution statements
#### 1. Membangun Model Content-Based Filtering Berbasis Similarity (Cosine Similarity dan TF-IDF)
Deskripsi Solusi: Model pertama akan menggunakan pendekatan Content-Based Filtering dengan representasi fitur film berdasarkan metadata seperti genre, sutradara, atau aktor. Metode yang digunakan untuk mengukur kesamaan antar film adalah Cosine Similarity pada vektor TF-IDF. Dengan pendekatan ini, sistem akan merekomendasikan film yang mirip berdasarkan konten yang telah ditonton pengguna sebelumnya.
#### 2. Menambahkan Metode Evaluasi
Metrik evaluasi yang digunakan adalah Mean Average Precision (MAP), Recall@k, dan Precision@K, untuk mengukur seberapa relevan rekomendasi yang diberikan oleh model. Hasil ini akan memberikan gambaran akurasi dari rekomendasi berdasarkan similaritas antar film.


## Data Understanding
Dataset yang digunakan dalam proyek ini adalah MovieLens 100k, yang merupakan salah satu dataset paling populer untuk penelitian sistem rekomendasi. Dataset ini terdiri dari 100.000 penilaian film yang diberikan oleh pengguna yang berbeda, termasuk informasi tentang film, pengguna, dan genre film. Dengan struktur data yang kaya, MovieLens 100k memungkinkan analisis yang mendalam mengenai preferensi pengguna dan karakteristik film, yang sangat sesuai untuk pendekatan content-based filtering. Data ini dapat diunduh dari situs resmi MovieLens:[MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/).

### Variabel-variabel pada MovieLens 100k dataset adalah sebagai berikut:
Dataset MovieLens 100k memiliki 3 file utama, ketiga file inilah yang digunakan dalam sistem rekomendasi yang dibangun, antara lain:
u.data: File ini berisi informasi penilaian dengan kolom user_id, movie_id, rating, dan timestamp.
u.item: File ini berisi informasi tentang film dengan kolom movie_id, title, dan genres.
u.user: File ini berisi informasi tentang penguna dengan kolom user_id, age, gender, occupation, dan zip_code 

**Rubrik/Kriteria Tambahan (Opsional)**:
Dalam upaya memahami dataset MovieLens 100k, langkah yang diambil adalah melakukan Univariate Exploratory Data Analysis (EDA). Univariate EDA berfokus pada analisis satu variabel pada satu waktu untuk mendapatkan wawasan yang lebih dalam tentang distribusi dan karakteristik data. Beberapa tahapan yang dilakukan dalam analisis ini meliputi:
1. Analisis Distribusi Rating: Untuk memahami sebaran penilaian yang diberikan oleh pengguna, kita dapat menggunakan histogram untuk memvisualisasikan frekuensi setiap nilai rating (1 hingga 5). Ini membantu dalam mengidentifikasi apakah ada bias dalam penilaian, misalnya, jika sebagian besar pengguna cenderung memberikan penilaian tinggi.
2. Visualisasi Genre Film: Dengan menggunakan diagram lingkaran atau grafik batang, kita dapat menganalisis proporsi berbagai genre film dalam dataset. Ini akan memberikan gambaran tentang genre mana yang paling banyak tersedia dan bagaimana distribusi genre film dapat mempengaruhi preferensi pengguna.
3. Analisis Usia Pengguna: Untuk variabel usia dari pengguna, kita dapat menggunakan box plot untuk melihat sebaran usia dan mengidentifikasi outlier. Ini memberikan wawasan mengenai kelompok umur mana yang paling aktif dalam memberikan penilaian film.
4. Statistik Deskriptif: Selain visualisasi, perhitungan statistik deskriptif seperti mean, median, modus, dan rentang juga dilakukan untuk memberikan ringkasan numerik dari variabel yang ada. Ini akan memberikan gambaran tentang pusat, penyebaran, dan bentuk distribusi data.

Melalui tahapan Univariate EDA ini, kita dapat memperoleh pemahaman yang lebih baik tentang dataset, karakteristik variabel, dan pola yang mungkin ada. Pemahaman ini penting untuk merancang dan mengembangkan sistem rekomendasi yang lebih efektif berdasarkan preferensi pengguna dan karakteristik film.

## Data Preparation
Data preparation ini meliputi
#### 1. Penggabungan Data: 
Setelah memuat kedua DataFrame, langkah berikutnya adalah menggabungkan informasi pengguna dengan rating mereka. Penggabungan dilakukan menggunakan metode merge() pada Pandas, dengan user_id sebagai kunci penggabungan. Ini memungkinkan kita untuk memiliki satu DataFrame yang berisi semua informasi yang relevan, termasuk rating, usia, dan pekerjaan pengguna.
#### 2. Pemeriksaan Missing Values:
Memeriksa apakah terdapat missing values dalam dataset. Dengan menggunakan fungsi isnull() dan sum(), kita dapat menghitung jumlah missing values untuk setiap kolom dalam DataFrame. Setelah mengetahui kolom mana yang memiliki missing values, langkah selanjutnya adalah menganalisis dampak dari missing values tersebut. 
#### 3. Penghapusan Kolom yang tidak relevan
Dari analisis yang dilakukan, ditemukan kolom video_release_date, IMDb URL, dan unknown yang tidak memberikan kontribusi signifikan terhadap model rekomendasi. Kolom-kolom ini dapat menyebabkan kebingungan dan tidak memberikan informasi tambahan yang berguna. Kolom-kolom yang tidak relevan akan dihapus dari DataFrame menggunakan metode drop(). Ini membantu menjaga dataset tetap bersih dan fokus pada fitur yang lebih penting bagi analisis dan model rekomendasi. Pembersihan data membantu meningkatkan kualitas dataset dengan menghapus atau menangani missing values, sehingga analisis yang dilakukan menjadi lebih akurat. Data yang bersih dan lengkap mengurangi kemungkinan adanya bias atau kesalahan dalam model yang dibangun.
#### 4. Mengurutkan berdasarkan user_id
Setelah melakukan pembersihan dan penghapusan kolom yang tidak relevan, langkah berikutnya adalah mengurutkan film berdasarkan user_id. Ini penting untuk mempersiapkan data sebelum dimasukkan ke dalam model.
#### 5. Menghapus data duplikat
Selanjutnya, kita hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, kita perlu menghapus data yang duplikat dengan fungsi drop_duplicates(). Dalam hal ini, kita membuang data duplikat pada kolom ‘user_id’ dan 'title'. Dengan langkah ini, film akan terurut berdasarkan user_id, dan data siap untuk diproses lebih lanjut dalam sistem rekomendasi.
#### 6. Konversi Data Series ke List
Setelah mengurutkan film dan memasukkannya ke dalam variabel film, kita perlu melakukan konversi data dari series menjadi list. Hal ini berguna untuk mempermudah pengolahan data dalam model rekomendasi.
#### 7. Membuat dictionary
Setelah kita memiliki list untuk user_id, item_id, title, dan genre, langkah selanjutnya adalah membuat dictionary. Dictionary ini akan membantu kita dalam menyimpan informasi terkait pengguna dan film dengan cara yang lebih terstruktur. Dengan mengurutkan data dan membuat dictionary, struktur data menjadi lebih jelas dan terorganisir. Hal ini memudahkan dalam mengakses dan memanipulasi data, sehingga proses pengembangan model rekomendasi menjadi lebih efisien.

Tahapan data preparation memastikan bahwa dataset siap untuk dimasukkan ke dalam model rekomendasi. Ini mencakup pembuatan format data yang diperlukan untuk algoritma yang akan digunakan, seperti dalam pendekatan content-based filtering yang memanfaatkan fitur-fitur dari film. Dengan melalui tahapan data preparation ini, kami dapat memastikan bahwa sistem rekomendasi yang dibangun lebih efektif dan mampu memberikan rekomendasi yang relevan kepada pengguna berdasarkan preferensi mereka.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

