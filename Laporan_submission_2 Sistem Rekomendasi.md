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
Bagian ini berusaha menjawab problem statement 2 dengan pendekatan yang lebih personal, yaitu mencocokkan fitur konten film yang relevan dengan preferensi individu pengguna. Dengan content-based filtering, sistem dapat menangkap preferensi unik pengguna sehingga rekomendasi menjadi lebih relevan dan personal.

### Solution statements
#### 1. Membangun Model Content-Based Filtering Berbasis Similarity (Cosine Similarity dan TF-IDF)
Deskripsi Solusi: Model pertama akan menggunakan pendekatan Content-Based Filtering dengan representasi fitur film berdasarkan metadata seperti genre, sutradara, atau aktor. Metode yang digunakan untuk mengukur kesamaan antar film adalah Cosine Similarity pada vektor TF-IDF. Dengan pendekatan ini, sistem akan merekomendasikan film yang mirip berdasarkan konten yang telah ditonton pengguna sebelumnya.
#### 2. Menambahkan Metode Evaluasi
Metrik evaluasi yang digunakan adalah Mean Average Precision (MAP), Recall@k, dan Precision@K, untuk mengukur seberapa relevan rekomendasi yang diberikan oleh model. Hasil ini akan memberikan gambaran akurasi dari rekomendasi berdasarkan similaritas antar film.


## Data Understanding
Dataset yang digunakan dalam proyek ini adalah MovieLens 100k, yang merupakan salah satu dataset paling populer untuk penelitian sistem rekomendasi. Dataset ini terdiri dari 100.000 penilaian film yang diberikan oleh pengguna yang berbeda, termasuk informasi tentang film, pengguna, dan genre film. Dengan struktur data yang kaya, MovieLens 100k memungkinkan analisis yang mendalam mengenai preferensi pengguna dan karakteristik film, yang sangat sesuai untuk pendekatan content-based filtering. Data ini dapat diunduh dari situs resmi MovieLens:[MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/).

### Variabel-variabel pada MovieLens 100k dataset adalah sebagai berikut:
Dataset MovieLens 100k memiliki 3 file utama, ketiga file inilah yang digunakan dalam sistem rekomendasi yang dibangun, antara lain:
1. u.data: File ini berisi informasi penilaian dengan 4 kolom, yaitu user_id, movie_id, rating, dan timestamp dan memiliki 10000 baris non-null. Dengan rincian sebagai berikut:
   |   |  Column    |   Non-Null Count |  Dtype |
   |---|------------|------------------|--------|
   | 0 |  user_id   |   100000 non-null|  int64 |
   | 1 |  item_id   |   100000 non-null|  int64 |
   | 2 |  rating    |   100000 non-null|  int64 |
   | 3 |  timestamp |  100000 non-null |  int64 |

   Tidak terdapat adanya missing value dan dengan data unik sebagai berikut:
    - Banyak data pengguna:  943
    - Banyak data film:  1682
    - Banyak data rating:  5
    - Banyak data waktu:  49282
   Uraian fitur pada data:
    - user_id: ID pengguna unik. Setiap pengguna memiliki ID yang berbeda, yang memungkinkan kita melacak riwayat penilaian atau preferensi dari masing-masing pengguna.
    - item_id (atau movie_id): ID film unik. Setiap film memiliki ID yang unik yang membantu mengidentifikasi film tertentu dalam dataset.
    - rating: Penilaian atau rating yang diberikan oleh pengguna untuk film tertentu, biasanya dalam rentang 1 hingga 5. Rating ini menunjukkan preferensi atau penilaian pengguna terhadap film tersebut.
    - timestamp: Waktu (dalam bentuk UNIX timestamp) saat rating tersebut diberikan. Ini memungkinkan analisis waktu, seperti tren penilaian atau perubahan preferensi pengguna dari waktu ke waktu.

3. u.user: File ini berisi informasi tentang penguna dengan 5 kolom, yaitu user_id, age, gender, occupation, dan zip_code dan memiliki 943 baris. Dengan rincian sebagai berikut:
   |   |  Column    |  Non-Null Count | Dtype| 
   |---|------------|-----------------|------|
   | 0 |  user_id   |  943 non-null   |int64 |
   | 1 |  age       |  943 non-null   |int64 |
   | 2 |  gender    |  943 non-null   |object|
   | 3 |  occupation|  943 non-null   |object|
   | 4 |  zip_code  |  943 non-null   |object|
   
   Tidak terdapat adanya missing value dan dengan data unik sebagai berikut:
    - Banyak data pengguna:  943
    - Banyak data usia:  61
    - Jenis kelamin pengguna:  [M, F]
    - Jenis pekerjaan pengguna:  ['technician' 'other' 'writer' 'executive' 'administrator' 'student'
 'lawyer' 'educator' 'scientist' 'entertainment' 'programmer' 'librarian'
 'homemaker' 'artist' 'engineer' 'marketing' 'none' 'healthcare' 'retired'
 'salesman' 'doctor']
    Uraian fitur pada data:
     - user_id: ID pengguna unik. Setiap pengguna memiliki ID yang berbeda, yang memungkinkan kita melacak riwayat penilaian atau preferensi dari masing-masing pengguna.
     - age : Usia pengguna. Informasi ini digunakan untuk analisis demografis, misalnya, untuk melihat preferensi berdasarkan kelompok usia tertentu.
     - gender: Menunjukkan jenis kelamin pengguna, pria atau wanita, Male atau Female.
     - occupation : Pekerjaan pengguna. Ini dapat digunakan untuk analisis preferensi berdasarkan pekerjaan atau profesi.
     - zip_code : Kode pos pengguna. Data lokasi dapat memberikan informasi tentang preferensi film berdasarkan wilayah atau area geografis.
       
5. u.item: File ini berisi informasi tentang film dengan 3 kolom, yaitu movie_id, title, dan genres dan memiliki 1681 baris. Dari data di atas kita mengetahui jumlah masing-masing dari jenis genre film yang ada. kita memiliki 943 pengguna dari 1682 film yang memiliki rating. Dengan rincian sebagai berikut:
|   |  Column            | Non-Null Count|  Dtype   |
|---|--------------------|---------------|----------|  
| 0 |  item_id           |  1682 non-null|   int64  | 
| 1 |  title             |  1682 non-null|   object | 
| 2 |  release_date      |  1681 non-null|   object | 
| 3 |  video_release_date|  0 non-null   |   float64|
| 4 |  IMDb URL          |  1679 non-null|   object |
| 5 |  unknown           |  1682 non-null|   int64  |
| 6 |  Action            |  1682 non-null|   int64  |
| 7 |  Adventure         |  1682 non-null|   int64  |
| 8 |  Animation         |  1682 non-null|   int64  |
| 9 |  Children’s        |  1682 non-null|   int64  |
| 10|  Comedy            |  1682 non-null|   int64  |
| 11|  Crime             |  1682 non-null|   int64  |
| 12|  Documentary       |  1682 non-null|   int64  |
| 13|  Drama             |  1682 non-null|   int64  |
| 14|  Fantasy           |  1682 non-null|   int64  |
| 15|  Film-Noir         |  1682 non-null|   int64  |
| 16|  Horror            |  1682 non-null|   int64  |
| 17|  Musical           |  1682 non-null|   int64  |
| 18|  Mystery           |  1682 non-null|   int64  |
| 19|  Romance           |  1682 non-null|   int64  |
| 20|  Sci-Fi            |  1682 non-null|   int64  |
| 21|  Thriller          |  1682 non-null|   int64  |
| 22|  War               |  1682 non-null|   int64  |
| 23|  Western           |  1682 non-null|   int64  |

Dari sekian banyak kolom di atas, terdapat missing value pada kolom release_date, video_release_date dan IMDb URL.  
Adapun penjelasan fitur dari variabel di atas
- item_id: ID unik untuk setiap film dalam dataset. Kolom ini berguna untuk mengidentifikasi setiap film, memungkinkan analisis atau pencocokan antar-tabel jika dataset terdiri dari lebih dari satu tabel.
- title: Judul film. Kolom ini menyimpan nama atau judul dari setiap film dalam dataset.
- release_date: Tanggal rilis film. Informasi ini berguna untuk memahami periode ketika film dirilis dan mungkin untuk melihat tren preferensi berdasarkan era tertentu.
- Genre film. Ini membantu dalam analisis preferensi pengguna berdasarkan genre. Genre yang ada adalah Action, Adventure, Animation, Children’s, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, dan Western.
- Video_release_date: Tanggal rilis video dari film tersebut. Kolom ini tidak memiliki data (0 non-null), sehingga bisa diabaikan atau dihapus dalam proses analisis karena tidak ada nilai yang tersedia untuk diolah.
- IMDb URL: URL IMDb dari film. Kolom ini mengarah ke halaman IMDb, yang bisa berguna untuk mengambil informasi tambahan tentang film seperti sutradara, pemeran, atau ulasan. Namun, karena data ini tidak lengkap (hanya 1679 dari 1682 baris yang berisi data).
     
Dalam upaya memahami dataset MovieLens 100k, langkah yang diambil adalah melakukan Univariate Exploratory Data Analysis (EDA). Univariate EDA berfokus pada analisis satu variabel pada satu waktu untuk mendapatkan wawasan yang lebih dalam tentang distribusi dan karakteristik data. Beberapa tahapan yang dilakukan dalam analisis ini meliputi:
1. Analisis Distribusi Rating: Untuk memahami sebaran penilaian yang diberikan oleh pengguna, kita dapat menggunakan histogram untuk memvisualisasikan frekuensi setiap nilai rating (1 hingga 5). Ini membantu dalam mengidentifikasi apakah ada bias dalam penilaian, misalnya, jika sebagian besar pengguna cenderung memberikan penilaian tinggi. Untuk lebih jelasnya dapat melihat histogram berikut ![image](https://github.com/user-attachments/assets/b64d1e9d-f9c5-4fac-9191-a2c0355304b2) Terlihat dari histogram tersebut, sebagian besar pengguna memberikan rating 4 dan rating 3
2. Analisis Usia Pengguna: Untuk variabel usia dari pengguna, kita dapat menggunakan box plot untuk melihat sebaran usia dan mengidentifikasi outlier. Ini memberikan wawasan mengenai kelompok umur mana yang paling aktif dalam memberikan penilaian film. Ditunjukkan pada gambar berikut. ![image](https://github.com/user-attachments/assets/da169fa1-f57c-448e-b938-4d516db83eb2) Dari gambar tersebut dapat teramati untuk kelompok umur yang aktif adalah di pertengahan 20 hingga pertengahan 40.
3. Visualisasi Genre Film: Dengan menggunakan diagram lingkaran atau grafik batang, kita dapat menganalisis proporsi berbagai genre film dalam dataset. Ini akan memberikan gambaran tentang genre mana yang paling banyak tersedia dan bagaimana distribusi genre film dapat mempengaruhi preferensi pengguna. ![image](https://github.com/user-attachments/assets/52b6d6aa-7267-4282-8413-6435e3975257) ![image](https://github.com/user-attachments/assets/6fd7219b-0540-4d21-948d-d6c43dc217c8)
4. Statistik Deskriptif: Selain visualisasi, perhitungan statistik deskriptif seperti mean, median, modus, dan rentang juga dilakukan untuk memberikan ringkasan numerik dari variabel yang ada. Ini akan memberikan gambaran tentang pusat, penyebaran, dan bentuk distribusi data. Untuk data rating ditunjukkan sebagai berikut:
- Mean Rating: 3.52986
- Median Rating: 4.0
- Mode Rating: 4

Melalui tahapan Univariate EDA ini, kita dapat memperoleh pemahaman yang lebih baik tentang dataset, karakteristik variabel, dan pola yang mungkin ada. Pemahaman ini penting untuk merancang dan mengembangkan sistem rekomendasi yang lebih efektif berdasarkan preferensi pengguna dan karakteristik film.

## Data Preparation
Data preparation ini meliputi
#### 1. Penggabungan Data: 
Setelah memuat kedua DataFrame, langkah berikutnya adalah menggabungkan informasi pengguna dengan rating mereka. Penggabungan dilakukan menggunakan metode merge() pada Pandas, dengan user_id sebagai kunci penggabungan. Ini memungkinkan kita untuk memiliki satu DataFrame yang berisi semua informasi yang relevan, termasuk rating, usia, dan pekerjaan pengguna.
#### 2. Penghapusan Kolom yang tidak relevan
Dari analisis yang dilakukan, ditemukan kolom video_release_date, IMDb URL, dan unknown yang tidak memberikan kontribusi signifikan terhadap model rekomendasi. Kolom-kolom ini dapat menyebabkan kebingungan dan tidak memberikan informasi tambahan yang berguna. Kolom-kolom yang tidak relevan akan dihapus dari DataFrame menggunakan metode drop(). Ini membantu menjaga dataset tetap bersih dan fokus pada fitur yang lebih penting bagi analisis dan model rekomendasi. Pembersihan data membantu meningkatkan kualitas dataset dengan menghapus atau menangani missing values, sehingga analisis yang dilakukan menjadi lebih akurat. Data yang bersih dan lengkap mengurangi kemungkinan adanya bias atau kesalahan dalam model yang dibangun.
#### 3. Mengurutkan berdasarkan user_id
Setelah melakukan pembersihan dan penghapusan kolom yang tidak relevan, langkah berikutnya adalah mengurutkan film berdasarkan user_id. Ini penting untuk mempersiapkan data sebelum dimasukkan ke dalam model.
#### 4. Menghapus data duplikat
Selanjutnya, kita hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, kita perlu menghapus data yang duplikat dengan fungsi drop_duplicates(). Dalam hal ini, kita membuang data duplikat pada kolom ‘user_id’ dan 'title'. Dengan langkah ini, film akan terurut berdasarkan user_id, dan data siap untuk diproses lebih lanjut dalam sistem rekomendasi.
#### 5. Konversi Data Series ke List
Setelah mengurutkan film dan memasukkannya ke dalam variabel film, kita perlu melakukan konversi data dari series menjadi list. Hal ini berguna untuk mempermudah pengolahan data dalam model rekomendasi.
#### 6. Membuat dictionary
Setelah kita memiliki list untuk user_id, item_id, title, dan genre, langkah selanjutnya adalah membuat dictionary. Dictionary ini akan membantu kita dalam menyimpan informasi terkait pengguna dan film dengan cara yang lebih terstruktur. Dengan mengurutkan data dan membuat dictionary, struktur data menjadi lebih jelas dan terorganisir. Hal ini memudahkan dalam mengakses dan memanipulasi data, sehingga proses pengembangan model rekomendasi menjadi lebih efisien.
#### 7. Menghapus duplikat untuk memastikan hanya satu entri per film
Untuk menghapus duplikat dalam dataset untuk memastikan hanya satu entri per film, dapat menggunakan metode drop_duplicates() dalam Python dengan pandas untuk kolom item_id yang mungkin berisi entri duplikat.
#### 8. Representasi Fitur dengan TF-IDF
TF-IDF (Term Frequency-Inverse Document Frequency) ini digunakan untuk memberikan bobot pada setiap kata dalam genre film. TF-IDF membantu dalam menyoroti kata yang unik dalam genre tiap film, sehingga membuat representasi fitur dari setiap film dalam bentuk vektor. Implementasinya dengan cara menghitung nilai TF-IDF untuk setiap kata yang muncul dalam genre dari film. Setiap film akan direpresentasikan sebagai vektor dalam ruang fitur berbasis kata, dengan bobot TF-IDF yang menggambarkan kepentingan relatif kata tersebut.

Tahapan data preparation memastikan bahwa dataset siap untuk dimasukkan ke dalam model rekomendasi. Ini mencakup pembuatan format data yang diperlukan untuk algoritma yang akan digunakan, seperti dalam pendekatan content-based filtering yang memanfaatkan fitur-fitur dari film. Dengan melalui tahapan data preparation ini, kami dapat memastikan bahwa sistem rekomendasi yang dibangun lebih efektif dan mampu memberikan rekomendasi yang relevan kepada pengguna berdasarkan preferensi mereka.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

### Model Development dengan Content Based Filtering
#### 1. Perhitungan Similarity dengan Cosine Similarity
Setelah mendapatkan vektor TF-IDF untuk setiap film, cosine similarity digunakan untuk mengukur kemiripan antara vektor film yang berbeda. Rumus Cosine Similarity: cosine_similarity(A,B) = ∥A∥∥B∥/A⋅B
​di mana 𝐴 dan 𝐵 adalah vektor TF-IDF dari dua film yang dibandingkan. Cosine similarity memberikan nilai antara -1 hingga 1, di mana nilai yang lebih tinggi menunjukkan bahwa dua film memiliki tingkat kemiripan konten yang tinggi.
Kelebihan:
- Independen Terhadap Panjang Vektor: Memungkinkan perbandingan antara dokumen dengan panjang yang berbeda.
- Interpretasi yang Jelas: Memberikan nilai yang mudah dipahami antara 0 (tidak mirip) dan 1 (sangat mirip).
Kekurangan:
- Tidak Mempertimbangkan Urutan Kata: Mengabaikan urutan kata dalam teks, yang bisa kehilangan konteks makna.
- Sensitif terhadap Dimensi: Meskipun tidak terpengaruh oleh panjang vektor, bisa terpengaruh oleh distribusi fitur.

#### 2. Rekomendasi Berdasarkan Similarity
Berdasarkan skor cosine similarity, rekomendasi film akan diberikan. Film yang memiliki cosine similarity tertinggi dengan film yang sedang ditonton atau disukai pengguna akan direkomendasikan. Pada sistem rekomendasi ini, evaluasi menggunakan metrik seperti Precision@K, Recall@K, dan MAP (Mean Average Precision) untuk menilai performa sistem dalam memberikan rekomendasi yang relevan bagi pengguna. 
Rekomendasi film yang mirip dengan 'Ridicule (1996)':
|  |                                     title |
|--|-------------------------------------------|
|0 | Scream of Stone (Schrei aus Stein) (1991) |
|1 |                       One Fine Day (1996) |

Hasil rekomendasi yang diberikan oleh model menunjukkan film-film yang dianggap memiliki kemiripan dengan Ridicule (1996) berdasarkan pendekatan content-based filtering. Dengan interpretasi Scream of Stone (Schrei aus Stein) (1991) dan One Fine Day (1996) dipilih oleh model karena memiliki kemiripan fitur dengan Ridicule (1996), dalam aspek genre film.

#### 3. Proses Improvement Model dengan Hyperparameter Tuning
Meskipun TF-IDF dan cosine similarity adalah teknik yang cukup sederhana, kita bisa meningkatkan hasil model dengan melakukan tuning pada beberapa parameter dalam tahap preprocessing dan penghitungan. 
Pengaturan Parameter dalam TF-IDF:
- max_features: Membatasi jumlah fitur yang digunakan untuk representasi TF-IDF agar hanya kata-kata yang paling relevan yang diambil.
- ngram_range: Mengubah rentang n-gram (misalnya, menggunakan bigrams) untuk menangkap informasi konteks yang lebih baik.
- sublinear_tf: Menggunakan penilaian sublinear untuk mengurangi bobot frekuensi kata yang sangat tinggi.
Proses Tuning:
- Menggunakan teknik Grid Search atau Random Search untuk mencoba kombinasi parameter yang berbeda.
- Menggunakan cross-validation untuk menilai performa dari kombinasi parameter yang berbeda.
- Memilih parameter yang menghasilkan nilai evaluasi terbaik (misalnya, Precision, Recall).

## Evaluation
### Metrik Evaluasi yang Digunakan
#### 1. Precision@K 
Precision@K mengukur proporsi rekomendasi yang relevan dari total K rekomendasi yang diberikan. Ini memberikan informasi tentang seberapa banyak rekomendasi yang dihasilkan sistem yang sesuai dengan preferensi pengguna.
Rumusnya:
Precision@K = Jumlah Rekomendasi Relevan 𝐾 / Jumlah Rekomendasi Relevan
​
#### 2. Recall@K
Recall@K mengukur seberapa banyak item relevan yang berhasil direkomendasikan dari total item relevan yang tersedia. Ini memberikan gambaran tentang kemampuan sistem untuk menangkap semua item relevan.
Rumusnya: Recall@K = Jumlah Rekomendasi Relevan / Total Item Relevan

#### 3. Mean Average Precision (MAP):
MAP adalah rata-rata dari precision pada setiap titik di mana sebuah item relevan ditemukan. Ini memberikan gambaran yang lebih baik tentang kualitas model dalam memberikan rekomendasi.
Rumusnya: MAP = 1/∣𝑄∣ ∑ 𝑞∈𝑄 AP(𝑞)
di mana 
𝐴𝑃(𝑞) adalah average precision untuk kueri q.

### Analisis Hasil:
Pada sistem rekomendasi coba dimasukkan film Ridicule (1996) untuk dicari film yang direkomendasikan oleh sistem. Hasil film dari rekomendasi sistem setelah menonton film Ridicule (1996), sistem merekomendasikan film Scream of Stone (Schrei aus Stein) (1991) dan One Fine Day (1996). Lalu jika digunakan metrik evaluasinya menghasilkan berikut 
Hasil Evaluasi untuk rekomendasi film 'Ridicule (1996)':
Precision@5: 0.4000
Recall@5: 1.0000
Mean Average Precision (MAP): 1.0000

#### Precision@5 (0.4):
Precision@5: 0.4000: Artinya, dari 5 rekomendasi teratas yang diberikan, 40% di antaranya (2 film dari 5) adalah film yang relevan atau sesuai dengan data uji. Ini menunjukkan bahwa sebagian besar rekomendasi yang diberikan model cukup relevan.
#### Recall@5 (1.0):
Recall@5: 1.0000: Recall mencapai 100%, yang berarti model berhasil menemukan semua film yang relevan dalam daftar rekomendasinya. Ini adalah hasil yang sangat baik karena menunjukkan bahwa model tidak melewatkan item relevan di antara 5 rekomendasi teratas.
#### Mean Average Precision (MAP 1.0):
Mean Average Precision (MAP): 1.0000: MAP dengan nilai 1.0 menunjukkan bahwa semua film yang relevan muncul dalam urutan yang ideal di antara rekomendasi. Ini berarti semua film relevan berada di posisi teratas, yang menandakan bahwa model memberikan rekomendasi yang sangat relevan dalam urutan yang optimal.

### Kesimpulan
Dari problem statement yang diuraikan bahwa adanya keterbatasan dalam menemukan film yang sesuai dengan preferensi pengguna di tengah banyaknya pilihan dan kurangnya personalisasi rekomendasi yang mempertimbangkan preferensi unik setiap pengguna, solusi yang ditawarkan adalah membangun model content-Based filtering berbasis similarity menggunakan algoritma machine learning serta tidak lupa untuk menambahkan metode evaluasi. Maka hasil yang diharapkan adalah dapat meningkatnya kemampuan sistem rekomendasi dalam menyaring pilihan film sesuai minat pengguna dan menyediakan rekomendasi yang lebih personal dengan menyesuaikan fitur konten film berdasarkan preferensi pengguna. Dari hasil menunjukkan adanya dua rekomendasi film dari objek film Ridicule (1996). Ini cukup untuk memberikan variasi rekomendasi kepada pengguna, sebagai saran untuk menonton film yang kurang lebih bergenre sama dengan film Ridicule (1996).

Dari data-data yang telah dilampirkan di atas secara keseluruhan, model ini memiliki potensi untuk memberikan dampak positif bagi pemahaman bisnis yang terkait dengan kepuasan dan pengalaman pengguna. Jika hasil evaluasi model menunjukkan performa yang baik, dapat disimpulkan bahwa solusi yang diterapkan sudah efektif dalam menjawab masalah utama dan mencapai tujuan yang diharapkan.



