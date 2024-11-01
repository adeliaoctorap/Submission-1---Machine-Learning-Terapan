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
#### 3. Terbatasnya kemampuan sistem rekomendasi dalam menangani pengguna baru dan film baru (masalah cold start)
Pengguna baru atau film yang baru ditambahkan seringkali tidak memiliki data rating yang cukup, sehingga sistem rekomendasi kesulitan memberikan rekomendasi yang akurat. Masalah ini disebut “cold start problem,” di mana sistem tidak memiliki cukup informasi historis untuk menentukan preferensi pengguna atau relevansi suatu film. Penyelesaian masalah ini memerlukan pendekatan algoritmik yang dapat memberikan rekomendasi meskipun data historis masih terbatas.
#### 4. Keterbatasan algoritma rekomendasi dalam memproses data besar secara efisien
Sistem rekomendasi membutuhkan analisis data yang besar dan kompleks, terutama pada dataset seperti MovieLens 100k, yang memiliki ribuan pengguna dan ribuan film. Algoritma yang tidak efisien akan membutuhkan waktu komputasi yang lebih lama, membebani sumber daya, dan menghasilkan hasil yang kurang responsif. Diperlukan metode yang lebih efisien dan optimal dalam memproses data besar agar sistem dapat berfungsi dengan baik di lingkungan nyata.
#### 5. Kurangnya evaluasi yang berkelanjutan untuk meningkatkan keakuratan sistem rekomendasi
Model rekomendasi perlu terus dievaluasi dan disempurnakan agar tetap relevan dan akurat seiring perubahan pola preferensi pengguna dan penambahan data baru. Tanpa evaluasi berkelanjutan, sistem rekomendasi dapat kehilangan efektivitasnya seiring waktu. Oleh karena itu, dibutuhkan rencana evaluasi rutin yang menggunakan metrik yang tepat untuk mengukur keberhasilan dan efektivitas rekomendasi yang diberikan oleh sistem.

### Goals
#### 1. Meningkatkan kemampuan sistem rekomendasi dalam menyaring pilihan film sesuai minat pengguna
Untuk mengatasi "information overload" yang dirasakan pengguna, tujuan utamanya adalah menyediakan rekomendasi yang sesuai dengan selera pengguna, sehingga mereka tidak perlu mencari film secara manual. Dengan content-based filtering, sistem dapat merekomendasikan film berdasarkan fitur dari film yang disukai pengguna, seperti genre, sutradara, atau aktor. Sebagai contoh, jika seorang pengguna cenderung menyukai film ber-genre “Drama” dan “Romance,” sistem akan merekomendasikan film dengan genre yang serupa.
#### 2. Menyediakan rekomendasi yang lebih personal dengan menyesuaikan fitur konten film berdasarkan preferensi pengguna
Penting untuk meningkatkan tingkat personalisasi rekomendasi sehingga sesuai dengan minat unik setiap pengguna. Content-based filtering dapat membantu dengan cara menyesuaikan rekomendasi berdasarkan kesamaan konten antara film yang sudah ditonton dan yang akan direkomendasikan. Pendekatan ini memungkinkan sistem untuk membuat profil pengguna berdasarkan film yang disukai, lalu mencocokkannya dengan film-film yang memiliki karakteristik serupa. Dengan begitu, rekomendasi yang diberikan lebih spesifik dan relevan.

Jawaban Pernyataan Masalah 3
Mengurangi dampak masalah cold start bagi pengguna baru dengan memanfaatkan preferensi konten eksplisit
Masalah cold start sering kali memengaruhi pengguna baru yang belum memiliki riwayat preferensi. Solusi content-based filtering dapat memanfaatkan informasi eksplisit dari pengguna baru, seperti preferensi genre atau rating awal yang diberikan pada beberapa film, untuk membangun profil preferensi awal. Berdasarkan data ini, sistem dapat memberikan rekomendasi film yang sesuai meskipun pengguna belum memiliki banyak riwayat menonton, sehingga pengalaman pengguna baru tetap personal dan memuaskan.

Jawaban Pernyataan Masalah 4
Mengoptimalkan efisiensi sistem rekomendasi dengan pendekatan berbasis konten yang dapat dikembangkan secara modular
Pendekatan content-based filtering lebih efisien dalam hal pemrosesan data karena sistem hanya perlu mengandalkan fitur konten dari film dan profil pengguna. Sistem ini dapat dikembangkan secara modular, di mana profil pengguna diperbarui secara berkala dengan menganalisis konten dari film yang telah mereka tonton. Selain itu, content-based filtering tidak membutuhkan data pengguna lain untuk melakukan rekomendasi, sehingga pemrosesan bisa lebih cepat dan tidak bergantung pada rating dari pengguna lain.

Jawaban Pernyataan Masalah 5
Menerapkan evaluasi yang berkelanjutan untuk meningkatkan akurasi rekomendasi konten
Agar sistem rekomendasi tetap efektif, perlu dilakukan evaluasi berkelanjutan menggunakan metrik evaluasi yang tepat, seperti precision atau recall, yang mengukur relevansi rekomendasi yang diberikan. Melalui evaluasi ini, sistem dapat dioptimalkan berdasarkan umpan balik pengguna atau akurasi hasil rekomendasi yang diberikan. Content-based filtering memungkinkan pemantauan terus-menerus terhadap pola preferensi pengguna, karena sistem dapat memperbarui profil pengguna sesuai dengan film yang baru saja mereka tonton.

Goals Utama Proyek dengan Content-Based Filtering
Memberikan rekomendasi yang relevan: Memastikan bahwa setiap rekomendasi sesuai dengan preferensi konten pengguna, sehingga meningkatkan kepuasan dan pengalaman pengguna dalam menemukan film yang menarik.
Meningkatkan personalisasi: Membangun profil pengguna berdasarkan film yang disukai dan memberikan rekomendasi film serupa dengan fitur yang sama, sehingga pengalaman menonton lebih dipersonalisasi.
Mengatasi cold start: Memanfaatkan fitur konten untuk memberikan rekomendasi kepada pengguna baru, bahkan ketika mereka belum memiliki riwayat yang panjang.
Efisiensi pemrosesan: Mengurangi ketergantungan pada data pengguna lain dengan menggunakan informasi konten, sehingga mempercepat proses rekomendasi.
Pemeliharaan berkelanjutan: Memperbarui profil pengguna sesuai dengan film yang ditonton, memungkinkan sistem untuk terus relevan dan akurat dalam jangka panjang.
Pendekatan content-based filtering ini dapat diimplementasikan secara efektif dalam proyek, di mana setiap tujuan diarahkan untuk menjawab permasalahan utama dengan solusi yang dapat diukur dan dioptimalkan seiring waktu.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

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

