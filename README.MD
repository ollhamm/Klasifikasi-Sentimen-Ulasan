# Petunjuk Menjalankan Aplikasi

Aplikasi ini terdiri dari backend yang ditulis menggunakan Flask dan frontend menggunakan Next.js. Berikut langkah-langkah untuk menjalankan aplikasi ini:

## Persiapan

Pastikan Anda telah menginstal Python, Node.js, dan npm di komputer Anda sebelum menjalankan aplikasi ini.

## Instalasi Paket

1. Buka terminal atau command prompt.
2. Masuk ke direktori `backend`.
3. Jalankan perintah berikut untuk menginstal paket Flask dan dependensinya:

    ```bash
    pip install -r requirements.txt
    ```

4. Kembali ke direktori utama aplikasi.
5. Masuk ke direktori `frontend`.
6. Jalankan perintah berikut untuk menginstal paket-paket yang dibutuhkan untuk frontend:

    ```bash
    npm install
    ```

## Menjalankan Backend

1. Kembali ke direktori utama aplikasi.
2. Masuk ke direktori `backend`.
3. Jalankan perintah berikut untuk menjalankan backend Flask:

    ```bash
    python src/app.py
    ```

## Menjalankan Frontend

1. Kembali ke direktori utama aplikasi.
2. Masuk ke direktori `frontend`.
3. Jalankan perintah berikut untuk menjalankan frontend Next.js:

    ```bash
    npm run dev
    ```

## Mengakses Aplikasi

Setelah backend dan frontend berjalan, Anda dapat mengakses aplikasi melalui browser dengan membuka `http://localhost:3000`.

