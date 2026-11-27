# Sistem Deteksi Dini Risiko Penyakit Diabetes dan Jantung Berbasis Machine Learning

## Masalah yang diselesaikan
Deteksi dini risiko penyakit kronis seperti diabetes melitus dan penyakit jantung sering terkendala keterbatasan akses pemeriksaan awal yang cepat dan terstruktur. Aplikasi web berbasis Django ini membantu melakukan skrining awal risiko diabetes dan penyakit jantung berdasarkan parameter klinis sederhana pasien menggunakan model *Machine Learning* terlatih.

## Dataset
1. **Dataset Risiko Diabetes (Pima Indians Diabetes Database)**
   * **Sumber:** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
   * **Lisensi:** CC0: Public Domain
   * **Jumlah data:** 768 baris, 8 fitur input (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`), 1 target (`Outcome`).

2. **Dataset Risiko Penyakit Jantung (Heart Disease Dataset)**
   * **Sumber:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
   * **Lisensi:** Public Domain
   * **Jumlah data:** 1.025 baris, 13 fitur input (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`), 1 target (`target`).

## Metode
Aplikasi menggunakan pendekatan klasifikasi *supervised learning* dengan algoritma **Random Forest Classifier** (`n_estimators=100`, `random_state=42`) melalui pustaka `scikit-learn`:
* **Preprocessing & Splitting:** Data dipisahkan menjadi matriks fitur (`X`) dan label target (`y`), kemudian dibagi menjadi **80% data latih (*train*)** dan **20% data uji (*test*)** menggunakan `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
* **Penyimpanan & Integrasi:** Model terlatih dan urutan nama kolom fitur disimpan menggunakan `joblib` (`models/model_diabetes.joblib` dan `models/model_heart.joblib`), lalu dimuat ke dalam aplikasi web Django (`advisor/views.py`) untuk melakukan prediksi kelas risiko beserta probabilitasnya.

## Hasil & evaluasi

### 1. Evaluasi Model Risiko Diabetes (Data Uji: 154 baris)
* **Akurasi Data Uji:** **75.97%**
* **Classification Report:**

| Kelas | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Normal / Risiko Rendah)** | 0.79 | 0.85 | 0.82 | 100 |
| **1 (Berisiko Diabetes)** | 0.68 | 0.59 | 0.63 | 54 |
| **Accuracy** | | | **0.76** | **154** |
| **Macro Avg** | 0.74 | 0.72 | 0.73 | 154 |
| **Weighted Avg** | 0.75 | 0.76 | 0.76 | 154 |

### 2. Evaluasi Model Risiko Penyakit Jantung (Data Uji: 205 baris)
* **Akurasi Data Uji:** **100.00%**
* **Classification Report:**

| Kelas | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Normal / Risiko Rendah)** | 1.00 | 1.00 | 1.00 | 100 |
| **1 (Berisiko Penyakit Jantung)** | 1.00 | 1.00 | 1.00 | 105 |
| **Accuracy** | | | **1.00** | **205** |
| **Macro Avg** | 1.00 | 1.00 | 1.00 | 205 |
| **Weighted Avg** | 1.00 | 1.00 | 1.00 | 205 |

## Cara menjalankan
1. Pastikan Python 3.10+ sudah terpasang, lalu aktifkan *virtual environment* dan instal dependensi:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt