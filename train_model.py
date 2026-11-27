# ==============================================================================
# SUMBER DATASET 1 (DIABETES) : Pima Indians Diabetes Database
# URL KAGGLE 1                : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# LISENSI 1                   : CC0: Public Domain
#
# SUMBER DATASET 2 (JANTUNG)  : Heart Disease Dataset
# URL KAGGLE 2                : https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# LISENSI 2                   : Public Domain
#
# CATATAN: Hanya file dataset (.csv) yang diambil dari Kaggle. Seluruh kode
#          preprocessing, training, evaluasi, dan web Django ditulis mandiri.
# ==============================================================================

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

KAGGLE_DIABETES_URL = "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
KAGGLE_HEART_URL = "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"

os.makedirs("models", exist_ok=True)

def train_and_evaluate(csv_path, target_col, model_out_path, label_name):
    print(f"\n{'='*55}\nTRAINING MODEL: {label_name}\n{'='*55}")
    df = pd.read_csv(csv_path)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Pembagian data latih 80% dan data uji 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Jumlah Data Total : {len(df)} baris")
    print(f"Fitur Input ({len(X.columns)})  : {list(X.columns)}")
    print(f"Akurasi Data Uji  : {acc * 100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump({"model": model, "features": list(X.columns)}, model_out_path)
    print(f"Model berhasil disimpan ke: {model_out_path}")

if __name__ == "__main__":
    train_and_evaluate("data/diabetes.csv", "Outcome", "models/model_diabetes.joblib", "RISIKO DIABETES")
    train_and_evaluate("data/heart.csv", "target", "models/model_heart.joblib", "RISIKO PENYAKIT JANTUNG")