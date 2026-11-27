# ==============================================================================
# SUMBER DATASET KAGGLE:
# 1. Diabetes : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# 2. Jantung  : https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# ==============================================================================

import os
import joblib
import pandas as pd
from django.conf import settings
from django.shortcuts import render

KAGGLE_DIABETES_URL = "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
KAGGLE_HEART_URL = "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"

DIABETES_BUNDLE = joblib.load(os.path.join(settings.BASE_DIR, "models", "model_diabetes.joblib"))
HEART_BUNDLE = joblib.load(os.path.join(settings.BASE_DIR, "models", "model_heart.joblib"))

def index(request):
    return render(request, "advisor/index.html", {
        "kaggle_diabetes": KAGGLE_DIABETES_URL,
        "kaggle_heart": KAGGLE_HEART_URL,
    })

def prediksi_diabetes(request):
    hasil = None
    if request.method == "POST":
        features = DIABETES_BUNDLE["features"]
        input_data = {col: float(request.POST.get(col, 0)) for col in features}
        df_input = pd.DataFrame([input_data])
        
        model = DIABETES_BUNDLE["model"]
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]
        
        hasil = {
            "status": "BERISIKO TINGGI DIABETES" if pred == 1 else "RISIKO RENDAH / NORMAL",
            "berisiko": bool(pred == 1),
            "probabilitas": round(proba[1] * 100, 2),
        }
    return render(request, "advisor/diabetes.html", {"hasil": hasil, "kaggle_url": KAGGLE_DIABETES_URL})

def prediksi_jantung(request):
    hasil = None
    if request.method == "POST":
        features = HEART_BUNDLE["features"]
        input_data = {col: float(request.POST.get(col, 0)) for col in features}
        df_input = pd.DataFrame([input_data])
        
        model = HEART_BUNDLE["model"]
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]
        
        hasil = {
            "status": "BERISIKO PENYAKIT JANTUNG" if pred == 1 else "RISIKO RENDAH / NORMAL",
            "berisiko": bool(pred == 1),
            "probabilitas": round(proba[1] * 100, 2),
        }
    return render(request, "advisor/jantung.html", {"hasil": hasil, "kaggle_url": KAGGLE_HEART_URL})