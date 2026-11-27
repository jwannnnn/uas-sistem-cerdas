from django.db import models

class RiwayatKos(models.Model):
    nama_kos = models.CharField(max_length=100)
    jarak_km = models.FloatField()
    rating = models.FloatField()
    ac = models.CharField(max_length=10)
    harga_tawaran = models.FloatField()     # Harga asli pemilik kos (ribu Rp)
    estimasi_ml = models.FloatField()       # Prediksi harga wajar ML (ribu Rp)
    status_fuzzy = models.CharField(max_length=60)
    skor_fuzzy = models.FloatField()
    analisis_ai = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.nama_kos} - {self.status_fuzzy}"