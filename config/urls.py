from django.contrib import admin
from django.urls import path
from advisor import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('diabetes/', views.prediksi_diabetes, name='prediksi_diabetes'),
    path('jantung/', views.prediksi_jantung, name='prediksi_jantung'),
]