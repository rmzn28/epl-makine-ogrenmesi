# ⚽ EPL Match Predictor & Analytics

Premier League maç sonuçlarını tahmin eden ve kapsamlı analitik sunan bir makine öğrenmesi projesi.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-blue)

## 📖 Proje Hakkında

Bu proje, **2015-2026** yılları arasındaki İngiltere Premier Lig maç verilerini kullanarak:

- **Maç sonucu tahmini** (Ev sahibi galibiyeti / Beraberlik / Deplasman galibiyeti)
- **Kesin skor tahmini** (Regresyon modelleriyle)
- **Maç istatistikleri tahmini** (Şut, Korner, İsabetli Şut)
- **Sezon sonu sımülasyonu** (Monte Carlo - 10.000 simülasyon)
- **Şans faktörü analizi** (xPTS)
- **SHAP açıklanabilirlik analizi**

gibi işlemleri gerçekleştirir.

## 🧠 Kullanılan Modeller

### Sınıflandırma (Maç Sonucu)
| Model | Açıklama |
|-------|----------|
| Logistic Regression | Multinomial sınıflandırma |
| Random Forest | Ensemble ağaç modeli |
| XGBoost | Gradient boosting |

### Regresyon (Skor & İstatistik)
| Model | Açıklama |
|-------|----------|
| Linear Regression | Temel regresyon |
| Random Forest Regressor | Ensemble regresyon |

## 🔧 Özellik Mühendisliği (Feature Engineering)

- Son **3** ve **5** maçlık hareketli ortalamalar (atılan gol, yenilen gol, şut, puan)
- **Head-to-Head (H2H)** istatistikleri
- Takım form durumu (son 5 maç)
- One-hot encoded takım bilgisi

## 🚀 Kurulum

```bash
# Depoyu klonla
git clone https://github.com/rmzn28/epl-makine-ogrenmesi.git
cd epl-makine-ogrenmesi

# Gerekli kütüphaneleri yükle
pip install streamlit pandas numpy scikit-learn xgboost shap plotly joblib scipy

# Modeli eğit
python train_model.py

# Uygulamayı başlat
streamlit run app.py
```

## 📁 Proje Yapısı

```
epl-makine-ogrenmesi/
├── app.py              # Streamlit web arayüzü
├── train_model.py      # Model eğitim scripti
├── Data/               # Sezon verileri (CSV)
│   ├── 15_16.csv
│   ├── 16_17.csv
│   ├── ...
│   └── 25_26.csv
├── models/             # Eğitilmiş modeller (gitignore)
│   ├── clf_models.pkl
│   ├── reg_models.pkl
│   └── ...
├── .gitignore
└── README.md
```

## 📊 Uygulama Özellikleri

### ⚽ Maç Tahmini
- Olasılık dağılımı (pasta grafik)
- Kesin skor tahmini
- Poisson skor olasılık matrisi
- Radar karşılaştırması
- SHAP yerel etmen analizi

### 📈 Tarihsel Üstünlük
- 38 maçlık hareketli ortalama puan grafiği
- 10+ yıllık takım performans karşılaştırması

### 🏆 Sezon Simülasyonu
- Monte Carlo simülasyonu (10.000 iterasyon)
- Şampiyonluk, UCL ve küme düşme olasılıkları
- xPTS şans faktörü analizi

### 📊 Model Değerlendirme
- Accuracy, Precision, Recall, F1-Score
- MAE, MSE, R² metrikleri

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
