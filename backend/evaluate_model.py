import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('dataset.csv')

# Konversi label ke angka
df['Label'] = df['Label'].map({'LAYAK': 1, 'TIDAK LAYAK': 0})

# Fitur yang digunakan
features = [
    'Total Aset (Juta)',
    'Penjualan Rata-rata Per Tahun (Juta)',
    'Jumlah Tenaga Kerja',
    'Proyeksi Pertumbuhan (%/tahun)',
    'Kebutuhan Biaya Kredit (Juta)',
    'Nilai Aset Jaminan Kredit',
    'Jumlah Dokumen Kredit'
]

# ========== Evaluasi KLASIFIKASI ==========
# Load model classifier
clf = joblib.load('model_classifier.pkl')

# Split data klasifikasi
X = df[features]
y_class = df['Label']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Prediksi dan evaluasi
y_pred_class = clf.predict(X_test_c)
print("===== Evaluasi Klasifikasi =====")
print("Akurasi:", accuracy_score(y_test_c, y_pred_class))
print("\nClassification Report:\n", classification_report(y_test_c, y_pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, y_pred_class))


# ========== Evaluasi REGRESI ==========
# Load model regressor
reg = joblib.load('model_regressor.pkl')

# Filter hanya data yang LAYAK untuk regresi
df_layak = df[df['Label'] == 1].copy()
X_reg = df_layak[features]
y_reg = df_layak['pendanaan']

# Split data regresi
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Prediksi dan evaluasi
y_pred_reg = reg.predict(X_test_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_reg))

print("\n===== Evaluasi Regresi =====")
print("MAE:", mean_absolute_error(y_test_r, y_pred_reg))
print("MSE:", mean_squared_error(y_test_r, y_pred_reg))
print("RMSE:", rmse)
print("R² Score:", r2_score(y_test_r, y_pred_reg))
