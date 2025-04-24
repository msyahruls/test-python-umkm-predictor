import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# Load dataset
df = pd.read_csv('dataset.csv')

# Fitur
features = [
    'Total Aset (Juta)',
    'Penjualan Rata-rata Per Tahun (Juta)',
    'Jumlah Tenaga Kerja',
    'Proyeksi Pertumbuhan (%/tahun)',
    'Kebutuhan Biaya Kredit (Juta)',
    'Nilai Aset Jaminan Kredit',
    'Jumlah Dokumen Kredit'
]

# Konversi label
df['Label'] = df['Label'].map({'LAYAK': 1, 'TIDAK LAYAK': 0})

# Split data untuk klasifikasi
X = df[features]
y_class = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(random_state=42) #RandomForestClassifier
clf.fit(X_train, y_train)

# Evaluasi klasifikasi
y_pred_class = clf.predict(X_test)
print("Evaluasi Klasifikasi:")
print("Akurasi:", accuracy_score(y_test, y_pred_class))
print("Classification Report:\n", classification_report(y_test, y_pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_class))

# Simpan classifier
joblib.dump(clf, 'model_classifier.pkl')

# Regresi hanya untuk data 'LAYAK'
df_layak = df[df['Label'] == 1]
X_reg = df_layak[features]
y_reg = df_layak['pendanaan']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train regressor
reg = RandomForestRegressor(random_state=42) #RandomForestRegressor
reg.fit(X_train_r, y_train_r)

# Evaluasi regresi
y_pred_reg = reg.predict(X_test_r)
print("Evaluasi Regresi:")
print("MAE:", mean_absolute_error(y_test_r, y_pred_reg))
print("MSE:", mean_squared_error(y_test_r, y_pred_reg))
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_reg))
print("RMSE:", rmse)
print("R² Score:", r2_score(y_test_r, y_pred_reg))

# Simpan regressor
joblib.dump(reg, 'model_regressor.pkl')
