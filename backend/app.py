from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Agar bisa diakses dari frontend

# Load model
clf = joblib.load('backend/model_classifier.pkl')
reg = joblib.load('backend/model_regressor.pkl')

# Rute home
@app.route("/")
def home():
    return "UMKM Funding Predictor API Aktif!"

# Rute prediksi
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        features = [
            float(data['aset']),
            float(data['penjualan']),
            int(data['karyawan']),
            float(data['pertumbuhan']),
            float(data['kredit']),
            float(data['jaminan']),
            int(data['dokumen']),
        ]

        prediction = clf.predict([features])[0]

        if prediction == 1:
            status = "LAYAK"
            estimasi = reg.predict([features])[0]
        else:
            status = "TIDAK LAYAK"
            estimasi = 0.0

        return jsonify({
            "kelayakan": status,
            "estimasi_pendanaan": round(estimasi, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Menjalankan server
if __name__ == "__main__":
    app.run(debug=True)
