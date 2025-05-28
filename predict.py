from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load model dan scaler yang sudah disimpan
model = load_model('diabetes_model.h5')
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    # Membuat array dari input
    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                      insulin, bmi, diabetes_pedigree, age]])
    
    # Normalisasi data menggunakan scaler yang sama dengan training
    data_scaled = scaler.transform(data)
    
    # Melakukan prediksi
    prediction = model.predict(data_scaled)
    probability = prediction[0][0]
    
    return {
        'probability': probability,
        'prediction': 'Positif Diabetes' if probability > 0.5 else 'Negatif Diabetes'
    }

# Contoh penggunaan
if __name__ == "__main__":
    # Contoh data pasien
    result = predict_diabetes(
        pregnancies=6,
        glucose=148,
        blood_pressure=72,
        skin_thickness=35,
        insulin=0,
        bmi=33.6,
        diabetes_pedigree=0.627,
        age=50
    )
    
    print("\nHasil Prediksi:")
    print(f"Probabilitas Diabetes: {result['probability']:.2%}")
    print(f"Prediksi: {result['prediction']}") 