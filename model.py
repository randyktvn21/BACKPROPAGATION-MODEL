import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle

# 2. Load dataset diabetes
print("Loading dataset...")
data = pd.read_csv('diabetes.csv')
print("Dataset shape:", data.shape)
print("\nSample data:")
print(data.head())
print("\nFeature statistics:")
print(data.describe())

# 3. Pisahkan fitur (X) dan target (y)
print("\nMemisahkan fitur dan target...")
X = data.drop('Outcome', axis=1).values  # Semua kolom kecuali Outcome
y = data['Outcome'].values.reshape(-1, 1)  # Outcome dalam bentuk 2D array

# 4. Normalisasi data dan split data training dan data test
print("\nNormalisasi dan split data...")
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Backpropagation
print("\nMembangun model backpropagation...")
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid untuk klasifikasi biner
])

# Kompilasi model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary cross-entropy untuk klasifikasi biner
              metrics=['accuracy'])

# Callback untuk early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 6. Melatih model dan evaluasi
print("\nMelatih model...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluasi model
print("\nEvaluasi model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Loss pada data uji: {test_loss:.4f}')
print(f'Akurasi pada data uji: {test_accuracy:.4f}')

# 7. Visualisasi hasil training
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# 8. Contoh prediksi
print("\nContoh prediksi dengan data baru...")
# Contoh data baru (sesuaikan dengan format dataset diabetes)
new_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # Contoh data pasien
new_data_scaled = scaler_X.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Probabilitas diabetes:", prediction[0][0])
print("Prediksi:", "Positif Diabetes" if prediction[0][0] > 0.5 else "Negatif Diabetes")

# Menampilkan ringkasan model
print("\nRingkasan arsitektur model:")
model.summary()

# Menyimpan model dan scaler
print("\nMenyimpan model dan scaler...")
model.save('diabetes_model.h5')

# Menyimpan scaler menggunakan pickle
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler_X, scaler_file)

print("\nModel dan scaler telah disimpan!")
print("- Model tersimpan sebagai: diabetes_model.h5")
print("- Scaler tersimpan sebagai: scaler.pkl")

# Cara menggunakan model yang tersimpan:
print("\nContoh cara load model:")
print("from tensorflow.keras.models import load_model")
print("import pickle")
print("loaded_model = load_model('diabetes_model.h5')")
print("with open('scaler.pkl', 'rb') as scaler_file:")
print("    loaded_scaler = pickle.load(scaler_file)")

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def load_data(self):
        # Load dataset
        data = pd.read_csv('diabetes.csv')
        
        # Separate features and target
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def build_model(self):
        # Create model architecture
        self.model = Sequential([
            Dense(16, activation='relu', input_shape=(8,)),
            Dense(12, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        
    def train_model(self, X_train, y_train, X_test, y_test):
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping]
        )
        
    def evaluate_model(self, X_test, y_test):
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy
    
    def predict(self, input_data):
        # Scale input data
        scaled_data = self.scaler.transform(input_data)
        # Make prediction
        prediction = self.model.predict(scaled_data)
        return prediction
    
    def plot_training_history(self):
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('static/training_history.png')
        plt.close()
