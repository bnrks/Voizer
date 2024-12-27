import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import librosa
import numpy as np


class AudioModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.label_encoder = LabelEncoder()

    def extract_features(self, file_path):
        audio, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0).reshape(1, -1)
        return features

    def extract_features_from_array(self, audio_array):
        try:
            mfccs = librosa.feature.mfcc(y=audio_array, sr=22050, n_mfcc=13)
            features = np.mean(mfccs.T, axis=0).reshape(1, -1)
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def train(self, X, y):
        # Convert list to numpy array and ensure correct shape
        X = np.vstack(X)
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)

    def predict(self, X):
        X = np.array(X).reshape(1, -1)
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def save_model(self, save_path):
        model_file = os.path.join(save_path, "audio_model.joblib")
        encoder_file = os.path.join(save_path, "label_encoder.joblib")
        joblib.dump(self.model, model_file)
        joblib.dump(self.label_encoder, encoder_file)

    def load_model(self, save_path):
        model_file = os.path.join(save_path, "audio_model.joblib")
        encoder_file = os.path.join(save_path, "label_encoder.joblib")
        self.model = joblib.load(model_file)
        self.label_encoder = joblib.load(encoder_file)
