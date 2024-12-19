import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class ModelTrainer:
    def __init__(
        self, model_path="speaker_model.joblib", scaler_path="speaker_scaler.joblib"
    ):
        """
        Initialize the Model Trainer

        :param model_path: Path to save/load trained model
        :param scaler_path: Path to save/load feature scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = StandardScaler()

    def save_model(self):
        """
        Save trained model and scaler to disk
        """
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print(f"Model saved to {self.model_path}")
            print(f"Scaler saved to {self.scaler_path}")

    def load_model(self):
        """
        Load pre-trained model and scaler from disk
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print(f"Model loaded from {self.model_path}")
                print(f"Scaler loaded from {self.scaler_path}")
                return True
        except Exception as e:
            print(f"Could not load model: {e}")
        return False

    def train_model(self, data_dir, feature_extractor):
        """
        Train speaker recognition model

        :param data_dir: Directory containing speaker audio samples
        :param feature_extractor: Feature extraction method from SpeakerRecognizer
        """
        features = []
        labels = []

        # Process audio files for each speaker
        for speaker in os.listdir(data_dir):
            speaker_path = os.path.join(data_dir, speaker)
            if os.path.isdir(speaker_path):
                for audio_file in os.listdir(speaker_path):
                    file_path = os.path.join(speaker_path, audio_file)
                    if file_path.lower().endswith((".wav", ".mp3", ".flac")):
                        feature_vector = feature_extractor(file_path)
                        if feature_vector is not None:  # Only add valid features
                            features.append(feature_vector)
                            labels.append(speaker)

        # Check if features and labels are populated
        if len(features) == 0 or len(labels) == 0:
            raise ValueError(
                "No valid audio data found for training. Please check your dataset."
            )

        # Prepare data for training
        features = np.array(features)
        labels = np.array(labels)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train SVM classifier
        self.model = SVC(kernel="rbf", probability=True)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # Save the trained model
        self.save_model()
        return self.model, self.scaler
