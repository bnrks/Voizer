import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tempfile
import joblib
import threading
import queue
import time


class SpeakerRecognizer:
    def __init__(
        self,
        sample_rate=22050,
        duration=3,
        realtime_duration=1,
        realtime_hop_length=0.5,
        model_path="speaker_model.joblib",
        scaler_path="speaker_scaler.joblib",
    ):
        """
        Initialize Speaker Recognizer with configuration parameters

        :param sample_rate: Audio sampling rate (default 22050 Hz)
        :param duration: Recording duration in seconds (default 3)
        :param realtime_duration: Real-time processing window duration
        :param realtime_hop_length: Overlap between real-time prediction windows
        :param model_path: Path to save/load trained model
        :param scaler_path: Path to save/load feature scaler
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.realtime_duration = realtime_duration
        self.realtime_hop_length = realtime_hop_length
        self.model_path = model_path
        self.scaler_path = scaler_path

        # Audio recording and real-time parameters
        self.block_size = int(realtime_duration * sample_rate)
        self.hop_size = int(realtime_hop_length * sample_rate)

        # Model and feature components
        self.model = None
        self.scaler = StandardScaler()

        # Real-time processing components
        self.audio_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        self.is_running = False
        self.recording_thread = None
        self.processing_thread = None

        # Try to load existing model and scaler
        self.load_model()

    def extract_features(self, audio_path_or_segment, is_realtime=False):
        """
        Extract audio features using MFCCs and spectral features

        :param audio_path_or_segment: Path to audio file or audio segment
        :param is_realtime: Flag to indicate real-time processing
        :return: Feature vector or None if processing fails
        """
        try:
            # Load audio file or use audio segment
            if is_realtime:
                y = audio_path_or_segment
                sr = self.sample_rate
            else:
                y, sr = librosa.load(
                    audio_path_or_segment, sr=self.sample_rate, duration=self.duration
                )

            # Ensure minimum length for feature extraction
            min_length = int(
                self.sample_rate
                * (self.realtime_duration if is_realtime else self.duration)
            )
            if len(y) < min_length:
                print(f"Warning: Short audio {'segment' if is_realtime else 'file'}")
                y = np.pad(y, (0, min_length - len(y)), mode="constant")

            # Extract Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # Extract additional features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

            # Aggregate features
            features = np.concatenate(
                [
                    np.mean(mfccs, axis=1),
                    [np.mean(spectral_centroids)],
                    [np.mean(spectral_bandwidth)],
                ]
            )

            return features

        except Exception as e:
            print(f"Error processing {'segment' if is_realtime else 'file'}: {e}")
            return None

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

    def record_audio(self, filename, duration=None):
        """
        Record audio from microphone

        :param filename: Output audio file name
        :param duration: Optional custom recording duration
        """
        record_duration = duration if duration is not None else self.duration
        print(f"Recording {record_duration} seconds of audio...")
        recording = sd.rec(
            int(record_duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float64",
        )
        sd.wait()
        sf.write(filename, recording, self.sample_rate)
        print(f"Recording saved as {filename}")

    def train_model(self, data_dir):
        """
        Train speaker recognition model

        :param data_dir: Directory containing speaker audio samples
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
                        feature_vector = self.extract_features(file_path)
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

    def predict_from_microphone(self):
        """
        Predict speaker using real-time microphone input
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Create a temporary file for the recording
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_filename = temp_audio.name

        try:
            # Record audio from microphone
            self.record_audio(temp_filename, duration=3)

            # Extract features
            features = self.extract_features(temp_filename)

            if features is None:
                raise ValueError("Could not extract features from the audio file.")

            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Predict speaker
            prediction = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)

            # Return prediction results
            return prediction[0], np.max(probabilities)

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None

        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass

    def predict_speaker(self, audio_path):
        """
        Predict speaker for a given audio file

        :param audio_path: Path to audio file for prediction
        :return: Predicted speaker and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Extract features and scale
        features = self.extract_features(audio_path)

        if features is None:
            raise ValueError("Could not extract features from the audio file.")

        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict speaker
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)

        return prediction[0], np.max(probabilities)

    # Real-time prediction methods
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for audio recording

        :param indata: Input audio data
        :param frames: Number of frames
        :param time: Timestamp
        :param status: Status of recording
        """
        if status:
            print(status)

        # Convert to mono if stereo
        if indata.ndim > 1:
            indata = indata[:, 0]

        # Add audio segment to queue
        self.audio_queue.put(indata.copy())

    def start_realtime_recording(self):
        """
        Start real-time audio recording and prediction
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Reset queues
        while not self.audio_queue.empty():
            self.audio_queue.get()
        while not self.prediction_queue.empty():
            self.prediction_queue.get()

        # Set running flag
        self.is_running = True

        # Start audio recording thread
        def recording_loop():
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.block_size,
            ):
                while self.is_running:
                    time.sleep(self.realtime_hop_length)

        # Start processing thread
        def processing_loop():
            audio_buffer = np.array([])
            while self.is_running:
                try:
                    # Get audio segments from queue
                    while not self.audio_queue.empty():
                        segment = self.audio_queue.get()
                        audio_buffer = np.concatenate([audio_buffer, segment])

                    # Slide window for prediction
                    if len(audio_buffer) >= self.block_size:
                        # Take a window and remove processed part
                        window = audio_buffer[: self.block_size]
                        audio_buffer = audio_buffer[self.hop_size :]

                        # Extract features
                        features = self.extract_features(window, is_realtime=True)

                        if features is not None:
                            # Scale features
                            features_scaled = self.scaler.transform(
                                features.reshape(1, -1)
                            )

                            # Predict speaker
                            prediction = self.model.predict(features_scaled)
                            probabilities = self.model.predict_proba(features_scaled)

                            # Put prediction in queue
                            self.prediction_queue.put(
                                {
                                    "speaker": prediction[0],
                                    "confidence": np.max(probabilities),
                                }
                            )

                except Exception as e:
                    print(f"Processing error: {e}")

                time.sleep(0.1)  # Prevent tight loop

        # Start threads
        self.recording_thread = threading.Thread(target=recording_loop)
        self.processing_thread = threading.Thread(target=processing_loop)

        self.recording_thread.start()
        self.processing_thread.start()

        print("Real-time speaker recognition started...")

    def get_latest_prediction(self):
        """
        Retrieve the latest speaker prediction

        :return: Latest prediction dictionary or None
        """
        try:
            return self.prediction_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_realtime_recording(self):
        """
        Stop real-time audio recording and prediction
        """
        self.is_running = False

        if self.recording_thread:
            self.recording_thread.join()
        if self.processing_thread:
            self.processing_thread.join()

        print("Real-time speaker recognition stopped.")


def main():
    print("Speaker Recognition App")
    recognizer = SpeakerRecognizer()

    while True:
        # Main menu
        print("\nSpeaker Recognition App")
        print("1. Train Model")
        print("2. Record Audio Sample")
        print("3. Predict Speaker from File")
        print("4. Predict Speaker from Microphone")
        print("5. Start Real-time Speaker Recognition")
        print("6. Exit")

        choice = input("Enter your choice (1/2/3/4/5/6): ")

        try:
            if choice == "1":
                # Train model with speaker audio samples
                data_directory = input(
                    "Enter directory path with speaker audio samples: "
                )
                recognizer.train_model(data_directory)
                print("Model training completed successfully!")

            elif choice == "2":
                # Record audio sample
                filename = input("Enter output filename (e.g., speaker1_sample.wav): ")
                recognizer.record_audio(filename)

            elif choice == "3":
                # Predict speaker from file
                if recognizer.model is None:
                    print("Please train the model first!")
                    continue

                audio_file = input("Enter audio file path for speaker prediction: ")
                try:
                    speaker, confidence = recognizer.predict_speaker(audio_file)
                    print(f"Predicted Speaker: {speaker}")
                    print(f"Confidence: {confidence * 100:.2f}%")
                except ValueError as e:
                    print(f"Prediction Error: {e}")

            elif choice == "4":
                # Predict speaker from microphone
                if recognizer.model is None:
                    print("Please train the model first!")
                    continue

                try:
                    speaker, confidence = recognizer.predict_from_microphone()
                    if speaker and confidence:
                        print(f"Predicted Speaker: {speaker}")
                        print(f"Confidence: {confidence * 100:.2f}%")
                except ValueError as e:
                    print(f"Prediction Error: {e}")

            elif choice == "5":
                # Real-time speaker recognition
                if recognizer.model is None:
                    print("Please train the model first!")
                    continue

                recognizer.start_realtime_recording()

                try:
                    while True:
                        prediction = recognizer.get_latest_prediction()
                        if prediction:
                            print(f"Detected Speaker: {prediction['speaker']} ")
                            print(f"Confidence: {prediction['confidence'] * 100:.2f}%")
                            time.sleep(0.5)
                except KeyboardInterrupt:
                    recognizer.stop_realtime_recording()
            elif choice == "6":
                print("Exiting...")
                break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
