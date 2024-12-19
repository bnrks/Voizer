import queue
import threading
import time
import librosa
import numpy as np
import sounddevice as sd
import queue

from modelTrainer import ModelTrainer


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
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.realtime_duration = realtime_duration
        self.realtime_hop_length = realtime_hop_length

        # Audio recording and real-time parameters
        self.block_size = int(realtime_duration * sample_rate)
        self.hop_size = int(realtime_hop_length * sample_rate)

        # Real-time processing components
        self.audio_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        self.is_running = False
        self.recording_thread = None
        self.processing_thread = None
        self.print_thread = None

        # Initialize model trainer
        self.trainer = ModelTrainer(model_path=model_path, scaler_path=scaler_path)
        self.model = None
        self.scaler = None

        # Try to load existing model and scaler
        if self.trainer.load_model():
            self.model = self.trainer.model
            self.scaler = self.trainer.scaler

    def extract_features(self, audio_path_or_segment, is_realtime=False):
        """
        Extract audio features using MFCCs and spectral features
        """
        try:
            if is_realtime:
                y = audio_path_or_segment
                sr = self.sample_rate
            else:
                y, sr = librosa.load(
                    audio_path_or_segment, sr=self.sample_rate, duration=self.duration
                )

            min_length = int(
                self.sample_rate
                * (self.realtime_duration if is_realtime else self.duration)
            )
            if len(y) < min_length:
                print(f"Warning: Short audio {'segment' if is_realtime else 'file'}")
                y = np.pad(y, (0, min_length - len(y)), mode="constant")

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

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

    def train_model(self, data_dir):
        """
        Train the model using the ModelTrainer
        """
        self.model, self.scaler = self.trainer.train_model(
            data_dir, self.extract_features
        )

    def start_realtime_prediction(self):
        """
        Başlatma: Gerçek zamanlı ses tahmini.
        """

        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Reset queues
        while not self.audio_queue.empty():
            self.audio_queue.get()
        while not self.prediction_queue.empty():
            self.prediction_queue.get()

        self.is_running = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.print_thread = threading.Thread(target=self.print_predicted_data)
        self.recording_thread.start()
        self.processing_thread.start()
        self.print_thread.start()

    def stop_realtime_prediction(self):
        """
        Durdurma: Gerçek zamanlı tahmin.
        """
        self.is_running = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join()
        if self.print_thread and self.print_thread.is_alive():
            self.print_thread.join()

    def _record_audio(self):
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.block_size,
        ):
            while self.is_running:
                time.sleep(self.realtime_hop_length)

    def _process_audio(self):
        audio_buffer = np.array([])

        while self.is_running:
            try:
                while not self.audio_queue.empty():
                    segment = self.audio_queue.get()
                    audio_buffer = np.concatenate([audio_buffer, segment])

                if len(audio_buffer) >= self.block_size:
                    window = audio_buffer[: self.block_size]
                    audio_buffer = audio_buffer[self.hop_size :]

                    features = self.extract_features(window, is_realtime=True)

                    if features is not None:
                        features_scaled = self.scaler.transform(features.reshape(1, -1))

                        prediction = self.model.predict(features_scaled)
                        probabilities = self.model.predict_proba(features_scaled)

                        self.prediction_queue.put(
                            {
                                "speaker": prediction[0],
                                "probability": np.max(probabilities),
                            }
                        )
            except Exception as e:
                print(f"Processing error: {e}")

            time.sleep(0.01)

    def get_realtime_prediction(self):
        """
        Kuyruktaki tahminleri al.
        """
        if not self.prediction_queue.empty():
            return self.prediction_queue.get()
        return None

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

    def print_predicted_data(self):
        while self.is_running:
            try:
                prediction = self.get_realtime_prediction()
                if prediction:
                    print("Speaker: ", prediction["speaker"])
                    print("Probability: ", prediction["probability"])
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)  # Small sleep to prevent tight looping
            except Exception as e:
                print(f"Error in print_predicted_data: {e}")
                time.sleep(0.1)
