import threading
import queue
import os
import pyaudio
import wave
import numpy as np
from models.audio_model import AudioModel

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 2

audio_model = AudioModel()
audio_model.load_model(
    os.path.join(os.path.dirname(__file__), "../saved_models")
)  # Update with the correct path


class RecognizerService:
    def __init__(self, model, websocket=None):
        self.model = model
        self.websocket = websocket
        self.latest_prediction = None
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording_thread = None

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._record_and_predict)
            self.recording_thread.daemon = True
            self.recording_thread.start()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread:
                self.recording_thread.join(timeout=2)
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

    def _record_and_predict(self):
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        while self.is_recording:
            try:
                frames = []
                for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    if not self.is_recording:
                        break
                    data = self.stream.read(CHUNK)
                    frames.append(data)

                if not self.is_recording:
                    break

                audio_data = np.frombuffer(b"".join(frames), dtype=np.float32)
                features = self.model.extract_features_from_array(audio_data)

                if features is not None:
                    prediction = self.model.predict(features)
                    self.latest_prediction = prediction[0]

                    if self.websocket:
                        self.websocket.emit(
                            "prediction", {"prediction": self.latest_prediction}
                        )
            except Exception as e:
                print(f"Error during recording: {e}")
                break

        self.stop_recording()

    def __del__(self):
        self.stop_recording()
        self.audio.terminate()
