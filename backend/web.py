import flask
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import numpy as np
import threading
import queue
import json
import sounddevice as sd
import librosa

from speakerRecognizer import SpeakerRecognizer

app = Flask(__name__)
CORS(app)


# Global variables with explicit initialization
class GlobalState:
    def __init__(self):
        self.recognizer = SpeakerRecognizer()
        self.audio_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        self.is_running = False
        self.recording_thread = None
        self.processing_thread = None


# Create a singleton global state
global_state = GlobalState()


def audio_callback(indata, frames, time, status):
    """
    Callback to capture audio in real-time
    """
    if status:
        print(status)

    # Convert to mono if stereo
    if indata.ndim > 1:
        indata = indata[:, 0]

    global_state.audio_queue.put(indata.copy())


def process_audio():
    """
    Process audio chunks and perform speaker recognition
    """
    audio_buffer = np.array([])

    while global_state.is_running:
        try:
            while not global_state.audio_queue.empty():
                segment = global_state.audio_queue.get()
                audio_buffer = np.concatenate([audio_buffer, segment])

            if len(audio_buffer) >= global_state.recognizer.block_size:
                window = audio_buffer[: global_state.recognizer.block_size]
                audio_buffer = audio_buffer[global_state.recognizer.hop_size :]

                features = global_state.recognizer.extract_features(
                    window, is_realtime=True
                )

                if features is not None:
                    features_scaled = global_state.recognizer.scaler.transform(
                        features.reshape(1, -1)
                    )

                    prediction = global_state.recognizer.model.predict(features_scaled)
                    probabilities = global_state.recognizer.model.predict_proba(
                        features_scaled
                    )

                    global_state.prediction_queue.put(
                        {
                            "speaker": prediction[0],
                            "probability": float(np.max(probabilities)),
                        }
                    )
        except Exception as e:
            print(f"Processing error: {e}")

        sd.sleep(10)  # Small sleep to prevent tight looping


@app.route("/")
def index():
    """
    Render the main HTML page
    """
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train_model():
    """
    Train the speaker recognition model
    """
    try:
        global_state.recognizer.train_model("Sounds/")
        return jsonify({"status": "success", "message": "Model trained successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/start-recognition")
def start_recognition():
    """
    Start real-time speaker recognition
    """
    if global_state.recognizer.model is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Model not trained. Train the model first.",
                }
            ),
            400,
        )

    # Reset queues
    while not global_state.audio_queue.empty():
        global_state.audio_queue.get()
    while not global_state.prediction_queue.empty():
        global_state.prediction_queue.get()

    global_state.is_running = True

    def stream_events():
        try:
            # Start audio input stream
            with sd.InputStream(
                samplerate=global_state.recognizer.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=global_state.recognizer.block_size,
            ):
                # Start processing thread
                global_state.processing_thread = threading.Thread(target=process_audio)
                global_state.processing_thread.start()

                while global_state.is_running:
                    try:
                        if not global_state.prediction_queue.empty():
                            prediction = global_state.prediction_queue.get()
                            yield f"data: {json.dumps(prediction)}\n\n"
                        sd.sleep(100)  # Prevent tight looping
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            global_state.is_running = False
            if (
                global_state.processing_thread
                and global_state.processing_thread.is_alive()
            ):
                global_state.processing_thread.join()

    return Response(
        stream_events(), mimetype="text/event-stream", content_type="text/event-stream"
    )


@app.route("/stop-recognition", methods=["POST"])
def stop_recognition():
    """
    Stop real-time speaker recognition
    """
    global_state.is_running = False
    return jsonify({"status": "success", "message": "Recognition stopped"})
