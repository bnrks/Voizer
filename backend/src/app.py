import os
import logging
import threading
import queue
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from services.training_service import TrainingService
from services.recognizer_service import RecognizerService
from models.audio_model import AudioModel

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
training_service = TrainingService()
logger = logging.getLogger(__name__)

recognition_queue = queue.Queue()
audio_model = AudioModel()

# Create absolute path for sounds folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "sounds")

audio_model.load_model(
    os.path.join(BASE_DIR, "saved_models")
)  # Update with the correct path

recognizer_service = None


# Create directory structure
def create_sample_structure():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)


create_sample_structure()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    try:
        if not os.listdir(UPLOAD_FOLDER):
            return jsonify(
                {"success": False, "error": "No data found in sounds folder"}
            )

        training_service.train_model(UPLOAD_FOLDER)
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/recognize", methods=["POST"])
def recognize():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    recognition_queue.put(file_path)
    return jsonify({"message": "File received, processing..."}), 200


@app.route("/get_prediction", methods=["GET"])
def get_prediction():
    global recognizer_service
    if recognizer_service is None:
        return jsonify({"prediction": "Recording not started yet"}), 200

    if recognizer_service.latest_prediction:
        return jsonify({"prediction": recognizer_service.latest_prediction}), 200
    else:
        return jsonify({"prediction": "No prediction yet"}), 200


@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recognizer_service
    try:
        if recognizer_service is None:
            recognizer_service = RecognizerService(audio_model, socketio)
        recognizer_service.start_recording()
        return jsonify({"message": "Started recording"}), 200
    except Exception as e:
        logger.error(f"Error starting recording: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    global recognizer_service
    try:
        if recognizer_service:
            recognizer_service.stop_recording()
            return jsonify({"message": "Stopped recording"}), 200
        return jsonify({"message": "No active recording"}), 200
    except Exception as e:
        logger.error(f"Error stopping recording: {str(e)}")
        return jsonify({"error": str(e)}), 500


@socketio.on("connect")
def handle_connect():
    print("Client connected")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
