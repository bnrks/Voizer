import os
import logging
from models.audio_model import AudioModel

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self):
        self.audio_model = AudioModel()
        self.model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "saved_models"
        )
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def train_model(self, data_directory):
        if not os.path.exists(data_directory):
            raise ValueError(f"Directory {data_directory} does not exist")

        features = []
        labels = []

        class_dirs = [
            d
            for d in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory, d))
        ]

        if not class_dirs:
            raise ValueError(f"No class directories found in {data_directory}")

        for label in class_dirs:
            path = os.path.join(data_directory, label)
            audio_files = [f for f in os.listdir(path) if f.endswith((".wav", ".mp3"))]

            if not audio_files:
                logger.warning(f"No audio files found in {path}")
                continue

            for audio_file in audio_files:
                file_path = os.path.join(path, audio_file)
                feature = self.audio_model.extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(label)

        if not features:
            raise ValueError("No features extracted from any audio files")

        logger.info(f"Training with {len(features)} samples")
        self.audio_model.train(features, labels)
        self.audio_model.save_model(self.model_dir)
        return True
