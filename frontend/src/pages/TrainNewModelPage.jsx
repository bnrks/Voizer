import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useMediaRecorder } from "../components/MediaRecorder";
import Header from "../components/Header";
import Footer from "../components/Footer";
import "../App.css";

const TrainNewModelPage = () => {
  const [characterName, setCharacterName] = useState("");
  const [characters, setCharacters] = useState([]);
  const [recordingForId, setRecordingForId] = useState(null);
  const [isStep2Enabled, setIsStep2Enabled] = useState(true);
  const [playingAudioId, setPlayingAudioId] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const MAX_RECORDING_TIME = 80;
  const navigate = useNavigate();
  const {
    isRecording,
    startRecording,
    stopRecording,
    recordedChunks,
    mediaRecorder,
  } = useMediaRecorder();
  useEffect(() => {
    let interval;
    if (isRecording && recordingTime < MAX_RECORDING_TIME) {
      interval = setInterval(() => {
        setRecordingTime((prev) => {
          if (prev >= MAX_RECORDING_TIME) {
            const recordingCharId = recordingForId;
            handleStopRecording(recordingCharId);
            return 0;
          }
          return prev + 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording, recordingTime]);
  const handleCharacterAdd = () => {
    if (characterName.trim()) {
      const newCharacter = {
        id: Date.now(),
        name: characterName.trim(),
        audioBlob: null,
        recordingStatus: "not_recorded", // 'not_recorded', 'recording', 'recorded'
      };

      setCharacters([...characters, newCharacter]);
      setCharacterName("");
    }
  };

  const handleStartRecording = async (characterId) => {
    setRecordingTime(0);
    setRecordingForId(characterId);
    setCharacters(
      characters.map((char) =>
        char.id === characterId
          ? { ...char, recordingStatus: "recording" }
          : char
      )
    );
    await startRecording();
  };

  const handleStopRecording = async (characterId) => {
    setRecordingTime(0);
    stopRecording();

    // Wait briefly for chunks to be available
    await new Promise((resolve) => setTimeout(resolve, 100));

    if (recordedChunks && recordedChunks.length > 0) {
      const recordedBlob = new Blob(recordedChunks, { type: "audio/wav" });

      setCharacters(
        characters.map((char) =>
          char.id === characterId
            ? { ...char, audioBlob: recordedBlob, recordingStatus: "recorded" }
            : char
        )
      );
    }
    setRecordingForId(null);
  };
  const handlePlayAudio = (characterId) => {
    const character = characters.find((char) => char.id === characterId);
    if (character?.audioBlob) {
      const audioUrl = URL.createObjectURL(character.audioBlob);
      const audio = new Audio(audioUrl);

      audio.onended = () => {
        setPlayingAudioId(null);
        URL.revokeObjectURL(audioUrl);
      };

      setPlayingAudioId(characterId);
      audio.play();
    }
  };
  const downloadRecordings = async () => {
    for (const character of characters) {
      if (character.audioBlob) {
        const url = URL.createObjectURL(character.audioBlob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${character.name.replace(/\s+/g, "_")}_voice.wav`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        // Small delay between downloads
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }
  };
  const handleTrainModel = async () => {
    await downloadRecordings();
    navigate("/loading");
  };

  const canTrainModel =
    characters.length >= 2 &&
    characters.every((char) => char.audioBlob !== null);

  return (
    <div>
      <Header />
      <div className="container py-4 min-vh-100" style={{ maxWidth: "800px" }}>
        <h1 className="text-center mb-4 poppins-regular">Train a new model</h1>
        <p className="poppins-regular">
          Add the names of the speakers and record their voice samples. Each
          recording should be around 80 seconds long.
        </p>

        <div className="card mb-3">
          <div className="card-header bg-primary text-white py-2">
            <h5 className="mb-0 poppins-bold"> Add Characters</h5>
          </div>
          <div className="card-body py-2">
            <div className="mb-2">
              <div className="input-group input-group-sm">
                <input
                  type="text"
                  className="form-control"
                  placeholder="Add character name"
                  value={characterName}
                  onChange={(e) => setCharacterName(e.target.value)}
                />
                <button
                  className="btn bg-tertiary poppins-regular"
                  onClick={handleCharacterAdd}
                  disabled={!isStep2Enabled || !characterName.trim()}
                >
                  Add
                </button>
              </div>
            </div>

            {/* Character List with Recording Controls */}
            <div
              className="character-list"
              style={{ maxHeight: "300px", overflowY: "auto" }}
            >
              {characters.map((character) => (
                <div key={character.id} className="card mb-2">
                  <div className="card-body p-2">
                    <div className="d-flex justify-content-between align-items-center">
                      <h6 className="mb-0 poppins-bold">{character.name}</h6>
                      <div className="d-flex gap-2">
                        {character.recordingStatus === "not_recorded" && (
                          <button
                            className="btn btn-sm btn-primary"
                            onClick={() => handleStartRecording(character.id)}
                            disabled={isRecording}
                          >
                            Start Recording
                          </button>
                        )}

                        {character.recordingStatus === "recording" &&
                          recordingForId === character.id && (
                            <div className="d-flex align-items-center gap-2">
                              <span className="text-primary">
                                {recordingTime}/{MAX_RECORDING_TIME}s
                              </span>
                              <button
                                className="btn btn-sm btn-danger"
                                onClick={() =>
                                  handleStopRecording(character.id)
                                }
                              >
                                Stop Recording
                              </button>
                            </div>
                          )}

                        {character.recordingStatus === "recorded" && (
                          <div className="d-flex align-items-center gap-2">
                            <button
                              className="btn btn-sm btn-success"
                              onClick={() => handlePlayAudio(character.id)}
                              disabled={playingAudioId === character.id}
                            >
                              {playingAudioId === character.id
                                ? "Playing..."
                                : "Play"}
                            </button>
                            <button
                              className="btn btn-sm btn-outline-primary"
                              onClick={() => handleStartRecording(character.id)}
                              disabled={isRecording}
                            >
                              Re-record
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Step 2: Train Model */}
        <div className="card-body py-3 text-center">
          <button
            className="btn btn-success poppins-regular"
            disabled={!canTrainModel}
            onClick={handleTrainModel}
          >
            Start Training
          </button>
          {!canTrainModel && (
            <p className="text-danger mt-2 poppins-regular">
              Please add at least 2 characters and record audio for all of them.
            </p>
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default TrainNewModelPage;
