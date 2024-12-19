import React, { useState, useEffect } from "react";
import { LiveAudioVisualizer } from "react-audio-visualize";
import { useMediaRecorder } from "./MediaRecorder";

const Visualizer = () => {
  const { mediaRecorder, isRecording, startRecording, stopRecording } =
    useMediaRecorder();
  const [isSpeaking, setIsSpeaking] = useState(false);

  useEffect(() => {
    if (mediaRecorder && isRecording) {
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const source = audioContext.createMediaStreamSource(mediaRecorder.stream);
      source.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const checkAudio = () => {
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        setIsSpeaking(average > 30); // Threshold for detecting speech
        requestAnimationFrame(checkAudio);
      };

      checkAudio();
    }
  }, [mediaRecorder, isRecording]);

  return (
    <div className="container mt-5 poppins-regular">
      <div className="row justify-content-center mb-4">
        <div className="col-auto">
          <button
            className="btn btn-primary me-2"
            onClick={startRecording}
            disabled={isRecording}
          >
            Start Recording
          </button>
          <button
            className="btn btn-danger"
            onClick={stopRecording}
            disabled={!isRecording}
          >
            Stop Recording
          </button>
        </div>
      </div>

      {/* Speaker Status Display */}
      {isRecording && (
        <div className="row justify-content-center mb-4">
          <div className="col-auto">
            <p className="mb-1">
              Status:{" "}
              <strong>{isSpeaking ? "Person 1 is speaking" : "Silence"}</strong>
            </p>
            <p className="mb-1">
              Subject: <strong>Car</strong>
            </p>
            <p className="mb-1">
              Emotional State: <strong>Happy</strong>
            </p>
          </div>
        </div>
      )}

      {/* Your existing visualization component */}
      {mediaRecorder && (
        <div className="row justify-content-center">
          <div className="col-auto">
            <LiveAudioVisualizer
              mediaRecorder={mediaRecorder}
              width={500}
              height={200}
              barColor="rgba(2, 21, 86, 1)"
              minDecibels={-85}
              smoothingTimeConstant={0.3}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default Visualizer;
