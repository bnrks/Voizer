// MediaRecorder.jsx
import React, { useState, useRef, useCallback } from "react";

// Rename to useMediaRecorder to follow hooks naming convention
const useMediaRecorder = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);

  const startRecording = async () => {
    try {
      setRecordedChunks([]);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false,
      });
      streamRef.current = stream;

      const recorder = new window.MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecordedChunks((prev) => [...prev, event.data]);
        }
      };

      recorder.start(100); // Start recording with 100ms time slices
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing media devices:", err);
    }
  };

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.addEventListener(
        "dataavailable",
        (event) => {
          if (event.data.size > 0) {
            setRecordedChunks((prev) => [...prev, event.data]);
          }
        },
        { once: true }
      );

      mediaRecorderRef.current.requestData(); // Request any pending data
      mediaRecorderRef.current.stop();
      streamRef.current.getTracks().forEach((track) => track.stop());
      setIsRecording(false);
    }
  }, [isRecording]);

  return {
    mediaRecorder: mediaRecorderRef.current,
    isRecording,
    startRecording,
    stopRecording,
    recordedChunks,
  };
};

export { useMediaRecorder };
