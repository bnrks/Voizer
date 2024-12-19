// MainPage.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import Footer from "../components/Footer";
import "../App.css";
const TrainNewModelPage = () => {
  const [characterName, setCharacterName] = useState("");
  const [characters, setCharacters] = useState([]);
  const [isStep2Enabled, setIsStep2Enabled] = useState(true); // Set to true initially
  const navigate = useNavigate();

  const handleCharacterAdd = () => {
    if (characterName.trim()) {
      const newCharacter = {
        id: Date.now(), // Her karaktere benzersiz bir ID ver
        name: characterName.trim(), // Karakter adı
        audio: null, // Başlangıçta audio null olsun
      };

      setCharacters([...characters, newCharacter]);
      setCharacterName(""); // Input'u temizle
      console.log("Character added:", newCharacter);
    }
  };

  const handleCharacterAudioUpload = (characterId, event) => {
    const file = event.target.files[0];
    if (file) {
      setCharacters(
        characters.map((char) =>
          char.id === characterId ? { ...char, audio: file } : char
        )
      );
    }
  };

  const handleTrainModel = () => {
    navigate("/loading");
  };
  const allCharactersHaveAudio =
    characters.length > 0 && characters.every((char) => char.audio);
  const canTrainModel =
    characters.length >= 2 && characters.every((char) => char.audio);
  return (
    <div>
      <Header />
      <div className="container py-4 min-vh-100" style={{ maxWidth: "800px" }}>
        <h1 className="text-center mb-4 poppins-regular">Train a new model.</h1>
        <p className="poppins-regular">
          First, add the names of the people speaking and their audio file.
          (Recommended audio file length is 80 seconds, file type is .wav).
          Then, we will train our model with the data you give us.
        </p>
        <div className="card mb-3">
          <div className="card-header bg-primary text-white py-2">
            <h5 className="mb-0 poppins-bold">1.Step: Add Characters</h5>
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

            {/* Character List */}
            <div
              className="character-list"
              style={{ maxHeight: "200px", overflowY: "auto" }}
            >
              {characters.map((character) => (
                <div key={character.id} className="card mb-2">
                  <div className="card-body p-2">
                    <div className="d-flex justify-content-between align-items-center">
                      <h6 className="mb-0 poppins-bold">{character.name}</h6>
                      <div style={{ width: "60%" }}>
                        <input
                          type="file"
                          className="form-control form-control-sm"
                          accept="audio/*"
                          onChange={(e) =>
                            handleCharacterAudioUpload(character.id, e)
                          }
                        />
                      </div>
                    </div>
                    {character.audio && (
                      <small className="text-success poppins-bold">
                        Audio uploaded.: {character.audio.name}
                      </small>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Step 3: Train Model */}
        <div className="card">
          <div className="card-header bg-primary text-white py-2">
            <h5 className="mb-0 poppins-bold">2.Step: Train Model</h5>
          </div>
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
                Please add at least 2 characters and upload audio files for all
                of them.
              </p>
            )}
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default TrainNewModelPage;
