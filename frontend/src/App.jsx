// App.jsx
import React from "react";
import { Routes, Route } from "react-router-dom";
import MainPage from "./pages/MainPage";
import TrainNewModelPage from "./pages/TrainNewModelPage";
import LoadingPage from "./pages/LoadingPage";
import LiveAnalysis from "./pages/LiveAnalysis";

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<MainPage />} />
      <Route path="/trainnewmodel" element={<TrainNewModelPage />} />
      <Route path="/loading" element={<LoadingPage />} />
      <Route path="/liveanalysis" element={<LiveAnalysis />} />
    </Routes>
  );
};

export default App;
