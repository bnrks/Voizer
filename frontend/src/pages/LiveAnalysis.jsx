import React from "react";
import Visualizer from "../components/Visualizer";
import "../App.css";
import Header from "../components/Header";
import Footer from "../components/Footer";
const LiveAnalysis = () => {
  return (
    <div>
      <Header />
      <div className="container mt-5 min-vh-100 text-center">
        <h1 className=" mb-4 poppins-bold">Live Analysis</h1>
        <p className="poppins-regular">
          {" "}
          This page will show who is speaking, what is being said and what is
          being said in real time using the data you have trained the model on.
        </p>
        <Visualizer></Visualizer>
      </div>
      <Footer />
    </div>
  );
};

export default LiveAnalysis;
