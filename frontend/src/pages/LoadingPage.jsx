import { useNavigate } from "react-router-dom";
import React, { useState, useEffect } from "react";
import "../App.css";
import Header from "../components/Header";
import Footer from "../components/Footer";
const LoadingPage = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigate("/liveanalysis"); // Redirect to liveanalysis page
    }, 1000000); // 5000ms = 5 seconds

    return () => clearTimeout(timer); // Cleanup timer
  }, [navigate]);
  return (
    <div>
      <Header />
      <div
        className="container min-vh-100 d-flex flex-column justify-content-center align-items-center"
        style={{ marginTop: "-60px" }}
      >
        <div className="text-center">
          <div
            className="spinner-border text-primary"
            role="status"
            style={{ width: "4rem", height: "4rem" }}
          >
            <span className="visually-hidden">Loading...</span>
          </div>
          <h2 className="mt-4 poppins-bold">We training the model.</h2>
          <p className="text-muted poppins-regular">
            This process may take some time. Why don't you pour yourself a cup
            of tea?
          </p>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default LoadingPage;
