import React from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import Footer from "../components/Footer";
import "../App.css";
import { ReactTyped } from "react-typed";
const MainPage = () => {
  const navigate = useNavigate();

  return (
    <div>
      <Header />
      <div className="container-fluid  color-primary poppins-regular min-vh-100">
        {/* Welcome Header Section */}
        <div className="row py-5  m-0">
          <div className="col-12 text-center">
            <h1 className="display-3 mb-3 poppins-bold ">
              <ReactTyped
                strings={["Voizer: Real-time audio analyzer"]}
                typeSpeed={50}
                backSpeed={50}
                loop
              ></ReactTyped>
            </h1>
            <p className=" w-75 fs-4 mx-auto poppins-extralight">
              Analyze speech patterns in real-time using our pre-trained model
              or train a new model with your custom dataset.
            </p>
          </div>
        </div>

        <div className="row justify-content-evenly h-100">
          {/* Left Section - Use Trained Model */}
          <div className="col-5 d-flex flex-column justify-content-center align-items-center rounded-5 main-page-shadow hover-class bg-secondary">
            <div className="text-center p-5">
              <h2 className="mb-4">Use Pre-trained Model</h2>
              <p className="mb-4">
                Start analyzing speech patterns immediately using our
                pre-trained model
              </p>
              <button
                className="btn bg-primary text-white btn-lg"
                onClick={() => navigate("/liveanalysis")}
              >
                Start Analysis
              </button>
            </div>
          </div>

          {/* Right Section - Train New Model */}
          <div className="col-5 d-flex flex-column justify-content-center align-items-center rounded-5 main-page-shadow hover-class bg-secondary">
            <div className="text-center p-5">
              <h2 className="mb-4 ">Train New Model</h2>
              <p className="mb-4">
                Create and train a new model with your custom dataset
              </p>
              <button
                className="btn bg-primary text-white btn-lg"
                onClick={() => navigate("/trainnewmodel")}
              >
                Train Model
              </button>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default MainPage;
