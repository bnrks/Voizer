import React from "react";
import logo from "../assets/logo-resized.png";
import "../App.css";
const Header = () => {
  return (
    <header className="fixed-top  shadow-sm header-bg">
      <div className="container-fluid py-2">
        <div className="row align-items-center">
          <div className="col">
            <img
              src={logo}
              alt="Logo"
              className="img-fluid logo"
              style={{ height: "80px" }} // Fixed height for logo
            />
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
