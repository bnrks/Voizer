import React from "react";
import "../App.css";
const Footer = () => {
  return (
    <footer className="footer poppins-regular sticky-bottom bg-dark text-white py-3 mt-auto w-100">
      <div className="container">
        <div className="row align-items-center">
          <div className="col-md-6 text-center text-md-start">
            <p className="mb-0">&copy; 2024 Voizer. All rights reserved.</p>
          </div>
          <div className="col-md-6">
            <ul className="list-inline text-center text-md-end mb-0">
              <li className="list-inline-item">
                <a
                  href="https://github.com"
                  className="text-white text-decoration-none"
                >
                  <i className="bi bi-github"></i>
                </a>
              </li>
              <li className="list-inline-item ms-3">
                <a
                  href="https://linkedin.com"
                  className="text-white text-decoration-none"
                >
                  <i className="bi bi-linkedin"></i>
                </a>
              </li>
              <li className="list-inline-item ms-3">
                <a href="/contact" className="text-white text-decoration-none">
                  Contact
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
