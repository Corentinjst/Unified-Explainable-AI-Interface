/**
 * Main App component with routing
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, NavLink } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import Analysis from './pages/Analysis';
import Compare from './pages/Compare';
import './App.css';

function App() {
  return (
    <AppProvider>
      <Router>
        <div className="app">
          <nav className="navbar">
            <div className="nav-container">
              <Link to="/" className="nav-logo">
                🔬 Unified XAI Interface
              </Link>
              <div className="nav-links">
                <NavLink to="/" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"} end>
                  Analysis
                </NavLink>
                <NavLink to="/compare" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
                  Compare
                </NavLink>
              </div>
            </div>
          </nav>

          <main className="main-content">
            <Routes>
              <Route path="/" element={<Analysis />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="*" element={<Analysis />} />
            </Routes>
          </main>

          <footer className="footer">
            <p>&copy; 2026 Unified Explainable AI Interface | Built with React + FastAPI</p>
          </footer>
        </div>
      </Router>
    </AppProvider>
  );
}

export default App;
