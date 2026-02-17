/**
 * Main App Component
 * Sets up routing between pages
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import UploadClaim from './pages/UploadClaim';
import Results from './pages/Results';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<UploadClaim />} />
        <Route path="/results" element={<Results />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
};

export default App;
