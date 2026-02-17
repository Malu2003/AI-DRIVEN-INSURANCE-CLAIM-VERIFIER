/**
 * Result Card Component
 * Displays a section of verification results
 */

import React from 'react';

const ResultCard = ({ title, children }) => {
  return (
    <div style={styles.card}>
      <h3 style={styles.title}>{title}</h3>
      <div style={styles.content}>
        {children}
      </div>
    </div>
  );
};

const styles = {
  card: {
    border: '1px solid #e0e0e0',
    borderRadius: '8px',
    padding: '20px',
    marginBottom: '20px',
    backgroundColor: '#f9f9f9',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  },
  title: {
    margin: '0 0 15px 0',
    color: '#2c3e50',
    fontSize: '18px',
    fontWeight: '600',
    borderBottom: '2px solid #3498db',
    paddingBottom: '10px',
  },
  content: {
    fontSize: '14px',
    color: '#333',
  },
};

export default ResultCard;
