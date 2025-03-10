import React, { useState, useEffect } from 'react';

const ModelTester = () => {
  const [models, setModels] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [directUrl, setDirectUrl] = useState('');
  const [directResponse, setDirectResponse] = useState('');
  const [directLoading, setDirectLoading] = useState(false);

  useEffect(() => {
    // Fetch models from API
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Try with the /api prefix (as configured in vite.config.js)
      const response = await fetch('/api/available_models');
      
      if (!response.ok) {
        throw new Error(`API responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Received models:', data);
      setModels(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Error fetching models:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchDirectUrl = async () => {
    if (!directUrl) return;
    
    setDirectLoading(true);
    try {
      const response = await fetch(directUrl);
      const data = await response.text();
      
      try {
        // Try to parse as JSON for prettier display
        const jsonData = JSON.parse(data);
        setDirectResponse(JSON.stringify(jsonData, null, 2));
      } catch (e) {
        // If not JSON, show as plain text
        setDirectResponse(data);
      }
    } catch (err) {
      setDirectResponse(`Error: ${err.message}`);
    } finally {
      setDirectLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto bg-white rounded-lg shadow-md">
      <h1 className="text-2xl font-bold mb-4">OMR Model API Tester</h1>
      
      <div className="mb-6 p-4 bg-gray-100 rounded-md">
        <h2 className="text-lg font-semibold mb-2">API Response</h2>
        {loading ? (
          <p className="text-blue-600">Loading models...</p>
        ) : error ? (
          <div className="text-red-600">
            <p className="font-bold">Error:</p>
            <p>{error}</p>
          </div>
        ) : (
          <div>
            <p className="mb-2">Found {models.length} models:</p>
            {models.length === 0 ? (
              <p className="italic text-gray-600">No models found</p>
            ) : (
              <ul className="list-disc pl-5">
                {models.map((model, index) => (
                  <li key={index}>{model}</li>
                ))}
              </ul>
            )}
          </div>
        )}
        <button 
          onClick={fetchModels} 
          className="mt-3 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Retry API Call
        </button>
      </div>
      
      <div className="mb-6 p-4 bg-gray-100 rounded-md">
        <h2 className="text-lg font-semibold mb-2">Try Direct URL</h2>
        <div className="flex items-center mb-3">
          <input
            type="text"
            value={directUrl}
            onChange={(e) => setDirectUrl(e.target.value)}
            placeholder="http://localhost:5000/available_models"
            className="flex-1 p-2 border border-gray-300 rounded"
          />
          <button 
            onClick={fetchDirectUrl}
            disabled={directLoading}
            className="ml-2 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-400"
          >
            {directLoading ? 'Loading...' : 'Fetch'}
          </button>
        </div>
        <div className="bg-black text-green-400 p-3 rounded-md font-mono whitespace-pre overflow-x-auto max-h-60">
          {directResponse || 'No response yet'}
        </div>
      </div>
      
      <div className="mb-6 p-4 bg-gray-100 rounded-md">
        <h2 className="text-lg font-semibold mb-2">Debug Information</h2>
        <div className="space-y-2">
          <p>
            <span className="font-bold">Vite Configuration:</span> API calls to /api/* are proxied to 
            http://localhost:5000/* with the /api prefix removed.
          </p>
          <p>
            <span className="font-bold">Backend URL:</span> {window.location.origin.replace('3000', '5000')}
          </p>
          <p>
            <span className="font-bold">Current Frontend URL:</span> {window.location.origin}
          </p>
        </div>
      </div>
    </div>
  );
};

export default ModelTester;