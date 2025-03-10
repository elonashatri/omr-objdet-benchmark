import React, { useState, useEffect } from 'react';

const ModelCard = ({ modelName, onSelect, isSelected }) => {
  const [modelInfo, setModelInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(`/api/model_info/${modelName}`);
        if (response.ok) {
          const data = await response.json();
          setModelInfo(data);
        } else {
          setError("Failed to load model information");
        }
      } catch (error) {
        console.error(`Error fetching info for ${modelName}:`, error);
        setError("Error loading model data");
      } finally {
        setIsLoading(false);
      }
    };

    fetchModelInfo();
  }, [modelName]);

  // Key parameters that might be interesting to display
  const keyParams = [
    { label: "Base", key: "model" },
    { label: "Epochs", key: "epochs" },
    { label: "Batch", key: "batch" },
    { label: "Size", key: "imgsz" }
  ];

  return (
    <div className={`border rounded overflow-hidden ${isSelected ? 'border-indigo-500 shadow-sm' : 'border-gray-200'}`}>
      <div 
        className={`px-3 py-2 flex justify-between items-center cursor-pointer ${isSelected ? 'bg-indigo-50' : 'bg-gray-50'}`}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h3 className={`text-sm font-medium ${isSelected ? 'text-indigo-700' : 'text-gray-700'}`}>
          {modelName}
          {isSelected && <span className="ml-1 text-xs bg-indigo-100 text-indigo-800 px-1 py-0.5 rounded-sm">Selected</span>}
        </h3>
        <div className="flex items-center">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onSelect(modelName);
            }}
            className={`mr-1 px-2 py-0.5 rounded text-xs font-medium ${
              isSelected 
                ? 'bg-indigo-600 text-white' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {isSelected ? 'Selected' : 'Select'}
          </button>
          <svg 
            className={`w-4 h-4 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24" 
            xmlns="http://www.w3.org/2000/svg"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
          </svg>
        </div>
      </div>
      
      {isExpanded && !isLoading && modelInfo && (
        <div className="p-2 border-t border-gray-200 text-xs bg-white">
          <div className="grid grid-cols-4 gap-x-2 gap-y-1">
            {keyParams.map(param => (
              modelInfo[param.key] !== undefined && (
                <div key={param.key} className="text-xs">
                  <span className="font-medium text-gray-600">{param.label}: </span>
                  <span className="text-gray-800">
                    {param.key === 'imgsz' ? modelInfo[param.key] : 
                     param.key === 'model' && modelInfo[param.key].length > 10 ? 
                     modelInfo[param.key].substring(0, 10) + '...' : 
                     modelInfo[param.key]}
                  </span>
                </div>
              )
            ))}
          </div>
          
          <details className="mt-1 text-xs">
            <summary className="cursor-pointer text-indigo-600 font-medium">
              More details
            </summary>
            <div className="mt-1 bg-gray-50 p-2 rounded max-h-32 overflow-y-auto">
              <pre className="text-xs text-gray-700 whitespace-pre-wrap">
                {JSON.stringify(modelInfo, null, 2)}
              </pre>
            </div>
          </details>
        </div>
      )}
      
      {isExpanded && isLoading && (
        <div className="p-2 border-t border-gray-200 flex justify-center">
          <div className="animate-spin h-4 w-4 border-t-2 border-indigo-500 rounded-full"></div>
        </div>
      )}
    </div>
  );
};

export default ModelCard;