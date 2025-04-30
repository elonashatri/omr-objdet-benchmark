import React, { useState, useEffect, useRef } from 'react';
import './index.css';
import MusicXMLViewer from './components/MusicXMLViewer';

// Mock model data (as in your code)
const MOCK_MODEL_DATA = {
  "train8": {
    "imgsz": 1024,
    "epochs": 120,
    "batch": 8,
    "patience": 80,
    "mosaic": 0.5,
    "box": 7.5,
    "lr0": 0.005,
    "pretrained": true,
    "optimizer": "AdamW",
    "data": "doremiv2",
    "model": "yolov8l.pt"
  },
  "train11": {
    "imgsz": 1280,
    "epochs": 100,
    "batch": 16,
    "patience": 50,
    "mosaic": 0.4,
    "box": 8.0,
    "lr0": 0.001,
    "pretrained": true,
    "optimizer": "SGD",
    "data": "doremiv2",
    "model": "yolov8m.pt"
  },
  "doremiv1": {
    "imgsz": 1280,
    "epochs": 100,
    "batch": 16,
    "patience": 50,
    "mosaic": 0.4,
    "box": 8.0,
    "lr0": 0.001,
    "pretrained": true,
    "optimizer": "SGD",
    "data": "doremiv1 v3",
    "model": "yolov8m.pt"
  },
  "train-202-24classes-yolo-9654-data-splits": {
    "imgsz": 1280,
    "epochs": 100,
    "batch": 16,
    "patience": 50,
    "mosaic": 0.4,
    "box": 8.0,
    "lr0": 0.001,
    "pretrained": true,
    "optimizer": "SGD",
    "data": "doremiv2",
    "model": "yolov8m.pt"
  },
  "train3-yolo-9654-data-splits": {
    "imgsz": 1280,
    "epochs": 100,
    "batch": 16,
    "patience": 50,
    "mosaic": 0.4,
    "box": 8.0,
    "lr0": 0.001,
    "pretrained": true,
    "optimizer": "SGD",
    "data": "doremiv2",
    "model": "yolov8m.pt"
  },  
  "experiment-1-staffline-enhacment-april": {
    "imgsz": 1280,
    "epochs": 100,
    "batch": 16,
    "patience": 50,
    "mosaic": 0.4,
    "box": 8.0,
    "lr0": 0.001,
    "pretrained": true,
    "optimizer": "SGD",
    "data": "doremiv2",
    "model": "yolov8m.pt"
  },
  "phase12": {
    "imgsz": 1280,
    "epochs": 130,
    "batch": 12,
    "patience": 70,
    "mosaic": 0.45,
    "box": 8.5,
    "lr0": 0.002,
    "pretrained": true,
    "optimizer": "Adam",
    "data": "combined_dataset",
    "model": "yolov8l.pt"
  },
  "staffline_extreme": {
    "imgsz": 1280,
    "epochs": 150,
    "batch": 4,
    "patience": 100,
    "mosaic": 0.3,
    "box": 10.0,
    "lr0": 0.01,
    "pretrained": true,
    "optimizer": "auto",
    "data": "staffline_enhanced_dataset",
    "model": "yolov8x.pt"
  },
  "faster-rcnn-full-no-staff": {
    "imgsz": 1024,
    "epochs": 100,
    "batch": 8,
    "patience": 50,
    "mosaic": 0.0,
    "box": 1.0,
    "lr0": 0.005,
    "pretrained": true,
    "optimizer": "SGD",
    "data": "full_dataset_no_staff",
    "model": "faster-rcnn"
  },
  "faster-rcnn-full-with-staff": {
    "imgsz": 1024,
    "epochs": 100,
    "batch": 8,
    "patience": 50,
    "mosaic": 0.0,
    "box": 1.0,
    "lr0": 0.005,
    "pretrained": true,
    "optimizer": "SGD",
    "data": "full_dataset_with_staff",
    "model": "faster-rcnn"
  },
  "faster-rcnn-half-older_config": {
    "imgsz": 800,
    "epochs": 80,
    "batch": 4,
    "patience": 40,
    "mosaic": 0.0,
    "box": 1.0,
    "lr0": 0.001,
    "pretrained": true,
    "optimizer": "SGD",
    "data": "half_dataset",
    "model": "faster-rcnn"
  },
  "faster-rcnn-staff-half-older_config": {
    "imgsz": 800,
    "epochs": 80,
    "batch": 4,
    "patience": 40,
    "mosaic": 0.0,
    "box": 1.0,
    "lr0": 0.001,
    "pretrained": true,
    "optimizer": "SGD",
    "data": "staff_half_dataset",
    "model": "faster-rcnn"
  }
};

// ModelCard component
const ModelCard = ({ model, data, isSelected, onSelect }) => {
  return (
    <div
      className={`flex-shrink-0 w-60 p-2 border rounded-md cursor-pointer ${
        isSelected 
          ? 'border-indigo-500 bg-indigo-50' 
          : 'border-gray-200 hover:border-indigo-300'
      }`}
      onClick={() => onSelect(model)}
    >
      <div className="font-medium text-sm mb-1 truncate">{model}</div>
      <div className="text-xs text-gray-500">
        <div className="grid grid-cols-2 gap-x-2 gap-y-1">
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500"></span>
            <span>IMG: {data.imgsz}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
            <span>EP: {data.epochs}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-purple-500"></span>
            <span>Batch: {data.batch}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-orange-500"></span>
            <span>Aug: {data.mosaic}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-red-500"></span>
            <span>BoxL: {data.box}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
            <span>LR: {data.lr0}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// ModelDetails component
const ModelDetails = ({ model, data }) => {
  return (
    <div className="mb-4 bg-gray-50 p-3 rounded-md text-xs max-h-64 overflow-y-auto">
      <h4 className="font-medium text-sm mb-2">Full Configuration: {model}</h4>
      
      <div className="space-y-3">
        {/* Training parameters */}
        <div>
          <h5 className="text-xs font-semibold text-indigo-700 mb-1 border-b border-indigo-200 pb-1">
            Training Parameters
          </h5>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {['epochs', 'batch', 'patience', 'optimizer', 'lr0', 'lrf', 'cos_lr'].map(key => (
              <div key={key} className="flex justify-between">
                <span className="font-medium">{key}:</span>
                <span className="text-gray-600 ml-1">
                  {data[key] !== undefined 
                    ? (typeof data[key] === 'boolean' ? (data[key] ? 'yes' : 'no') : data[key]) 
                    : '-'}
                </span>
              </div>
            ))}
          </div>
        </div>
        
        {/* Model parameters */}
        <div>
          <h5 className="text-xs font-semibold text-green-700 mb-1 border-b border-green-200 pb-1">
            Model Parameters
          </h5>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {['imgsz', 'pretrained', 'single_cls', 'model', 'data'].map(key => (
              <div key={key} className="flex justify-between">
                <span className="font-medium">{key}:</span>
                <span className="text-gray-600 ml-1">
                  {data[key] !== undefined
                    ? (typeof data[key] === 'boolean' ? (data[key] ? 'yes' : 'no') : data[key])
                    : '-'}
                </span>
              </div>
            ))}
          </div>
        </div>
        
        {/* Augmentation parameters */}
        <div>
          <h5 className="text-xs font-semibold text-blue-700 mb-1 border-b border-blue-200 pb-1">
            Augmentation Parameters
          </h5>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {['mosaic', 'mixup', 'degrees', 'translate', 'scale', 'fliplr', 'hsv_h', 'hsv_s'].map(key => (
              <div key={key} className="flex justify-between">
                <span className="font-medium">{key}:</span>
                <span className="text-gray-600 ml-1">
                  {data[key] !== undefined ? data[key] : '-'}
                </span>
              </div>
            ))}
          </div>
        </div>
        
        {/* Loss parameters */}
        <div>
          <h5 className="text-xs font-semibold text-purple-700 mb-1 border-b border-purple-200 pb-1">
            Loss Parameters
          </h5>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {['box', 'cls', 'dfl'].map(key => (
              <div key={key} className="flex justify-between">
                <span className="font-medium">{key}:</span>
                <span className="text-gray-600 ml-1">
                  {data[key] !== undefined ? data[key] : '-'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const App = () => {
  // State
  const [originalImage, setOriginalImage] = useState(null);
  const [linkedImage, setLinkedImage] = useState(null);
  const [pitchedImage, setPitchedImage] = useState(null);
  const [originalZoom, setOriginalZoom] = useState(1);
  const [linkedZoom, setLinkedZoom] = useState(1);
  const [pitchedZoom, setPitchedZoom] = useState(1);
  const [detections, setDetections] = useState([]);
  const [pitchData, setPitchData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Model states
  const [availableModels] = useState(Object.keys(MOCK_MODEL_DATA));
  const [selectedModel, setSelectedModel] = useState(Object.keys(MOCK_MODEL_DATA)[0]);
  const [modelDetailsVisible, setModelDetailsVisible] = useState(true);
  const [modelCardsVisible, setModelCardsVisible] = useState(false);
  const [configVisible, setConfigVisible] = useState(false);
  
  // Other UI states
  const [showOriginal, setShowOriginal] = useState(true);
  const [showLinked, setShowLinked] = useState(true);
  const [showPitched, setShowPitched] = useState(true);
  const [showDetections, setShowDetections] = useState(false);
  const [showMusicXML, setShowMusicXML] = useState(true);
  const [currentFile, setCurrentFile] = useState(null);
  const [processingHistory, setProcessingHistory] = useState([]);
  const fileInputRef = useRef(null);

  // Attempt to fetch actual model info from the backend
  useEffect(() => {
    fetch('/api/available_models')
      .then(response => response.json())
      .then(models => {
        console.log("API models:", models);
      })
      .catch(err => {
        console.error('Error fetching models:', err);
      });
  }, []);

  // const processImage = async (file, model) => {
  //   if (!file) return;

  //   setIsLoading(true);
  //   setError(null);

  //   const formData = new FormData();
  //   formData.append('file', file);
  //   formData.append('model', model);

  //   try {
  //     const response = await fetch('/api/run_omr_pipeline', {
  //       method: 'POST',
  //       body: formData
  //     });

  //     const responseText = await response.text();
  //     console.log('Response text length:', responseText.length);
      
  //     try {
  //       const data = JSON.parse(responseText);
        
  //       // Detailed debugging information
  //       console.log('Response data keys:', Object.keys(data));
  //       console.log('Original image:', data.original_image ? `Present (length: ${data.original_image.length})` : 'Missing');
  //       console.log('Linked image:', data.linked_image ? `Present (length: ${data.linked_image.length})` : 'Missing');
  //       console.log('Pitched image:', data.pitched_image ? `Present (length: ${data.pitched_image.length})` : 'Missing');
  //       console.log('Processed image (legacy):', data.processed_image ? `Present (length: ${data.processed_image.length})` : 'Missing');
  //       console.log('Pitch data present:', data.pitch_data ? 'Yes' : 'No');
        
  //       // For backward compatibility with the old API response format
  //       const processedImage = data.processed_image;

  //       setOriginalImage(data.original_image);
        
  //       if (!data.linked_image && processedImage) {
  //         console.log('Using legacy processed_image as linked_image');
  //         setLinkedImage(processedImage);
  //       } else {
  //         setLinkedImage(data.linked_image);
  //       }
        
  //       setPitchedImage(data.pitched_image);
        
  //       // Extract detections from various possible sources
  //       let extractedDetections = [];
  //       if (data.detections && Array.isArray(data.detections)) {
  //         console.log('Using detections from root level');
  //         extractedDetections = data.detections;
  //       } else if (data.pitch_data && data.pitch_data.detections) {
  //         console.log('Using detections from pitch_data.detections');
  //         extractedDetections = data.pitch_data.detections;
  //       } else if (data.pitch_data && data.pitch_data.symbols) {
  //         console.log('Using detections from pitch_data.symbols');
  //         extractedDetections = data.pitch_data.symbols;
  //       }
                
  //       setDetections(extractedDetections.length > 0 ? extractedDetections : []);
  //       setPitchData(data.pitch_data || null);
        
  //       setProcessingHistory(prev => [
  //         ...prev, 
  //         {
  //           model,
  //           timestamp: new Date().toLocaleTimeString(),
  //           detectionCount: extractedDetections.length
  //         }
  //       ]);

  //       // Always make visualization containers visible even if images aren't available
  //       setShowOriginal(true);
  //       setShowLinked(true);
  //       setShowPitched(true);
        
  //     } catch (parseError) {
  //       console.error('JSON parse error:', parseError);
  //       setError('Failed to parse response from server');
  //     }
  //   } catch (err) {
  //     console.error('Network error:', err);
  //     setError(err.message || 'An unexpected error occurred');
  //   } finally {
  //     setIsLoading(false);
  //   }
  // };
  const processImage = async (file, model) => {
      if (!file) return;
    
      setIsLoading(true);
      setError(null);
    
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model', model);
    
      try {
        const response = await fetch('/api/run_omr_pipeline', {
          method: 'POST',
          body: formData
        });
    
        const responseText = await response.text();
        console.log('Response text length:', responseText.length);
        
        try {
          const data = JSON.parse(responseText);
          
          // Detailed debugging information
          console.log('Response data keys:', Object.keys(data));
          console.log('Original image:', data.original_image ? `Present (length: ${data.original_image.length})` : 'Missing');
          console.log('Linked image:', data.linked_image ? `Present (length: ${data.linked_image.length})` : 'Missing');
          console.log('Pitched image:', data.pitched_image ? `Present (length: ${data.pitched_image.length})` : 'Missing');
          console.log('Processed image (legacy):', data.processed_image ? `Present (length: ${data.processed_image.length})` : 'Missing');
          console.log('Pitch data present:', data.pitch_data ? 'Yes' : 'No');
          console.log('Base name:', data.base_name || 'Missing');
          
          // For backward compatibility with the old API response format
          const processedImage = data.processed_image;
    
          setOriginalImage(data.original_image);
          
          if (!data.linked_image && processedImage) {
            console.log('Using legacy processed_image as linked_image');
            setLinkedImage(processedImage);
          } else {
            setLinkedImage(data.linked_image);
          }
          
          setPitchedImage(data.pitched_image);
          
          // Extract detections from various possible sources
          let extractedDetections = [];
          if (data.detections && Array.isArray(data.detections)) {
            console.log('Using detections from root level');
            extractedDetections = data.detections;
          } else if (data.pitch_data && data.pitch_data.detections) {
            console.log('Using detections from pitch_data.detections');
            extractedDetections = data.pitch_data.detections;
          } else if (data.pitch_data && data.pitch_data.symbols) {
            console.log('Using detections from pitch_data.symbols');
            extractedDetections = data.pitch_data.symbols;
          }
                  
          setDetections(extractedDetections.length > 0 ? extractedDetections : []);
          setPitchData(data.pitch_data || null);
          
          // Save processing history including base_name
          setProcessingHistory(prev => [
            ...prev, 
            {
              model,
              timestamp: new Date().toLocaleTimeString(),
              detectionCount: extractedDetections.length,
              baseName: data.base_name // Store the base_name from the response
            }
          ]);
    
          // Always make visualization containers visible even if images aren't available
          setShowOriginal(true);
          setShowLinked(true);
          setShowPitched(true);
          
        } catch (parseError) {
          console.error('JSON parse error:', parseError);
          setError('Failed to parse response from server');
        }
      } catch (err) {
        console.error('Network error:', err);
        setError(err.message || 'An unexpected error occurred');
      } finally {
        setIsLoading(false);
      }
    };
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setCurrentFile(file);
    setOriginalImage(null);
    setLinkedImage(null);
    setPitchedImage(null);
    setDetections([]);
    setPitchData(null);
    setProcessingHistory([]);
    
    await processImage(file, selectedModel);
  };

  const handleReprocessWithModel = async (modelName) => {
    if (!currentFile) {
      setError("No image available to process");
      return;
    }
    
    setSelectedModel(modelName);
    await processImage(currentFile, modelName);
  };

  // Summarize detections by type
  const getSymbolStats = () => {
    if (!detections || detections.length === 0) return {};
    const typeCount = {};
    detections.forEach(d => {
      const className = d.class_name || d.class || 'Unknown';
      typeCount[className] = (typeCount[className] || 0) + 1;
    });
    return typeCount;
  };

  const symbolStats = getSymbolStats();

  return (
    <div className="max-w-6xl mx-auto p-4 bg-gray-50 min-h-screen">
      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <h1 className="text-2xl font-bold mb-2 text-center text-indigo-800">
          Optical Music Recognition Pipeline
        </h1>
        <p className="text-center text-gray-600 mb-4">
          Upload sheet music and process it with different detection models
        </p>

        {/* Model Selection Section */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-lg font-semibold text-gray-700">Model Selection</h2>
            <button 
              onClick={() => setModelDetailsVisible(!modelDetailsVisible)}
              className="text-xs text-indigo-600 hover:text-indigo-800"
            >
              {modelDetailsVisible ? "Hide Details" : "Show Details"}
            </button>
          </div>
          
          {modelDetailsVisible && (
            <div className="mb-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="flex-grow">
                  <select 
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500"
                  >
                    {availableModels.map(model => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                </div>
                <button
                  onClick={() => setModelCardsVisible(!modelCardsVisible)}
                  className="px-3 py-2 text-sm bg-indigo-100 text-indigo-700 rounded hover:bg-indigo-200"
                >
                  {modelCardsVisible ? "Hide Cards" : "Show Cards"}
                </button>
                <button
                  onClick={() => setConfigVisible(!configVisible)}
                  className="px-3 py-2 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200"
                >
                  Config
                </button>
              </div>
              
              {/* Full configuration details */}
              {configVisible && (
                <ModelDetails model={selectedModel} data={MOCK_MODEL_DATA[selectedModel]} />
              )}
              
              {/* Model cards in a scrollable container */}
              {modelCardsVisible && (
                <div className="overflow-x-auto pb-2 mt-2">
                  <div className="flex gap-2 pr-2">
                    {availableModels.map(model => (
                      <ModelCard 
                        key={model}
                        model={model}
                        data={MOCK_MODEL_DATA[model]}
                        isSelected={model === selectedModel}
                        onSelect={setSelectedModel}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* File Upload */}
              <div className="mt-4 p-3 bg-gray-50 rounded-md">
                <div className="flex items-center gap-2">
                  <div className="flex-grow">
                    <input 
                      ref={fileInputRef}
                      type="file" 
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="w-full p-1.5 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500"
                      disabled={isLoading || availableModels.length === 0}
                    />
                  </div>
                  
                  <div className="flex-shrink-0">
                    <span className="px-2 py-1 bg-indigo-100 text-indigo-800 text-sm rounded-md font-medium">
                      {selectedModel}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex justify-center items-center my-4 p-2">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-indigo-500"></div>
            <span className="ml-3 text-indigo-700">Processing with {selectedModel}...</span>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-3 rounded my-4" role="alert">
            <p className="font-bold">Error</p>
            <p>{error}</p>
          </div>
        )}
      </div>

      {/* Results Section */}
      {(originalImage || linkedImage || pitchedImage) && (
        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-xl font-bold mb-3 text-indigo-800">Results</h2>
          
          {/* Processing History */}
          {processingHistory.length > 0 && (
            <div className="mb-4">
              <h3 className="text-base font-semibold mb-2 text-gray-700">Processing History:</h3>
              <div className="overflow-x-auto pb-2">
                <div className="flex gap-2">
                  {processingHistory.map((item, index) => (
                    <div 
                      key={index} 
                      className="flex-shrink-0 px-2 py-1 rounded-full bg-indigo-100 text-indigo-800 text-sm cursor-pointer hover:bg-indigo-200"
                      onClick={() => handleReprocessWithModel(item.model)}
                    >
                      <span className="font-medium">{item.model}</span>
                      <span className="mx-1">•</span>
                      <span>{item.detectionCount}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Try with different model */}
          {currentFile && !isLoading && (
            <div className="mb-4 p-3 bg-indigo-50 rounded-md">
              <div className="flex items-center gap-3">
                <select 
                  className="p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 flex-grow"
                  value=""
                  onChange={(e) => {
                    if (e.target.value) {
                      handleReprocessWithModel(e.target.value);
                    }
                  }}
                >
                  <option value="">Try with a different model...</option>
                  {availableModels.filter(model => model !== selectedModel).map(model => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => handleReprocessWithModel(selectedModel)}
                  disabled={isLoading}
                  className="px-3 py-2 rounded-md text-sm font-medium bg-indigo-600 text-white hover:bg-indigo-700"
                >
                  Reprocess
                </button>
              </div>
            </div>
          )}

        {/* Display Controls with Image Availability Indicators */}
          <div className="flex flex-wrap gap-2 mb-4">
            <button 
              onClick={() => setShowOriginal(!showOriginal)}
              className={`px-3 py-1 rounded text-sm font-medium ${
                showOriginal 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {showOriginal ? "Hide" : "Show"} Original
              {originalImage ? "" : " (Missing)"}
            </button>
            
            <button 
              onClick={() => setShowLinked(!showLinked)}
              className={`px-3 py-1 rounded text-sm font-medium ${
                showLinked 
                  ? 'bg-green-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {showLinked ? "Hide" : "Show"} Linked
              {linkedImage ? "" : " (Missing)"}
            </button>
            
            <button 
              onClick={() => setShowPitched(!showPitched)}
              className={`px-3 py-1 rounded text-sm font-medium ${
                showPitched 
                  ? 'bg-purple-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {showPitched ? "Hide" : "Show"} Pitched
              {pitchedImage ? "" : " (Missing)"}
            </button>
            
            <button 
              onClick={() => setShowDetections(!showDetections)}
              className={`px-3 py-1 rounded text-sm font-medium ${
                showDetections 
                  ? 'bg-indigo-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {showDetections ? "Hide" : "Show"} Table
            </button>
            
            <button 
              onClick={() => setShowMusicXML(!showMusicXML)}
              className={`px-3 py-1 rounded text-sm font-medium ${
                showMusicXML 
                  ? 'bg-orange-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {showMusicXML ? "Hide" : "Show"} MusicXML
            </button>
          </div>

          {/* Debug Information Panel */}
          <div className="mb-4 p-3 bg-gray-100 rounded-md">
            <h3 className="text-sm font-semibold mb-2 text-gray-700">Image Diagnostics:</h3>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className={`p-1 rounded ${originalImage ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                Original: {originalImage ? '✓ Available' : '✗ Missing'}
              </div>
              <div className={`p-1 rounded ${linkedImage ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                Linked: {linkedImage ? '✓ Available' : '✗ Missing'}
              </div>
              <div className={`p-1 rounded ${pitchedImage ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                Pitched: {pitchedImage ? '✓ Available' : '✗ Missing'}
              </div>
            </div>
          </div>

          {/* Symbol Statistics */}
          {Object.keys(symbolStats).length > 0 && (
            <div className="mb-4 p-3 bg-gray-50 rounded-md">
              <h3 className="text-base font-semibold mb-2 text-gray-700">Symbol Distribution:</h3>
              <div className="overflow-x-auto pb-2">
                <div className="flex flex-wrap gap-2">
                  {Object.entries(symbolStats).map(([type, count], i) => (
                    <div key={i} className="px-2 py-1 bg-white border border-gray-200 rounded text-sm">
                      <span className="font-medium text-gray-800">{type}:</span>
                      <span className="ml-1 text-indigo-600 font-bold">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 
            Images Grid Layout - Now showing up to 3 images in a responsive grid
            If all 3 images are visible, they'll be shown in a 3-column layout on large screens
            If only 2 are visible, they'll be shown in a 2-column layout
            On smaller screens they'll stack
          */}
          <div className={`grid gap-4 mb-4 ${
            (showOriginal && showLinked && showPitched) 
              ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3' 
              : (showOriginal && (showLinked || showPitched)) 
                ? 'grid-cols-1 md:grid-cols-2' 
                : 'grid-cols-1'
          }`}>
            {/* Original Image Section */}
            {showOriginal && originalImage && (
              <div className="border-2 border-gray-300 rounded-lg overflow-hidden shadow-md">
                <div className="bg-blue-100 px-3 py-1 border-b border-blue-300">
                  <h3 className="font-medium text-sm text-blue-700">Original Image</h3>
                </div>
                <div className="flex items-center justify-center h-[400px] bg-gray-50">
                  <img 
                    src={originalImage}
                    alt="Original" 
                    className="max-h-full max-w-full object-contain"
                    style={{ transformOrigin: 'center', transform: `scale(${originalZoom})` }}
                    onError={(e) => {
                      console.error("Error loading original image");
                      e.target.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
                      e.target.alt = "Failed to load image";
                    }}
                  />
                </div>
                <div className="flex items-center justify-center p-2 space-x-2 bg-blue-100 border-t border-blue-300">
                  <button 
                    onClick={() => setOriginalZoom(prev => Math.max(prev - 0.25, 0.5))}
                    className="px-2 py-1 text-xs bg-blue-200 text-blue-700 rounded hover:bg-blue-300"
                  >
                    −
                  </button>
                  <span className="text-xs text-blue-700">{Math.round(originalZoom * 100)}%</span>
                  <button 
                    onClick={() => setOriginalZoom(prev => Math.min(prev + 0.25, 3))}
                    className="px-2 py-1 text-xs bg-blue-200 text-blue-700 rounded hover:bg-blue-300"
                  >
                    +
                  </button>
                  <button 
                    onClick={() => setOriginalZoom(1)}
                    className="px-2 py-1 text-xs bg-blue-200 text-blue-700 rounded hover:bg-blue-300"
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}

            {/* Linked Image Section */}
            {showLinked && (
              <div className="border-2 border-gray-300 rounded-lg overflow-hidden shadow-md">
                <div className="bg-green-100 px-3 py-1 border-b border-green-300 flex justify-between">
                  <h3 className="font-medium text-sm text-green-700">Linked Visualization</h3>
                  {!linkedImage && <span className="text-xs text-red-600">Image unavailable</span>}
                </div>
                <div className="flex items-center justify-center h-[400px] bg-gray-50">
                  {linkedImage ? (
                    <img 
                      src={linkedImage}
                      alt="Linked Visualization" 
                      className="max-h-full max-w-full object-contain"
                      style={{ transformOrigin: 'center', transform: `scale(${linkedZoom})` }}
                      onError={(e) => {
                        console.error("Error loading linked image");
                        e.target.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
                        e.target.alt = "Failed to load image";
                      }}
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center text-gray-500">
                      <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                      </svg>
                      <p className="mt-2 text-sm">No linked visualization available</p>
                    </div>
                  )}
                </div>
                <div className="flex items-center justify-center p-2 space-x-2 bg-green-100 border-t border-green-300">
                  <button 
                    onClick={() => setLinkedZoom(prev => Math.max(prev - 0.25, 0.5))}
                    className={`px-2 py-1 text-xs ${linkedImage ? 'bg-green-200 text-green-700 hover:bg-green-300' : 'bg-gray-200 text-gray-500 cursor-not-allowed'} rounded`}
                    disabled={!linkedImage}
                  >
                    −
                  </button>
                  <span className="text-xs text-green-700">{Math.round(linkedZoom * 100)}%</span>
                  <button 
                    onClick={() => setLinkedZoom(prev => Math.min(prev + 0.25, 3))}
                    className={`px-2 py-1 text-xs ${linkedImage ? 'bg-green-200 text-green-700 hover:bg-green-300' : 'bg-gray-200 text-gray-500 cursor-not-allowed'} rounded`}
                    disabled={!linkedImage}
                  >
                    +
                  </button>
                  <button 
                    onClick={() => setLinkedZoom(1)}
                    className={`px-2 py-1 text-xs ${linkedImage ? 'bg-green-200 text-green-700 hover:bg-green-300' : 'bg-gray-200 text-gray-500 cursor-not-allowed'} rounded`}
                    disabled={!linkedImage}
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}

            {/* Pitched Image Section */}
            {showPitched && (
              <div className="border-2 border-gray-300 rounded-lg overflow-hidden shadow-md">
                <div className="bg-purple-100 px-3 py-1 border-b border-purple-300 flex justify-between">
                  <h3 className="font-medium text-sm text-purple-700">Pitched Visualization</h3>
                  {!pitchedImage && <span className="text-xs text-red-600">Image unavailable</span>}
                </div>
                <div className="flex items-center justify-center h-[400px] bg-gray-50">
                  {pitchedImage ? (
                    <img 
                      src={pitchedImage}
                      alt="Pitched Visualization" 
                      className="max-h-full max-w-full object-contain"
                      style={{ transformOrigin: 'center', transform: `scale(${pitchedZoom})` }}
                      onError={(e) => {
                        console.error("Error loading pitched image");
                        e.target.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
                        e.target.alt = "Failed to load image";
                      }}
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center text-gray-500">
                      <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                      </svg>
                      <p className="mt-2 text-sm">No pitched visualization available</p>
                    </div>
                  )}
                </div>
                <div className="flex items-center justify-center p-2 space-x-2 bg-purple-100 border-t border-purple-300">
                  <button 
                    onClick={() => setPitchedZoom(prev => Math.max(prev - 0.25, 0.5))}
                    className={`px-2 py-1 text-xs ${pitchedImage ? 'bg-purple-200 text-purple-700 hover:bg-purple-300' : 'bg-gray-200 text-gray-500 cursor-not-allowed'} rounded`}
                    disabled={!pitchedImage}
                  >
                    −
                  </button>
                  <span className="text-xs text-purple-700">{Math.round(pitchedZoom * 100)}%</span>
                  <button 
                    onClick={() => setPitchedZoom(prev => Math.min(prev + 0.25, 3))}
                    className={`px-2 py-1 text-xs ${pitchedImage ? 'bg-purple-200 text-purple-700 hover:bg-purple-300' : 'bg-gray-200 text-gray-500 cursor-not-allowed'} rounded`}
                    disabled={!pitchedImage}
                  >
                    +
                  </button>
                  <button 
                    onClick={() => setPitchedZoom(1)}
                    className={`px-2 py-1 text-xs ${pitchedImage ? 'bg-purple-200 text-purple-700 hover:bg-purple-300' : 'bg-gray-200 text-gray-500 cursor-not-allowed'} rounded`}
                    disabled={!pitchedImage}
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Detections Table */}
          {showDetections && detections.length > 0 && (
            <div className="border border-gray-200 rounded-lg overflow-hidden shadow-md mb-4">
              <div className="bg-gray-100 px-3 py-1 border-b flex justify-between items-center">
                <h3 className="font-medium text-sm text-gray-700">
                  Detected Symbols ({detections.length})
                </h3>
                <button
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(detections, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'detections.json';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                  }}
                  className="px-2 py-1 bg-indigo-600 text-white text-xs rounded hover:bg-indigo-700"
                >
                  Download JSON
                </button>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full bg-white text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">#</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Pitch</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Conf.</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Details</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {detections.slice(0, 20).map((detection, index) => {
                      try {
                        const className = detection.class_name || detection.class || 'Unknown';
                        const pitch = detection.pitch || 'N/A';
                        const confidence = detection.confidence !== undefined 
                          ? (typeof detection.confidence === 'number' 
                            ? (detection.confidence * 100).toFixed(1) 
                            : String(detection.confidence)) + '%'
                          : 'N/A';
                        
                        const details = [];
                        if (detection.staff_assignment !== undefined) {
                          details.push(`Staff: ${detection.staff_assignment}`);
                        }
                        if (detection.line_number !== undefined) {
                          details.push(`Line: ${detection.line_number}`);
                        }
                        if (detection.linked_symbols) {
                          details.push(`Links: ${detection.linked_symbols.length}`);
                        }
                        
                        return (
                          <tr key={index} className="hover:bg-gray-50">
                            <td className="px-3 py-1.5 whitespace-nowrap text-xs text-gray-900">{index + 1}</td>
                            <td className="px-3 py-1.5 whitespace-nowrap text-xs font-medium text-gray-900">
                              {className}
                            </td>
                            <td className="px-3 py-1.5 whitespace-nowrap text-xs text-gray-600">
                              {pitch}
                            </td>
                            <td className="px-3 py-1.5 whitespace-nowrap text-xs text-gray-600">
                              {confidence}
                            </td>
                            <td className="px-3 py-1.5 whitespace-nowrap text-xs text-gray-600">
                              {details.join(', ') || 'N/A'}
                            </td>
                          </tr>
                        );
                      } catch (err) {
                        console.error(`Error rendering detection ${index}:`, err);
                        return (
                          <tr key={index} className="bg-red-50">
                            <td colSpan="5" className="px-3 py-1.5 text-center text-red-600 text-xs">
                              Error displaying detection #{index + 1}
                            </td>
                          </tr>
                        );
                      }
                    })}
                    {detections.length > 20 && (
                      <tr>
                        <td colSpan="5" className="px-3 py-2 text-center text-xs text-gray-500">
                          Showing 20 of {detections.length} detections
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          
          {/* MusicXML Viewer Section (updated) */}
          {showMusicXML && (
            <div className="mb-4">
              <div className="bg-orange-100 px-3 py-2 border border-orange-300 rounded-t">
                <h3 className="font-medium text-sm text-orange-700">MusicXML Notation Preview</h3>
              </div>
              <div className="border border-orange-300 border-t-0 rounded-b">
                <MusicXMLViewer 
                  pitchData={pitchData} 
                  baseName={pitchData ? processingHistory[processingHistory.length - 1]?.baseName : null} 
                />
              </div>
            </div>
          )}
          {/* {showMusicXML && pitchData && (
            <div className="mb-4">
              <div className="bg-orange-100 px-3 py-2 border border-orange-300 rounded-t">
                <h3 className="font-medium text-sm text-orange-700">MusicXML Notation Preview</h3>
              </div>
              <div className="border border-orange-300 border-t-0 rounded-b">
                <MusicXMLViewer pitchData={pitchData} />
              </div>
            </div>
          )} */}
        </div>
      )}
      
      <footer className="text-center text-gray-500 text-xs mt-6 mb-2">
        Optical Music Recognition Pipeline © 2025
      </footer>
    </div>
  );
};

export default App;