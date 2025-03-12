// import React, { useState, useEffect, useRef } from 'react';

// const MusicXMLViewer = ({ pitchData }) => {
//   const [musicXML, setMusicXML] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [timeSignature, setTimeSignature] = useState('4/4');
//   const [beamAll, setBeamAll] = useState(false);
//   const iframeRef = useRef(null);
  
//   const convertToMusicXML = async () => {
//     if (!pitchData) {
//       setError('No pitch data available for conversion');
//       return;
//     }
    
//     setLoading(true);
//     setError(null);
    
//     try {
//       const response = await fetch('/api/convert_to_musicxml', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//           json_data: pitchData,
//           time_signature: timeSignature,
//           beam_all: beamAll
//         })
//       });
      
//       if (!response.ok) {
//         throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
//       }
      
//       const data = await response.json();
//       setMusicXML(data.musicxml);
//     } catch (err) {
//       console.error('Error converting to MusicXML:', err);
//       setError(err.message || 'Failed to convert to MusicXML');
//     } finally {
//       setLoading(false);
//     }
//   };
  
//   const downloadMusicXML = () => {
//     if (!musicXML) return;
    
//     const blob = new Blob([musicXML], { type: 'application/vnd.recordare.musicxml+xml' });
//     const url = URL.createObjectURL(blob);
//     const a = document.createElement('a');
//     a.href = url;
//     a.download = 'score.musicxml';
//     document.body.appendChild(a);
//     a.click();
//     document.body.removeChild(a);
//     URL.revokeObjectURL(url);
//   };
  
//   // Load Verovio Toolkit for rendering MusicXML
//   useEffect(() => {
//     if (!musicXML) return;
    
//     const script = document.createElement('script');
//     script.src = 'https://www.verovio.org/javascript/latest/verovio-toolkit.js';
//     script.async = true;
    
//     script.onload = () => {
//       try {
//         // Initialize Verovio toolkit
//         const tk = new window.verovio.toolkit();
        
//         // Set options for rendering
//         const options = {
//           scale: 40,
//           adjustPageHeight: 1,
//           pageWidth: iframeRef.current.clientWidth,
//           pageHeight: 2000,
//           footer: 'none',
//           unit: 'px'
//         };
        
//         tk.setOptions(options);
        
//         // Load MusicXML and render to SVG
//         tk.loadData(musicXML);
//         const svg = tk.renderToSVG(1);
        
//         // Display in iframe
//         const iframeDocument = iframeRef.current.contentDocument;
//         iframeDocument.open();
//         iframeDocument.write(`
//           <!DOCTYPE html>
//           <html>
//             <head>
//               <style>
//                 body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
//                 svg { width: 100%; height: auto; }
//               </style>
//             </head>
//             <body>${svg}</body>
//           </html>
//         `);
//         iframeDocument.close();
//       } catch (err) {
//         console.error('Error rendering MusicXML:', err);
//         setError('Failed to render MusicXML. The converter may have generated invalid markup.');
//       }
//     };
    
//     script.onerror = () => {
//       setError('Failed to load the Verovio toolkit');
//     };
    
//     document.body.appendChild(script);
    
//     return () => {
//       document.body.removeChild(script);
//     };
//   }, [musicXML]);
  
//   return (
//     <div className="music-xml-section bg-white p-6 rounded-lg shadow-md mb-6">
//       <h2 className="text-xl font-bold mb-3 text-indigo-800">MusicXML Export</h2>
      
//       <div className="bg-gray-50 p-4 rounded-md mb-4">
//         <div className="grid grid-cols-2 gap-4 mb-4">
//           <div>
//             <label className="block text-sm font-medium text-gray-700 mb-1">
//               Time Signature
//             </label>
//             <input 
//               type="text" 
//               value={timeSignature} 
//               onChange={(e) => setTimeSignature(e.target.value)}
//               placeholder="e.g. 4/4, 3/4"
//               className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500"
//             />
//           </div>
          
//           <div className="flex items-center">
//             <label className="flex items-center cursor-pointer">
//               <input 
//                 type="checkbox"
//                 checked={beamAll}
//                 onChange={(e) => setBeamAll(e.target.checked)}
//                 className="w-4 h-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
//               />
//               <span className="ml-2 text-sm text-gray-700">
//                 Treat all beamed notes as 32nd notes
//               </span>
//             </label>
//           </div>
//         </div>
        
//         <div className="flex gap-2">
//           <button 
//             onClick={convertToMusicXML} 
//             disabled={loading || !pitchData}
//             className={`px-4 py-2 rounded-md text-sm font-medium ${
//               loading || !pitchData
//                 ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
//                 : 'bg-indigo-600 text-white hover:bg-indigo-700'
//             }`}
//           >
//             {loading ? (
//               <span className="flex items-center">
//                 <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
//                   <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
//                   <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
//                 </svg>
//                 Converting...
//               </span>
//             ) : 'Convert to MusicXML'}
//           </button>
          
//           {musicXML && (
//             <button 
//               onClick={downloadMusicXML}
//               className="px-4 py-2 rounded-md text-sm font-medium bg-green-600 text-white hover:bg-green-700"
//             >
//               Download MusicXML
//             </button>
//           )}
//         </div>
//       </div>
      
//       {error && (
//         <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded mb-4">
//           <p className="font-bold">Error</p>
//           <p>{error}</p>
//         </div>
//       )}
      
//       {musicXML && (
//         <div className="mt-4">
//           <h3 className="text-lg font-semibold mb-2 text-gray-700">Score Preview</h3>
//           <div className="border border-gray-300 rounded-md overflow-hidden">
//             <iframe 
//               ref={iframeRef}
//               style={{ width: '100%', height: '500px', border: 'none' }}
//               title="MusicXML Preview"
//             />
//           </div>
//           <div className="mt-2 text-sm text-gray-500">
//             <p>Download the MusicXML file to open in music notation software like:</p>
//             <ul className="list-disc list-inside ml-2 mt-1">
//               <li><a href="https://musescore.org/en/download" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">MuseScore</a> (Free, desktop app)</li>
//               <li><a href="https://flat.io" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">Flat.io</a> (Free tier, web-based)</li>
//             </ul>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default MusicXMLViewer;
import React, { useState, useEffect, useRef } from 'react';

const MusicXMLViewer = ({ pitchData, baseName }) => {
  const [musicXML, setMusicXML] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [timeSignature, setTimeSignature] = useState('4/4');
  const [beamAll, setBeamAll] = useState(false);
  const iframeRef = useRef(null);
  
  const convertToMusicXML = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Use baseName if available, otherwise fallback to sending pitch data directly
      const requestBody = baseName 
        ? {
            base_name: baseName,
            time_signature: timeSignature,
            beam_all: beamAll
          }
        : {
            json_data: pitchData || {},
            time_signature: timeSignature,
            beam_all: beamAll
          };
      
      console.log('Sending MusicXML conversion request:', requestBody);
      
      const response = await fetch('/api/convert_to_musicxml', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.error || 
          `Server responded with ${response.status}: ${response.statusText}`
        );
      }
      
      const data = await response.json();
      if (!data.musicxml) {
        throw new Error('Server returned a response without MusicXML data');
      }
      
      setMusicXML(data.musicxml);
    } catch (err) {
      console.error('Error converting to MusicXML:', err);
      setError(err.message || 'Failed to convert to MusicXML');
    } finally {
      setLoading(false);
    }
  };
  
  const downloadMusicXML = () => {
    if (!musicXML) return;
    
    const blob = new Blob([musicXML], { type: 'application/vnd.recordare.musicxml+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'score.musicxml';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  // Load Verovio Toolkit for rendering MusicXML
  useEffect(() => {
    if (!musicXML) return;
    
    const script = document.createElement('script');
    script.src = 'https://www.verovio.org/javascript/latest/verovio-toolkit.js';
    script.async = true;
    
    script.onload = () => {
      try {
        // Initialize Verovio toolkit
        const tk = new window.verovio.toolkit();
        
        // Set options for rendering
        const options = {
          scale: 40,
          adjustPageHeight: 1,
          pageWidth: iframeRef.current.clientWidth,
          pageHeight: 2000,
          footer: 'none',
          unit: 'px'
        };
        
        tk.setOptions(options);
        
        // Load MusicXML and render to SVG
        tk.loadData(musicXML);
        const svg = tk.renderToSVG(1);
        
        // Display in iframe
        const iframeDocument = iframeRef.current.contentDocument;
        iframeDocument.open();
        iframeDocument.write(`
          <!DOCTYPE html>
          <html>
            <head>
              <style>
                body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
                svg { width: 100%; height: auto; }
              </style>
            </head>
            <body>${svg}</body>
          </html>
        `);
        iframeDocument.close();
      } catch (err) {
        console.error('Error rendering MusicXML:', err);
        setError('Failed to render MusicXML. The converter may have generated invalid markup.');
      }
    };
    
    script.onerror = () => {
      setError('Failed to load the Verovio toolkit');
    };
    
    document.body.appendChild(script);
    
    return () => {
      document.body.removeChild(script);
    };
  }, [musicXML]);
  
  return (
    <div className="music-xml-section bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-bold mb-3 text-indigo-800">MusicXML Export</h2>
      
      <div className="bg-gray-50 p-4 rounded-md mb-4">
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Time Signature
            </label>
            <input 
              type="text" 
              value={timeSignature} 
              onChange={(e) => setTimeSignature(e.target.value)}
              placeholder="e.g. 4/4, 3/4"
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          
          <div className="flex items-center">
            <label className="flex items-center cursor-pointer">
              <input 
                type="checkbox"
                checked={beamAll}
                onChange={(e) => setBeamAll(e.target.checked)}
                className="w-4 h-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
              />
              <span className="ml-2 text-sm text-gray-700">
                Treat all beamed notes as 32nd notes
              </span>
            </label>
          </div>
        </div>
        
        <div className="flex gap-2">
          <button 
            onClick={convertToMusicXML} 
            disabled={loading || (!pitchData && !baseName)}
            className={`px-4 py-2 rounded-md text-sm font-medium ${
              loading || (!pitchData && !baseName)
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-indigo-600 text-white hover:bg-indigo-700'
            }`}
          >
            {loading ? (
              <span className="flex items-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Converting...
              </span>
            ) : 'Convert to MusicXML'}
          </button>
          
          {musicXML && (
            <button 
              onClick={downloadMusicXML}
              className="px-4 py-2 rounded-md text-sm font-medium bg-green-600 text-white hover:bg-green-700"
            >
              Download MusicXML
            </button>
          )}
        </div>
      </div>
      
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded mb-4">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      )}
      
      {musicXML && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2 text-gray-700">Score Preview</h3>
          <div className="border border-gray-300 rounded-md overflow-hidden">
            <iframe 
              ref={iframeRef}
              style={{ width: '100%', height: '500px', border: 'none' }}
              title="MusicXML Preview"
            />
          </div>
          <div className="mt-2 text-sm text-gray-500">
            <p>Download the MusicXML file to open in music notation software like:</p>
            <ul className="list-disc list-inside ml-2 mt-1">
              <li><a href="https://musescore.org/en/download" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">MuseScore</a> (Free, desktop app)</li>
              <li><a href="https://flat.io" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">Flat.io</a> (Free tier, web-based)</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default MusicXMLViewer;