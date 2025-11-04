const { useState, useRef, useEffect } = React;

function MNISTDrawer() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [predictedDigit, setPredictedDigit] = useState(null);
  const [serverUrl] = useState('https://api.digitimagepredictor.com/predict');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const setupCanvasContext = (canvas) => {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';

    const originalWidth = 380;
    const originalLineWidth = 20;
    ctx.lineWidth = (canvas.width / originalWidth) * originalLineWidth;

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      const { clientWidth, clientHeight } = canvas;

      if (canvas.width !== clientWidth || canvas.height !== clientHeight) {
        canvas.width = clientWidth;
        canvas.height = clientHeight;
        setupCanvasContext(canvas);
      }
    };

    resizeCanvas();

    window.addEventListener('resize', resizeCanvas);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  const getCoordinates = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX, clientY;
    if (e.touches && e.touches[0]) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  };

  const startDrawing = (e) => {
    e.preventDefault();
    setIsDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { x, y } = getCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { x, y } = getCoordinates(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = (e) => {
    e.preventDefault();
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    setupCanvasContext(canvas);
    setPredictedDigit(null);
    setError(null);
  };

  // MNIST prep functions

  const getBoundingBox = (imageData, width, height) => {
    let minX = width, minY = height, maxX = 0, maxY = 0;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const avg = (imageData[i] + imageData[i + 1] + imageData[i + 2]) / 3;
        if (avg < 250) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }

    return { minX, minY, maxX, maxY };
  };

  const getCenterOfMass = (imageData, width, height, bbox) => {
    let totalMass = 0;
    let comX = 0;
    let comY = 0;
    for (let y = bbox.minY; y <= bbox.maxY; y++) {
      for (let x = bbox.minX; x <= bbox.maxX; x++) {
        const i = (y * width + x) * 4;
        const avg = (imageData[i] + imageData[i + 1] + imageData[i + 2]) / 3;
        const mass = 255 - avg;
        totalMass += mass;
        comX += x * mass;
        comY += y * mass;
      }
    }

    if (totalMass === 0) return { x: width / 2, y: height / 2 };

    return { x: comX / totalMass, y: comY / totalMass };
  };

  const applyGaussianBlur = (imageData, width, height) => {
    const output = new Uint8ClampedArray(imageData.length);

    const kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
    const kernelSum = 16;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const py = Math.min(Math.max(y + ky, 0), height - 1);
            const px = Math.min(Math.max(x + kx, 0), width - 1);
            const i = (py * width + px) * 4;
            const weight = kernel[ky + 1][kx + 1];
            sum += imageData[i] * weight;
          }
        }
        const i = (y * width + x) * 4;
        const blurred = sum / kernelSum;
        output[i] = output[i + 1] = output[i + 2] = blurred;
        output[i + 3] = 255;
      }
    }

    return output;
  };

  const getImageData = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const drawingData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const bbox = getBoundingBox(drawingData.data, canvas.width, canvas.height);

    const drawnWidth = bbox.maxX - bbox.minX + 1;
    const drawnHeight = bbox.maxY - bbox.minY + 1;

    const tempCanvas = document.createElement('canvas');

    tempCanvas.width = drawnWidth;
    tempCanvas.height = drawnHeight;

    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(
      ctx.getImageData(bbox.minX, bbox.minY, drawnWidth, drawnHeight),
      0, 0
    );

    const croppedData = tempCtx.getImageData(0, 0, drawnWidth, drawnHeight);
    const com = getCenterOfMass(croppedData.data, drawnWidth, drawnHeight,
      { minX: 0, minY: 0, maxX: drawnWidth - 1, maxY: drawnHeight - 1 });

    const scale = Math.min(20 / drawnWidth, 20 / drawnHeight);
    const scaledWidth = drawnWidth * scale;
    const scaledHeight = drawnHeight * scale;

    const intermediateCanvas = document.createElement('canvas');
    intermediateCanvas.width = scaledWidth;
    intermediateCanvas.height = scaledHeight;

    const intermediateCtx = intermediateCanvas.getContext('2d');
    intermediateCtx.imageSmoothingEnabled = true;
    intermediateCtx.imageSmoothingQuality = 'high';
    intermediateCtx.drawImage(tempCanvas, 0, 0, scaledWidth, scaledHeight);

    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = 28;
    finalCanvas.height = 28;

    const finalCtx = finalCanvas.getContext('2d');
    finalCtx.fillStyle = 'white';
    finalCtx.fillRect(0, 0, 28, 28);

    const comXScaled = com.x * scale;
    const comYScaled = com.y * scale;
    const offsetX = 14 - comXScaled;
    const offsetY = 14 - comYScaled;
    finalCtx.drawImage(intermediateCanvas, offsetX, offsetY);
    let imageData = finalCtx.getImageData(0, 0, 28, 28);

    const blurredData1 = applyGaussianBlur(imageData.data, 28, 28);
    imageData.data.set(blurredData1);
    finalCtx.putImageData(imageData, 0, 0);
    imageData = finalCtx.getImageData(0, 0, 28, 28);

    const blurredData2 = applyGaussianBlur(imageData.data, 28, 28);
    imageData.data.set(blurredData2);
    finalCtx.putImageData(imageData, 0, 0);
    imageData = finalCtx.getImageData(0, 0, 28, 28);

    const pixels = imageData.data;
    const normalized = [];

    for (let i = 0; i < pixels.length; i += 4) {
      const avg = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
      normalized.push(1.0 - (avg / 255.0));
    }

    return normalized;
  };

  const sendToServer = async () => {
    setIsLoading(true);
    setError(null);
    setPredictedDigit(null);
    try {
      const imageData = getImageData();
      const response = await fetch(serverUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData }),
      });
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      const result = await response.json();
      setPredictedDigit(result.prediction ?? result.digit ?? result);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 flex items-center justify-center p-4 font-sans">
      <div className="w-full max-w-md">
        <div className="text-center mb-4">
          <h1 className="text-4xl font-extrabold text-gray-100 mb-3 tracking-tight">
            Digit Image Predictor
          </h1>
          <p className="text-gray-100 text-lg">
            Draw a single number from 0-9 below
          </p>
        </div>

        <div className="bg-gray-200 backdrop-blur-sm rounded-3xl shadow-2xl p-6">
          <div className="mb-4 flex justify-center">
            <canvas
              ref={canvasRef}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              onTouchStart={startDrawing}
              onTouchMove={draw}
              onTouchEnd={stopDrawing}
              className="rounded-2xl cursor-crosshair shadow-inner bg-white touch-none w-full aspect-square"
              style={{ touchAction: 'none' }}
            />
          </div>

          <div className="flex flex-col sm:flex-row gap-3 mb-4">
            <button
              onClick={clearCanvas}
              className="flex-1 px-6 py-3 bg-gray-200 text-gray-900 rounded-xl hover:bg-gray-300 active:bg-gray-400 transition-all font-semibold text-base shadow-sm"
            >
              Clear
            </button>
            <button
              onClick={sendToServer}
              disabled={isLoading}
              className="flex-1 px-6 py-3 bg-blue-900 text-gray-100 rounded-xl hover:bg-blue-700 active:bg-blue-800 transition-all font-semibold text-base shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Predicting...' : 'Predict'}
            </button>
          </div>

          <div className="min-h-40">
            {error && (
              <div className="p-4 bg-red-50 border-2 border-red-200 rounded-xl text-red-700 text-center">
                <strong>Error:</strong> {error}
              </div>
            )}

            {predictedDigit !== null && !error && (
              <div className="text-center py-6">
                <p className="text-gray-900 text-xl mb-2">You drew:</p>
                <p className="text-7xl font-extrabold text-transparent bg-clip-text bg-blue-900">
                  {predictedDigit}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

ReactDOM.render(<MNISTDrawer />, document.getElementById('root'));